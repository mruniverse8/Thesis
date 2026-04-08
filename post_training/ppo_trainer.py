from __future__ import annotations

import math
from pathlib import Path
from typing import Sequence

import torch
import torch.nn.functional as F

from reward_utils.defaults import RewardConfig
from src.io_utils import ensure_dir, write_json, write_jsonl

from .policy_model import PolicyValueModel
from .ppo_types import PPOConfig, StageTrajectory
from .reward_adapter import summarize_reward_breakdowns
from .rollout import (
    compute_action_stats,
    encode_decoder_prefix,
    encode_prompt,
    sample_rollout_for_example,
)


def standardize_tensor(values: torch.Tensor, eps: float = 1.0e-8) -> torch.Tensor:
    if values.numel() <= 1:
        return torch.zeros_like(values)
    return (values - values.mean()) / values.std(unbiased=False).clamp(min=eps)


def compute_clipped_policy_objective(
    ratio: torch.Tensor,
    advantages: torch.Tensor,
    clip_range: float,
) -> torch.Tensor:
    clipped_ratio = torch.clamp(ratio, 1.0 - clip_range, 1.0 + clip_range)
    return torch.minimum(ratio * advantages, clipped_ratio * advantages)


class MoleculeWisePPOTrainer:
    def __init__(
        self,
        *,
        policy_model: PolicyValueModel,
        reference_model: torch.nn.Module,
        tokenizer,
        config: PPOConfig,
        reward_config: RewardConfig | None = None,
        device: torch.device | None = None,
    ) -> None:
        self.policy_model = policy_model
        self.reference_model = reference_model
        self.tokenizer = tokenizer
        self.config = config
        self.reward_config = reward_config
        self.device = device or next(policy_model.parameters()).device
        self.policy_model.to(self.device)
        self.reference_model.to(self.device)
        self.reference_model.eval()

        trainable_parameters = [
            parameter for parameter in self.policy_model.parameters() if parameter.requires_grad
        ]
        self.optimizer = torch.optim.AdamW(
            trainable_parameters,
            lr=self.config.learning_rate,
        )

    def collect_rollouts(self, examples: Sequence[dict[str, object]]) -> list[StageTrajectory]:
        trajectories: list[StageTrajectory] = []
        for example in examples:
            trajectories.extend(
                sample_rollout_for_example(
                    self.policy_model,
                    self.reference_model,
                    self.tokenizer,
                    example,
                    generation_config=self.config.rollout,
                    reward_config=self.reward_config,
                    device=self.device,
                )
            )
        return trajectories

    def _compute_stage_statistics(
        self,
        trajectory: StageTrajectory,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        prompt_inputs = encode_prompt(
            self.tokenizer,
            trajectory.prompt_text,
            max_source_length=self.config.rollout.max_source_length,
            device=self.device,
        )
        decoder_prefix_ids = encode_decoder_prefix(
            self.tokenizer,
            trajectory.decoder_prefix_text,
            decoder_start_token_id=int(self.policy_model.policy_model.config.decoder_start_token_id),
            device=self.device,
        )

        logprob_sum, entropy_sum = compute_action_stats(
            self.policy_model.policy_model,
            input_ids=prompt_inputs["input_ids"],
            attention_mask=prompt_inputs["attention_mask"],
            decoder_input_ids=decoder_prefix_ids,
            action_token_ids=trajectory.action_token_ids,
        )
        value_prediction = self.policy_model.compute_values(
            input_ids=prompt_inputs["input_ids"],
            attention_mask=prompt_inputs["attention_mask"],
            decoder_input_ids=decoder_prefix_ids,
        ).squeeze(0)
        return logprob_sum, entropy_sum, value_prediction

    def train_iteration(
        self,
        examples: Sequence[dict[str, object]],
        *,
        iteration_index: int,
    ) -> dict[str, float]:
        trajectories = self.collect_rollouts(examples)
        if not trajectories:
            return {"num_stage_trajectories": 0.0}

        rewards = torch.tensor(
            [trajectory.reward for trajectory in trajectories],
            dtype=torch.float32,
            device=self.device,
        )
        old_logprobs = torch.tensor(
            [trajectory.action_logprob_sum_old for trajectory in trajectories],
            dtype=torch.float32,
            device=self.device,
        )
        reference_logprobs = torch.tensor(
            [trajectory.reference_logprob_sum for trajectory in trajectories],
            dtype=torch.float32,
            device=self.device,
        )
        old_values = torch.tensor(
            [trajectory.value_old for trajectory in trajectories],
            dtype=torch.float32,
            device=self.device,
        )
        returns = rewards
        advantages = standardize_tensor(rewards - old_values)

        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_kl = 0.0
        total_entropy = 0.0
        num_optimizer_steps = 0

        for _ in range(self.config.ppo_epochs_per_batch):
            permutation = torch.randperm(len(trajectories))
            for start in range(0, len(trajectories), self.config.mini_batch_size):
                batch_indices = permutation[start : start + self.config.mini_batch_size]
                minibatch = [trajectories[index] for index in batch_indices.tolist()]

                new_logprobs: list[torch.Tensor] = []
                new_entropies: list[torch.Tensor] = []
                new_values: list[torch.Tensor] = []
                for trajectory in minibatch:
                    logprob_sum, entropy_sum, value_prediction = self._compute_stage_statistics(
                        trajectory
                    )
                    new_logprobs.append(logprob_sum)
                    new_entropies.append(entropy_sum)
                    new_values.append(value_prediction)

                new_logprobs_tensor = torch.stack(new_logprobs)
                new_entropies_tensor = torch.stack(new_entropies)
                new_values_tensor = torch.stack(new_values)
                batch_old_logprobs = old_logprobs[batch_indices]
                batch_reference_logprobs = reference_logprobs[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_returns = returns[batch_indices]

                ratio = torch.exp(new_logprobs_tensor - batch_old_logprobs)
                clipped_objective = compute_clipped_policy_objective(
                    ratio,
                    batch_advantages,
                    clip_range=self.config.clip_range,
                )
                kl = new_logprobs_tensor - batch_reference_logprobs
                policy_loss = -(clipped_objective - self.config.kl_penalty * kl).mean()
                value_loss = F.mse_loss(new_values_tensor, batch_returns)
                entropy_bonus = new_entropies_tensor.mean()
                loss = (
                    policy_loss
                    + self.config.value_loss_coef * value_loss
                    - self.config.entropy_coef * entropy_bonus
                )

                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.policy_model.parameters(),
                    self.config.max_grad_norm,
                )
                self.optimizer.step()

                total_policy_loss += float(policy_loss.item())
                total_value_loss += float(value_loss.item())
                total_kl += float(kl.mean().item())
                total_entropy += float(entropy_bonus.item())
                num_optimizer_steps += 1

        reward_summary = summarize_reward_breakdowns(
            [trajectory.reward_breakdown for trajectory in trajectories]
        )
        metrics = {
            "iteration": float(iteration_index),
            "num_stage_trajectories": float(len(trajectories)),
            "mean_policy_loss": total_policy_loss / max(num_optimizer_steps, 1),
            "mean_value_loss": total_value_loss / max(num_optimizer_steps, 1),
            "mean_kl": total_kl / max(num_optimizer_steps, 1),
            "mean_entropy": total_entropy / max(num_optimizer_steps, 1),
            "mean_reward": float(rewards.mean().item()),
            "mean_old_value": float(old_values.mean().item()),
            "mean_return": float(returns.mean().item()),
            "mean_molecules_per_rollout": (
                len(trajectories) / max(len(examples), 1)
            ),
            **reward_summary,
        }

        if (
            iteration_index % self.config.save_every_iterations == 0
            or iteration_index == self.config.ppo_iterations
        ):
            self.save_checkpoint(iteration_index=iteration_index, metrics=metrics, trajectories=trajectories)

        return metrics

    def save_checkpoint(
        self,
        *,
        iteration_index: int,
        metrics: dict[str, float],
        trajectories: Sequence[StageTrajectory],
    ) -> Path:
        checkpoint_dir = ensure_dir(
            Path(self.config.output_dir) / "checkpoints" / f"iteration-{iteration_index:04d}"
        )
        self.policy_model.save_checkpoint(
            checkpoint_dir,
            tokenizer=self.tokenizer,
            config=self.config.to_dict(),
            metrics=metrics,
        )
        write_jsonl(
            checkpoint_dir / "rollout_summary.jsonl",
            [trajectory.to_dict() for trajectory in trajectories],
        )
        write_json(checkpoint_dir / "iteration_metrics.json", metrics)
        return checkpoint_dir
