from __future__ import annotations

import argparse
import json
import random
from typing import Sequence

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from reward_utils.defaults import RewardConfig
from src.io_utils import PROJECT_ROOT, ensure_dir, load_yaml, resolve_path, set_seed, write_json
from src.training import choose_device

from post_training.logging import BaseTracker, NullTracker, build_tracker
from post_training.shared.config import resolve_ppo_checkpoint_source, resolve_ppo_config_paths
from post_training.sft_multi.dataset import MultiMoleculeDataset

from .checkpointing import prepare_ppo_output_dir, save_iteration_artifacts, write_ppo_history
from .config import PPOConfig, StageTrajectory, build_ppo_config
from .model import (
    PolicyValueModel,
    assert_checkpoint_tokenizer_matches_model,
    load_reference_model,
)
from .rewarding import build_reward_config, summarize_reward_breakdowns
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
                clipped_ratio = torch.clamp(
                    ratio,
                    1.0 - self.config.clip_range,
                    1.0 + self.config.clip_range,
                )
                clipped_objective = torch.minimum(
                    ratio * batch_advantages,
                    clipped_ratio * batch_advantages,
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
            "mean_molecules_per_rollout": len(trajectories) / max(len(examples), 1),
            **reward_summary,
        }

        if (
            iteration_index % self.config.save_every_iterations == 0
            or iteration_index == self.config.ppo_iterations
        ):
            self.save_checkpoint(
                iteration_index=iteration_index,
                metrics=metrics,
                trajectories=trajectories,
            )

        return metrics

    def save_checkpoint(
        self,
        *,
        iteration_index: int,
        metrics: dict[str, float],
        trajectories: Sequence[StageTrajectory],
    ):
        return save_iteration_artifacts(
            output_dir=self.config.output_dir,
            iteration_index=iteration_index,
            policy_model=self.policy_model,
            tokenizer=self.tokenizer,
            config=self.config.to_dict(),
            metrics=metrics,
            trajectories=trajectories,
            create_archive=iteration_index == self.config.ppo_iterations,
        )


def sample_examples(dataset: MultiMoleculeDataset, count: int) -> list[dict[str, object]]:
    return [dataset[random.randrange(len(dataset))] for _ in range(count)]


def _build_ppo_tracking_config_payload(
    config: dict[str, object],
    *,
    output_dir: str,
) -> dict[str, object]:
    return {
        "stage_name": "molecule_wise_ppo",
        "output_dir": output_dir,
        "seed": config.get("seed"),
        "resolved_config": config,
    }


def _build_ppo_tracking_summary(summary: dict[str, object]) -> dict[str, object]:
    history = summary.get("history", [])
    last_metrics = history[-1] if history else {}
    tracking_summary: dict[str, object] = {
        "output_dir": summary["output_dir"],
        "num_iterations": summary["num_iterations"],
    }
    for key in ("mean_reward", "mean_kl", "mean_entropy", "mean_policy_loss", "mean_value_loss"):
        if key in last_metrics:
            tracking_summary[f"last_{key}"] = last_metrics[key]
    return tracking_summary


def run_molecule_stage_ppo(config: dict[str, object]) -> dict[str, object]:
    resolved_config = dict(config)
    model_config = dict(resolved_config.get("model", {}))
    resolved_checkpoint_path_or_id = resolve_ppo_checkpoint_source(
        model_config["checkpoint"],
        project_root=PROJECT_ROOT,
    )
    model_config["checkpoint"] = resolved_checkpoint_path_or_id
    resolved_config["model"] = model_config
    config = resolved_config

    tokenizer = AutoTokenizer.from_pretrained(resolved_checkpoint_path_or_id, use_fast=True)
    tokenizer.model_max_length = int(1e9)

    ppo_config = build_ppo_config(config)
    reward_config = build_reward_config(
        config.get("reward", {}),
        dataset_hint=str(config.get("data", {}).get("train_file", "")),
    )
    device = choose_device(config["training"].get("device", "auto"))

    policy_model = PolicyValueModel.from_pretrained(
        resolved_checkpoint_path_or_id,
        use_lora=ppo_config.use_lora,
        lora_rank=ppo_config.lora_rank,
        lora_alpha=ppo_config.lora_alpha,
        lora_dropout=ppo_config.lora_dropout,
        target_modules=ppo_config.target_modules,
        freeze_base_model_without_lora=ppo_config.freeze_base_model_without_lora,
    )
    assert_checkpoint_tokenizer_matches_model(
        tokenizer,
        policy_model.policy_model,
        context="PPO policy checkpoint",
    )
    policy_model.to(device)
    reference_model = load_reference_model(resolved_checkpoint_path_or_id)
    assert_checkpoint_tokenizer_matches_model(
        tokenizer,
        reference_model,
        context="PPO reference checkpoint",
    )
    reference_model.to(device)

    trainer = MoleculeWisePPOTrainer(
        policy_model=policy_model,
        reference_model=reference_model,
        tokenizer=tokenizer,
        config=ppo_config,
        reward_config=reward_config,
        device=device,
    )

    train_dataset = MultiMoleculeDataset.from_jsonl(config["data"]["train_file"])
    output_dir = prepare_ppo_output_dir(
        ppo_config.output_dir,
        config=config,
        reward_config=reward_config.__dict__,
    )
    ensure_dir(output_dir)

    tracker: BaseTracker = NullTracker()
    history: list[dict[str, float]] = []
    try:
        tracker = build_tracker(
            config,
            stage_name="molecule_wise_ppo",
            output_dir=output_dir,
        )
        tracker.log_config(
            _build_ppo_tracking_config_payload(config, output_dir=str(output_dir))
        )

        for iteration in range(1, ppo_config.ppo_iterations + 1):
            iteration_examples = sample_examples(train_dataset, ppo_config.batch_size)
            metrics = trainer.train_iteration(iteration_examples, iteration_index=iteration)
            history.append(metrics)
            write_ppo_history(output_dir, history)
            tracker.log_metrics(metrics, step=iteration, prefix="ppo")

        summary = {
            "output_dir": str(output_dir),
            "num_iterations": ppo_config.ppo_iterations,
            "history": history,
            "resolved_checkpoint_source": resolved_checkpoint_path_or_id,
            "resolved_checkpoint_path_or_id": resolved_checkpoint_path_or_id,
        }
        write_json(output_dir / "run_summary.json", summary)
        tracker.log_summary(_build_ppo_tracking_summary(summary), prefix="ppo")
        tracker.finish(status="success")
        return summary
    except Exception:
        tracker.finish(status="failed")
        raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the molecule-stage PPO post-training stage.")
    parser.add_argument(
        "--config",
        default="configs/molecule_wise_ppo.yaml",
        help="Path to the YAML PPO config relative to the project root.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = resolve_path(args.config, PROJECT_ROOT)
    config = resolve_ppo_config_paths(load_yaml(config_path), project_root=PROJECT_ROOT)
    set_seed(int(config.get("seed", 42)))
    summary = run_molecule_stage_ppo(config)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
