from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
from time import perf_counter
from typing import Any, Sequence

import torch
from transformers import AutoTokenizer

from reward_utils.defaults import RewardConfig
from src.constants import EOM_TOKEN
from src.io_utils import PROJECT_ROOT, ensure_dir, load_yaml, resolve_path, set_seed, write_json
from src.training import choose_device

from post_training.logging import BaseTracker, NullTracker, build_tracker
from post_training.shared.config import resolve_gflownet_config_paths, resolve_ppo_checkpoint_source
from post_training.shared.decoding import (
    StageTokenConstraints,
    build_stage_token_constraints,
)
from post_training.shared.diagnostics import (
    build_categorized_metric_record,
    categorize_metric_payload,
    iter_categorized_tracker_payloads,
)
from post_training.sft_multi.dataset import MultiMoleculeDataset

from .buffer import OnPolicyBatch, TrajectoryReplayBuffer
from .checkpointing import (
    append_iteration_diagnostics,
    append_iteration_diagnostics_categorized,
    append_trajectory_previews,
    prepare_gflownet_output_dir,
    save_gflownet_checkpoint_artifacts,
    save_gflownet_iteration_artifacts,
    write_gflownet_history,
)
from .config import GFlowNetConfig, build_gflownet_config
from .diagnostics import (
    GFlowNetTrainIterationResult,
    all_finite,
    build_trajectory_preview_payload,
    rollout_stage_metrics,
    safe_rate,
    stack_scalar_likes,
    termination_reason_metrics,
    tracker_diagnostic_metrics,
    tracker_headline_metrics,
)
from .losses import (
    detailed_balance_loss,
    detailed_balance_residuals,
    subtrajectory_balance_loss,
    subtrajectory_balance_residuals,
    trajectory_balance_loss,
    trajectory_balance_residual,
)
from .model import GFlowNetModel, assert_checkpoint_tokenizer_matches_model
from .rewarding import build_reward_config
from .rollout import encode_decoder_prefix, encode_prompt, sample_stage_trajectories_for_example
from .trajectory import SampledStageTrajectory, ScoredStageTrajectory


def _tensor_mean(values: torch.Tensor) -> float:
    if values.numel() == 0:
        return 0.0
    return float(values.mean().item())


def _tensor_std(values: torch.Tensor) -> float:
    if values.numel() <= 1:
        return 0.0
    return float(values.std(unbiased=False).item())


class MultiMoleculeGFlowNetTrainer:
    def __init__(
        self,
        *,
        model: GFlowNetModel,
        tokenizer,
        config: GFlowNetConfig,
        reward_config: RewardConfig | None = None,
        device: torch.device | None = None,
        stage_token_constraints: StageTokenConstraints | None = None,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.reward_config = reward_config
        self.device = device or next(model.parameters()).device
        set_constraints = getattr(self.model, "set_stage_token_constraints", None)
        if callable(set_constraints):
            set_constraints(stage_token_constraints)
        elif stage_token_constraints is not None:
            setattr(self.model, "_stage_token_constraints", stage_token_constraints)
        self.model.to(self.device)

        trainable_parameters = [
            parameter for parameter in self.model.parameters() if parameter.requires_grad
        ]
        self.optimizer = torch.optim.AdamW(
            trainable_parameters,
            lr=self.config.learning_rate,
        )
        self.replay_buffer = (
            TrajectoryReplayBuffer(
                self.config.replay.capacity,
                max_total_action_tokens=self.config.replay.max_total_action_tokens,
            )
            if self.config.replay.capacity > 0
            else None
        )
        self.replay_rng = random.Random(0)
        self.best_objective_loss: float | None = None
        self.best_checkpoint_iteration: int | None = None
        self.best_checkpoint_dir: str | None = None
        self.best_checkpoint_zip: str | None = None

    def collect_on_policy_trajectories(
        self,
        examples: Sequence[dict[str, object]],
        *,
        iteration_index: int,
    ) -> list[SampledStageTrajectory]:
        trajectories: list[SampledStageTrajectory] = []
        for example_index, example in enumerate(examples):
            rollout_id = f"iter-{iteration_index:04d}-sample-{example_index:04d}-{example['id']}"
            trajectories.extend(
                sample_stage_trajectories_for_example(
                    self.model,
                    self.tokenizer,
                    example,
                    rollout_id=rollout_id,
                    generation_config=self.config.rollout,
                    reward_config=self.reward_config,
                    invalid_terminal_reward=self.config.invalid_terminal_reward,
                    device=self.device,
                )
            )
        return trajectories

    def score_trajectory(
        self,
        trajectory: SampledStageTrajectory,
    ) -> ScoredStageTrajectory:
        prompt_inputs = encode_prompt(
            self.tokenizer,
            trajectory.prompt_text,
            max_source_length=self.config.rollout.max_source_length,
            device=self.device,
        )
        decoder_start_token_id = int(self.model.policy_model.config.decoder_start_token_id)
        decoder_prefix_ids = encode_decoder_prefix(
            self.tokenizer,
            trajectory.decoder_prefix_text,
            decoder_start_token_id=decoder_start_token_id,
            device=self.device,
        )
        stop_token_id = self.tokenizer.convert_tokens_to_ids(EOM_TOKEN)
        log_pf_tokens, log_stop, log_state_flows = self.model.score_action_sequence(
            input_ids=prompt_inputs["input_ids"],
            attention_mask=prompt_inputs["attention_mask"],
            decoder_prefix_ids=decoder_prefix_ids,
            action_token_ids=trajectory.action_token_ids,
            stop_token_id=int(stop_token_id),
        )
        return ScoredStageTrajectory(
            sampled=trajectory,
            log_pf_tokens=log_pf_tokens,
            log_stop=log_stop,
            log_state_flows=log_state_flows,
        )

    def score_trajectories(
        self,
        trajectories: Sequence[SampledStageTrajectory],
    ) -> list[ScoredStageTrajectory]:
        return [self.score_trajectory(trajectory) for trajectory in trajectories]

    def _compute_objective_loss(
        self,
        scored_trajectories: Sequence[ScoredStageTrajectory],
    ) -> tuple[torch.Tensor, dict[str, float]]:
        if not scored_trajectories:
            zero = torch.zeros((), device=self.device)
            return zero, {
                "objective_loss": 0.0,
                "objective_residual_mean": 0.0,
                "objective_residual_std": 0.0,
            }

        if self.config.objective == "tb":
            residuals = torch.stack(
                [trajectory_balance_residual(trajectory) for trajectory in scored_trajectories]
            )
            loss = trajectory_balance_loss(scored_trajectories)
            diagnostics = {
                "objective_loss": float(loss.item()),
                "objective_residual_mean": _tensor_mean(residuals),
                "objective_residual_std": _tensor_std(residuals),
                "mean_root_log_flow": _tensor_mean(
                    torch.stack([trajectory.log_state_flows[0] for trajectory in scored_trajectories])
                ),
                "mean_terminal_stop_logprob": _tensor_mean(
                    torch.stack([trajectory.log_stop[-1] for trajectory in scored_trajectories])
                ),
            }
            return loss, diagnostics

        if self.config.objective == "db":
            residual_sets = [
                detailed_balance_residuals(trajectory) for trajectory in scored_trajectories
            ]
            residuals = torch.cat(residual_sets) if residual_sets else torch.zeros((), device=self.device)
            loss_terms = [detailed_balance_loss(trajectory) for trajectory in scored_trajectories]
            loss = torch.stack(loss_terms).mean() if loss_terms else torch.zeros((), device=self.device)
            diagnostics = {
                "objective_loss": float(loss.item()),
                "objective_residual_mean": _tensor_mean(residuals),
                "objective_residual_std": _tensor_std(residuals),
                "mean_root_log_flow": _tensor_mean(
                    torch.stack([trajectory.log_state_flows[0] for trajectory in scored_trajectories])
                ),
                "mean_terminal_stop_logprob": _tensor_mean(
                    torch.stack([trajectory.log_stop[-1] for trajectory in scored_trajectories])
                ),
            }
            return loss, diagnostics

        if self.config.objective == "subtb":
            residual_sets = [
                subtrajectory_balance_residuals(trajectory)[0] for trajectory in scored_trajectories
            ]
            residuals = torch.cat(residual_sets) if residual_sets else torch.zeros((), device=self.device)
            loss = subtrajectory_balance_loss(scored_trajectories)
            diagnostics = {
                "objective_loss": float(loss.item()),
                "objective_residual_mean": _tensor_mean(residuals),
                "objective_residual_std": _tensor_std(residuals),
                "mean_root_log_flow": _tensor_mean(
                    torch.stack([trajectory.log_state_flows[0] for trajectory in scored_trajectories])
                ),
                "mean_terminal_stop_logprob": _tensor_mean(
                    torch.stack([trajectory.log_stop[-1] for trajectory in scored_trajectories])
                ),
            }
            return loss, diagnostics

        raise ValueError(f"Unsupported objective: {self.config.objective!r}")

    def train_iteration(
        self,
        examples: Sequence[dict[str, object]],
        *,
        iteration_index: int,
    ) -> GFlowNetTrainIterationResult:
        iteration_start = perf_counter()
        sampling_start = perf_counter()
        on_policy_trajectories = self.collect_on_policy_trajectories(
            examples,
            iteration_index=iteration_index,
        )
        sampling_duration_sec = perf_counter() - sampling_start
        on_policy_batch = OnPolicyBatch.from_trajectories(on_policy_trajectories)
        if not on_policy_trajectories:
            iteration_duration_sec = perf_counter() - iteration_start
            metrics: dict[str, Any] = {
                "iteration": float(iteration_index),
                "learning_rate": float(self.optimizer.param_groups[0]["lr"]),
                "num_on_policy_trajectories": 0.0,
                "num_replay_trajectories": 0.0,
                "replay_fraction": 0.0,
                "replay_size": float(len(self.replay_buffer) if self.replay_buffer is not None else 0),
                "replay_total_action_tokens": float(
                    self.replay_buffer.total_action_tokens if self.replay_buffer is not None else 0
                ),
                "sampling_duration_sec": sampling_duration_sec,
                "replay_sampling_duration_sec": 0.0,
                "scoring_duration_sec": 0.0,
                "loss_duration_sec": 0.0,
                "backward_duration_sec": 0.0,
                "optimizer_duration_sec": 0.0,
                "iteration_duration_sec": iteration_duration_sec,
                "on_policy_trajectories_per_sec": 0.0,
                "optimization_batch_trajectories_per_sec": 0.0,
                "trajectories_per_sec": 0.0,
                "action_tokens_per_sec": 0.0,
                "all_finite": True,
                **termination_reason_metrics(on_policy_trajectories),
                **rollout_stage_metrics(
                    on_policy_trajectories,
                    max_molecules_per_sequence=self.config.rollout.max_molecules_per_sequence,
                    invalid_terminal_reward=self.config.invalid_terminal_reward,
                ),
            }
            diagnostic_metrics = None
            categorized_diagnostic_metrics = None
            if iteration_index % self.config.diagnostic_log_every_iterations == 0:
                diagnostic_metrics = tracker_diagnostic_metrics(metrics)
                categorized_diagnostic_metrics = categorize_metric_payload(
                    diagnostic_metrics,
                    metadata_keys=("iteration",),
                )
            return GFlowNetTrainIterationResult(
                metrics=metrics,
                diagnostic_metrics=diagnostic_metrics,
                categorized_diagnostic_metrics=categorized_diagnostic_metrics,
            )

        replay_trajectories: list[SampledStageTrajectory] = []
        replay_sampling_start = perf_counter()
        if self.replay_buffer is not None and self.config.replay.replay_batch_size > 0:
            replay_trajectories = self.replay_buffer.sample(
                self.config.replay.replay_batch_size,
                rng=self.replay_rng,
                with_replacement=self.config.replay.with_replacement,
            )
        replay_sampling_duration_sec = perf_counter() - replay_sampling_start

        if self.replay_buffer is not None:
            self.replay_buffer.extend(on_policy_trajectories)

        optimization_trajectories = [*on_policy_trajectories, *replay_trajectories]
        scoring_start = perf_counter()
        scored_trajectories = self.score_trajectories(optimization_trajectories)
        scoring_duration_sec = perf_counter() - scoring_start

        loss_start = perf_counter()
        loss, diagnostics = self._compute_objective_loss(scored_trajectories)
        loss_duration_sec = perf_counter() - loss_start

        self.optimizer.zero_grad(set_to_none=True)
        backward_start = perf_counter()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.model.parameters(),
            self.config.max_grad_norm,
        )
        backward_duration_sec = perf_counter() - backward_start
        optimizer_start = perf_counter()
        self.optimizer.step()
        optimizer_duration_sec = perf_counter() - optimizer_start
        iteration_duration_sec = perf_counter() - iteration_start

        on_policy_rewards = torch.tensor(
            [trajectory.terminal_reward for trajectory in on_policy_trajectories],
            dtype=torch.float32,
            device=self.device,
        )
        optimization_rewards = torch.tensor(
            [trajectory.terminal_reward for trajectory in optimization_trajectories],
            dtype=torch.float32,
            device=self.device,
        )

        action_counts = torch.tensor(
            [trajectory.num_actions for trajectory in on_policy_trajectories],
            dtype=torch.float32,
            device=self.device,
        )
        total_action_tokens = sum(trajectory.num_actions for trajectory in optimization_trajectories)
        optimization_phase_duration_sec = (
            scoring_duration_sec
            + loss_duration_sec
            + backward_duration_sec
            + optimizer_duration_sec
        )
        all_log_pf_tokens = stack_scalar_likes(
            [value for trajectory in scored_trajectories for value in trajectory.log_pf_tokens],
            device=self.device,
        )
        all_log_pb_tokens = stack_scalar_likes(
            [
                value
                for trajectory in scored_trajectories
                for value in trajectory.effective_log_pb_tokens()
            ],
            device=self.device,
        )
        all_log_state_flows = stack_scalar_likes(
            [value for trajectory in scored_trajectories for value in trajectory.log_state_flows],
            device=self.device,
        )
        terminal_stop_logprobs = stack_scalar_likes(
            [trajectory.log_stop[-1] for trajectory in scored_trajectories],
            device=self.device,
        )
        gradient_tensors = [
            parameter.grad.detach()
            for parameter in self.model.parameters()
            if parameter.grad is not None
        ]

        metrics: dict[str, Any] = {
            "iteration": float(iteration_index),
            "learning_rate": float(self.optimizer.param_groups[0]["lr"]),
            "objective_loss": diagnostics["objective_loss"],
            "mean_stage_reward": on_policy_batch.mean_stage_reward(),
            "stage_reward_std": _tensor_std(on_policy_rewards),
            "mean_training_stage_reward": _tensor_mean(optimization_rewards),
            "training_stage_reward_std": _tensor_std(optimization_rewards),
            "valid_fraction": on_policy_batch.valid_fraction(),
            "duplicate_fraction": on_policy_batch.duplicate_fraction(),
            "mean_num_actions": on_policy_batch.mean_num_actions(),
            "max_num_actions": float(action_counts.max().item()) if action_counts.numel() > 0 else 0.0,
            "mean_stage_index": on_policy_batch.mean_stage_index(),
            "num_on_policy_trajectories": float(len(on_policy_trajectories)),
            "num_replay_trajectories": float(len(replay_trajectories)),
            "replay_fraction": float(len(replay_trajectories) / len(optimization_trajectories)),
            "replay_size": float(len(self.replay_buffer) if self.replay_buffer is not None else 0),
            "replay_total_action_tokens": float(
                self.replay_buffer.total_action_tokens if self.replay_buffer is not None else 0
            ),
            "grad_norm": float(grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm),
            "mean_log_pf_token": _tensor_mean(all_log_pf_tokens),
            "mean_log_pb_token": _tensor_mean(all_log_pb_tokens),
            "mean_log_state_flow": _tensor_mean(all_log_state_flows),
            "sampling_duration_sec": sampling_duration_sec,
            "replay_sampling_duration_sec": replay_sampling_duration_sec,
            "scoring_duration_sec": scoring_duration_sec,
            "loss_duration_sec": loss_duration_sec,
            "backward_duration_sec": backward_duration_sec,
            "optimizer_duration_sec": optimizer_duration_sec,
            "iteration_duration_sec": iteration_duration_sec,
            "on_policy_trajectories_per_sec": safe_rate(
                len(on_policy_trajectories),
                sampling_duration_sec,
            ),
            "optimization_batch_trajectories_per_sec": safe_rate(
                len(optimization_trajectories),
                optimization_phase_duration_sec,
            ),
            "trajectories_per_sec": safe_rate(
                len(optimization_trajectories),
                iteration_duration_sec,
            ),
            "action_tokens_per_sec": safe_rate(total_action_tokens, iteration_duration_sec),
            "all_finite": all_finite(
                loss.detach().reshape(1),
                all_log_pf_tokens,
                all_log_pb_tokens,
                all_log_state_flows,
                terminal_stop_logprobs,
                *gradient_tensors,
            ),
            **diagnostics,
            **termination_reason_metrics(on_policy_trajectories),
            **rollout_stage_metrics(
                on_policy_trajectories,
                max_molecules_per_sequence=self.config.rollout.max_molecules_per_sequence,
                invalid_terminal_reward=self.config.invalid_terminal_reward,
            ),
        }

        diagnostic_metrics = None
        categorized_diagnostic_metrics = None
        if iteration_index % self.config.diagnostic_log_every_iterations == 0:
            diagnostic_metrics = tracker_diagnostic_metrics(metrics)
            categorized_diagnostic_metrics = categorize_metric_payload(
                diagnostic_metrics,
                metadata_keys=("iteration",),
            )

        trajectory_preview = None
        if iteration_index % self.config.trajectory_preview_every_iterations == 0:
            trajectory_preview = build_trajectory_preview_payload(
                on_policy_trajectories,
                iteration_index=iteration_index,
                num_samples=self.config.trajectory_preview_num_samples,
                max_chars=self.config.trajectory_preview_max_chars,
                tokenizer=self.tokenizer,
            )

        if (
            iteration_index % self.config.save_every_iterations == 0
            or iteration_index == self.config.gflownet_iterations
        ):
            self.save_checkpoint(
                iteration_index=iteration_index,
                metrics=metrics,
                trajectories=on_policy_trajectories,
            )
        self.save_best_checkpoint(
            iteration_index=iteration_index,
            metrics=metrics,
            trajectories=on_policy_trajectories,
        )

        return GFlowNetTrainIterationResult(
            metrics=metrics,
            diagnostic_metrics=diagnostic_metrics,
            categorized_diagnostic_metrics=categorized_diagnostic_metrics,
            trajectory_preview=trajectory_preview,
        )

    def save_checkpoint(
        self,
        *,
        iteration_index: int,
        metrics: dict[str, Any],
        trajectories: Sequence[SampledStageTrajectory],
    ):
        return save_gflownet_iteration_artifacts(
            output_dir=self.config.output_dir,
            iteration_index=iteration_index,
            model=self.model,
            tokenizer=self.tokenizer,
            config=self.config.to_dict(),
            metrics=metrics,
            trajectories=trajectories,
            create_archive=iteration_index == self.config.gflownet_iterations,
        )

    def save_best_checkpoint(
        self,
        *,
        iteration_index: int,
        metrics: dict[str, Any],
        trajectories: Sequence[SampledStageTrajectory],
    ) -> Path | None:
        objective_loss = metrics.get("objective_loss")
        if objective_loss is None:
            return None

        objective_loss_value = float(objective_loss)
        if (
            self.best_objective_loss is not None
            and objective_loss_value >= self.best_objective_loss
        ):
            return None

        checkpoint_dir, checkpoint_zip = save_gflownet_checkpoint_artifacts(
            checkpoint_dir=Path(self.config.output_dir) / "checkpoints" / "best",
            model=self.model,
            tokenizer=self.tokenizer,
            config=self.config.to_dict(),
            metrics=metrics,
            trajectories=trajectories,
            create_archive=True,
        )
        self.best_objective_loss = objective_loss_value
        self.best_checkpoint_iteration = iteration_index
        self.best_checkpoint_dir = str(checkpoint_dir)
        self.best_checkpoint_zip = str(checkpoint_zip) if checkpoint_zip is not None else None
        return checkpoint_dir


def sample_examples(dataset: MultiMoleculeDataset, count: int) -> list[dict[str, object]]:
    return [dataset[random.randrange(len(dataset))] for _ in range(count)]


def _build_gflownet_tracking_config_payload(
    config: dict[str, object],
    *,
    output_dir: str,
) -> dict[str, object]:
    return {
        "stage_name": "multi_molecule_gflownet",
        "output_dir": output_dir,
        "seed": config.get("seed"),
        "resolved_config": config,
    }


def _build_gflownet_tracking_summary(summary: dict[str, object]) -> dict[str, object]:
    history = summary.get("history", [])
    last_metrics = history[-1] if history else {}
    tracking_summary: dict[str, object] = {
        "output_dir": summary["output_dir"],
        "num_iterations": summary["num_iterations"],
    }
    if summary.get("best_objective_loss") is not None:
        tracking_summary["best_objective_loss"] = summary["best_objective_loss"]
    if summary.get("best_checkpoint_iteration") is not None:
        tracking_summary["best_checkpoint_iteration"] = summary["best_checkpoint_iteration"]
    for key in (
        "mean_stage_reward",
        "valid_fraction",
        "objective_loss",
    ):
        if key in last_metrics:
            tracking_summary[f"last_{key}"] = last_metrics[key]
    return tracking_summary


def run_multi_molecule_gflownet(config: dict[str, object]) -> dict[str, object]:
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
    tokenizer.model_max_length = int(1.0e9)

    gflownet_config = build_gflownet_config(config)
    reward_config = build_reward_config(
        config.get("reward", {}),
        dataset_hint=str(config.get("data", {}).get("train_file", "")),
    )
    device = choose_device(config["training"].get("device", "auto"))

    model = GFlowNetModel.from_pretrained(
        resolved_checkpoint_path_or_id,
        use_lora=gflownet_config.use_lora,
        lora_rank=gflownet_config.lora_rank,
        lora_alpha=gflownet_config.lora_alpha,
        lora_dropout=gflownet_config.lora_dropout,
        target_modules=gflownet_config.target_modules,
        freeze_base_model_without_lora=gflownet_config.freeze_base_model_without_lora,
    )
    assert_checkpoint_tokenizer_matches_model(
        tokenizer,
        model.policy_model,
        context="GFlowNet checkpoint",
    )
    model.to(device)

    train_dataset = MultiMoleculeDataset.from_jsonl(config["data"]["train_file"])
    set_constraints = getattr(model, "set_stage_token_constraints", None)
    if gflownet_config.rollout.constrained_decoding:
        stage_token_constraints = build_stage_token_constraints(
            tokenizer,
            train_dataset,
            selfies_dict_path=gflownet_config.rollout.selfies_dict_path,
            separator_token=gflownet_config.rollout.stage_separator,
        )
        if callable(set_constraints):
            set_constraints(stage_token_constraints)
        else:
            setattr(model, "_stage_token_constraints", stage_token_constraints)
    else:
        if callable(set_constraints):
            set_constraints(None)
        else:
            setattr(model, "_stage_token_constraints", None)

    trainer = MultiMoleculeGFlowNetTrainer(
        model=model,
        tokenizer=tokenizer,
        config=gflownet_config,
        reward_config=reward_config,
        device=device,
    )

    output_dir = prepare_gflownet_output_dir(
        gflownet_config.output_dir,
        config=config,
        reward_config=reward_config.__dict__,
    )
    ensure_dir(output_dir)

    tracker: BaseTracker = NullTracker()
    history: list[dict[str, Any]] = []
    try:
        tracker = build_tracker(
            config,
            stage_name="multi_molecule_gflownet",
            output_dir=output_dir,
        )
        tracker.log_config(
            _build_gflownet_tracking_config_payload(config, output_dir=str(output_dir))
        )

        for iteration in range(1, gflownet_config.gflownet_iterations + 1):
            iteration_examples = sample_examples(train_dataset, gflownet_config.batch_size)
            iteration_result = trainer.train_iteration(
                iteration_examples,
                iteration_index=iteration,
            )
            history.append(iteration_result.metrics)
            write_gflownet_history(output_dir, history)
            tracker.log_metrics(
                tracker_headline_metrics(iteration_result.metrics),
                step=iteration,
                prefix="gflownet",
            )
            if iteration_result.diagnostic_metrics is not None:
                append_iteration_diagnostics(
                    output_dir,
                    [iteration_result.diagnostic_metrics],
                )
                append_iteration_diagnostics_categorized(
                    output_dir,
                    [
                        build_categorized_metric_record(
                            iteration_result.diagnostic_metrics,
                            metadata_keys=("iteration",),
                            categories=iteration_result.categorized_diagnostic_metrics,
                        )
                    ],
                )
                tracker.log_metrics(
                    iteration_result.diagnostic_metrics,
                    step=iteration,
                    prefix="gflownet_diagnostics",
                )
                for prefix, payload in iter_categorized_tracker_payloads(
                    iteration_result.diagnostic_metrics,
                    base_prefix="gflownet_diagnostics",
                    metadata_keys=("iteration",),
                ):
                    tracker.log_metrics(
                        payload,
                        step=iteration,
                        prefix=prefix,
                    )
            if iteration_result.trajectory_preview is not None:
                append_trajectory_previews(
                    output_dir,
                    iteration_result.trajectory_preview["records"],
                )
                print(
                    "\n".join(
                        [
                            f"[gflownet][iteration {iteration}] trajectory preview",
                            iteration_result.trajectory_preview["tracker_text"],
                        ]
                    ),
                    flush=True,
                )
                tracker.log_summary(
                    {
                        "latest_trajectory_preview": iteration_result.trajectory_preview[
                            "tracker_text"
                        ],
                        "latest_trajectory_preview_iteration": float(iteration),
                    },
                    prefix="gflownet",
                )

        summary = {
            "output_dir": str(output_dir),
            "num_iterations": gflownet_config.gflownet_iterations,
            "history": history,
            "resolved_checkpoint_source": resolved_checkpoint_path_or_id,
            "resolved_checkpoint_path_or_id": resolved_checkpoint_path_or_id,
            "best_objective_loss": getattr(trainer, "best_objective_loss", None),
            "best_checkpoint_iteration": getattr(trainer, "best_checkpoint_iteration", None),
            "best_checkpoint_dir": getattr(trainer, "best_checkpoint_dir", None),
            "best_checkpoint_zip": getattr(trainer, "best_checkpoint_zip", None),
        }
        write_json(output_dir / "run_summary.json", summary)
        tracker.log_summary(_build_gflownet_tracking_summary(summary), prefix="gflownet")
        tracker.finish(status="success")
        return summary
    except Exception:
        tracker.finish(status="failed")
        raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the multi-molecule GFlowNet post-training stage."
    )
    parser.add_argument(
        "--config",
        default="configs/multi_molecule_gflownet.yaml",
        help="Path to the YAML GFlowNet config relative to the project root.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = resolve_path(args.config, PROJECT_ROOT)
    config = resolve_gflownet_config_paths(load_yaml(config_path), project_root=PROJECT_ROOT)
    set_seed(int(config.get("seed", 42)))
    summary = run_multi_molecule_gflownet(config)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
