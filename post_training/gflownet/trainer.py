from __future__ import annotations

import argparse
import copy
from dataclasses import dataclass
import json
from pathlib import Path
import random
from time import perf_counter
from typing import Any, Callable, Sequence

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

from .buffer import OnPolicyBatch, ReplaySampleBatch, build_replay_buffer
from .checkpointing import (
    append_gflownet_report_metrics,
    append_iteration_diagnostics,
    append_iteration_diagnostics_categorized,
    append_trajectory_previews,
    prepare_gflownet_output_dir,
    save_gflownet_checkpoint_artifacts,
    save_gflownet_iteration_artifacts,
    save_gflownet_last_artifacts,
    write_gflownet_history,
)
from .config import GFlowNetConfig, build_gflownet_config
from .diagnostics import (
    GFlowNetReportAccumulator,
    GFlowNetTrainIterationResult,
    all_finite,
    build_trajectory_preview_payload,
    on_policy_novelty_metrics,
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
from .rollout import (
    build_target_teacher_stage_trajectory_for_example,
    encode_decoder_prefix,
    encode_prompt,
    sample_stage_trajectories_for_example,
    sample_target_prefix_stage_trajectory_for_example,
)
from .trajectory import SampledStageTrajectory, ScoredStageTrajectory


def _tensor_mean(values: torch.Tensor) -> float:
    if values.numel() == 0:
        return 0.0
    return float(values.mean().item())


def _tensor_std(values: torch.Tensor) -> float:
    if values.numel() <= 1:
        return 0.0
    return float(values.std(unbiased=False).item())


def _compute_replay_target_count(
    *,
    on_policy_count: int,
    replay_fraction: float | None,
    legacy_replay_batch_size: int,
) -> int:
    if on_policy_count <= 0:
        return 0
    if replay_fraction is not None:
        if replay_fraction <= 0.0:
            return 0
        return max(
            0,
            int(round(on_policy_count * replay_fraction / max(1.0 - replay_fraction, 1.0e-6))),
        )
    return max(0, int(legacy_replay_batch_size))


def _compute_target_guided_target_count(
    *,
    anchor_count: int,
    on_policy_fraction: float,
    source_fraction: float,
    off_policy_fraction: float,
) -> int:
    if anchor_count <= 0 or source_fraction <= 0.0:
        return 0
    if on_policy_fraction > 0.0:
        return max(0, int(round(anchor_count * source_fraction / on_policy_fraction)))
    if off_policy_fraction <= 0.0:
        return 0
    return max(0, int(round(anchor_count * source_fraction / off_policy_fraction)))


@dataclass
class EpochBatchSampler:
    dataset: MultiMoleculeDataset
    batch_size: int
    rng: random.Random

    def __post_init__(self) -> None:
        self.batch_size = max(1, int(self.batch_size))
        self.indices = list(range(len(self.dataset)))
        self.cursor = 0
        self.epoch = 0
        self._start_next_epoch()

    def _start_next_epoch(self) -> None:
        self.rng.shuffle(self.indices)
        self.cursor = 0
        self.epoch += 1

    def next_batch(self) -> tuple[list[dict[str, object]], int]:
        if not self.indices:
            return [], self.epoch

        if self.cursor >= len(self.indices):
            self._start_next_epoch()

        end = min(self.cursor + self.batch_size, len(self.indices))
        batch_indices = self.indices[self.cursor:end]
        self.cursor = end

        return [self.dataset[index] for index in batch_indices], self.epoch


def _select_optimization_trajectories(
    *,
    target_teacher_trajectories: Sequence[SampledStageTrajectory],
    on_policy_trajectories: Sequence[SampledStageTrajectory],
    target_prefix_trajectories: Sequence[SampledStageTrajectory],
    replay_trajectories: Sequence[SampledStageTrajectory],
    max_optimization_trajectories_per_iter: int | None,
) -> tuple[list[SampledStageTrajectory], list[SampledStageTrajectory]]:
    optimization_candidates = [
        *target_teacher_trajectories,
        *on_policy_trajectories,
        *target_prefix_trajectories,
        *replay_trajectories,
    ]
    if max_optimization_trajectories_per_iter is None:
        return optimization_candidates, optimization_candidates
    max_trajectories = max(0, int(max_optimization_trajectories_per_iter))
    return optimization_candidates, optimization_candidates[:max_trajectories]


@dataclass(frozen=True)
class _ScoringMicrobatch:
    start_index: int
    trajectories: tuple[SampledStageTrajectory, ...]


def _set_model_stage_token_constraints(
    model: torch.nn.Module,
    stage_token_constraints: StageTokenConstraints | None,
) -> None:
    set_constraints = getattr(model, "set_stage_token_constraints", None)
    if callable(set_constraints):
        set_constraints(stage_token_constraints)
    elif stage_token_constraints is not None:
        setattr(model, "_stage_token_constraints", stage_token_constraints)


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
        self.parallel_devices = self._resolve_parallel_devices(self.device)
        self.parallel_training_enabled = bool(self.config.parallel_training.enabled)
        if self.parallel_training_enabled:
            self.device = self.parallel_devices[0]
        self.stage_token_constraints = stage_token_constraints
        _set_model_stage_token_constraints(self.model, stage_token_constraints)
        self.parallel_models: list[torch.nn.Module] = [self.model]
        self._last_parallel_scoring_counts: dict[str, int] = {
            str(self.device): 0,
        }
        self._last_parallel_scored_groups: list[list[ScoredStageTrajectory]] = []
        if self.parallel_training_enabled:
            self._initialize_parallel_replicas()
        self.model.to(self.device)

        trainable_parameters = [
            parameter for parameter in self.model.parameters() if parameter.requires_grad
        ]
        self.optimizer = torch.optim.AdamW(
            trainable_parameters,
            lr=self.config.learning_rate,
        )
        if self.parallel_training_enabled:
            self._sync_parallel_replicas_from_primary()
        replay_enabled = bool(self.config.replay.enabled) and not bool(
            self.config.target_guidance.enabled
        )
        self.replay_buffer = (
            build_replay_buffer(
                capacity=self.config.replay.capacity,
                max_total_action_tokens=self.config.replay.max_total_action_tokens,
                buffer_type=self.config.replay.buffer_type,
                recent_fraction=self.config.replay.recent_fraction,
                reward_fraction=self.config.replay.reward_fraction,
                uniform_fraction=self.config.replay.uniform_fraction,
                tb_residual_fraction=self.config.replay.tb_residual_fraction,
                reward_temperature=self.config.replay.reward_temperature,
                tb_residual_temperature=self.config.replay.tb_residual_temperature,
                recent_window_size=self.config.replay.recent_window_size,
                max_invalid_fraction=self.config.replay.max_invalid_fraction,
                max_duplicate_fraction=self.config.replay.max_duplicate_fraction,
            )
            if replay_enabled and self.config.replay.capacity > 0
            else None
        )
        self.replay_rng = random.Random(0)
        self.rollout_rng = random.Random(0)
        self.target_guidance_rng = random.Random(0)
        self.best_objective_loss: float | None = None
        self.best_checkpoint_iteration: int | None = None
        self.best_checkpoint_dir: str | None = None
        self.best_checkpoint_zip: str | None = None
        self.last_checkpoint_iteration: int | None = None
        self.last_checkpoint_dir: str | None = None
        self.last_checkpoint_zip: str | None = None

    def _resolve_parallel_devices(
        self,
        primary_device: torch.device,
    ) -> tuple[torch.device, ...]:
        parallel_config = self.config.parallel_training
        if not parallel_config.enabled:
            return (primary_device,)

        devices = tuple(torch.device(device_name) for device_name in parallel_config.devices)
        if len(devices) < 2:
            raise ValueError("enabled parallel_training requires at least two devices.")
        if parallel_config.mode != "replicated_scoring":
            raise ValueError("parallel_training.mode must be: replicated_scoring.")

        if parallel_config.strict:
            if any(device.type != "cuda" for device in devices):
                raise RuntimeError(
                    "parallel_training.strict requires CUDA devices; "
                    f"got {[str(device) for device in devices]}."
                )
            if not torch.cuda.is_available():
                raise RuntimeError("parallel_training requested CUDA devices, but CUDA is unavailable.")
            device_indices = tuple(0 if device.index is None else int(device.index) for device in devices)
            if len(set(device_indices)) != len(device_indices):
                raise RuntimeError(
                    "parallel_training.strict requires distinct CUDA device indices."
                )
            available_device_count = int(torch.cuda.device_count())
            if max(device_indices) >= available_device_count:
                raise RuntimeError(
                    "parallel_training requested devices "
                    f"{[str(device) for device in devices]}, but only "
                    f"{available_device_count} CUDA device(s) are available."
                )
        return devices

    def _initialize_parallel_replicas(self) -> None:
        self.parallel_models = [self.model]
        for device in self.parallel_devices[1:]:
            replica = copy.deepcopy(self.model)
            _set_model_stage_token_constraints(replica, self.stage_token_constraints)
            replica.to(device)
            self.parallel_models.append(replica)

    def _sync_parallel_replicas_from_primary(self) -> None:
        if not self.parallel_training_enabled:
            return
        primary_trainable = {
            name: parameter.detach()
            for name, parameter in self.model.named_parameters()
            if parameter.requires_grad
        }
        with torch.no_grad():
            for replica, device in zip(self.parallel_models[1:], self.parallel_devices[1:]):
                replica_parameters = dict(replica.named_parameters())
                for name, source_parameter in primary_trainable.items():
                    replica_parameters[name].copy_(source_parameter.to(device))

    def _zero_parallel_replica_gradients(self) -> None:
        if not self.parallel_training_enabled:
            return
        for replica in self.parallel_models[1:]:
            for parameter in replica.parameters():
                parameter.grad = None

    def _aggregate_parallel_replica_gradients(self) -> None:
        if not self.parallel_training_enabled:
            return
        primary_parameters = dict(self.model.named_parameters())
        for replica in self.parallel_models[1:]:
            for name, replica_parameter in replica.named_parameters():
                if not replica_parameter.requires_grad or replica_parameter.grad is None:
                    continue
                primary_parameter = primary_parameters[name]
                replica_grad = replica_parameter.grad.detach().to(primary_parameter.device)
                if primary_parameter.grad is None:
                    primary_parameter.grad = replica_grad.clone()
                else:
                    primary_parameter.grad.add_(replica_grad)

    def _parallel_training_metrics(self) -> dict[str, Any]:
        metrics: dict[str, Any] = {
            "parallel_training_enabled": float(self.parallel_training_enabled),
            "parallel_training_num_devices": float(
                len(self.parallel_devices) if self.parallel_training_enabled else 1
            ),
            "parallel_training_devices": ",".join(str(device) for device in self.parallel_devices),
            "parallel_training_mode": (
                self.config.parallel_training.mode
                if self.parallel_training_enabled
                else "disabled"
            ),
        }
        for device_name, count in sorted(self._last_parallel_scoring_counts.items()):
            normalized_device_name = (
                device_name.replace(":", "_")
                .replace("/", "_")
                .replace("-", "_")
                .replace(" ", "_")
            )
            metrics[f"parallel_training_scored_trajectories_{normalized_device_name}"] = float(count)
        return metrics

    def _scheduled_learning_rate(self, iteration_index: int) -> float:
        warmup_iterations = round(
            max(0, int(self.config.gflownet_iterations))
            * float(self.config.warmup_ratio)
        )
        if warmup_iterations <= 0:
            return float(self.config.learning_rate)
        if iteration_index <= warmup_iterations:
            warmup_progress = max(
                0.0,
                float(iteration_index) / float(warmup_iterations),
            )
            return float(self.config.learning_rate) * warmup_progress
        return float(self.config.learning_rate)

    def _final_iteration_index(self) -> int:
        return int(self.config.start_iteration) + int(self.config.gflownet_iterations)

    def _set_optimizer_learning_rate(self, learning_rate: float) -> None:
        for parameter_group in self.optimizer.param_groups:
            parameter_group["lr"] = float(learning_rate)

    def _run_sampling_eval(self, callback: Callable[[], Any]) -> Any:
        was_training = self.model.training
        self.model.eval()
        try:
            return callback()
        finally:
            self.model.train(was_training)

    def collect_on_policy_trajectories(
        self,
        examples: Sequence[dict[str, object]],
        *,
        iteration_index: int,
    ) -> list[SampledStageTrajectory]:
        trajectories: list[SampledStageTrajectory] = []
        return_last_valid_trajectory_only = self.config.objective == "subtb"
        max_retained_trajectories = max(1, int(self.config.rollout.max_molecules_per_sequence))
        for example_index, example in enumerate(examples):
            rollout_id = f"iter-{iteration_index:04d}-sample-{example_index:04d}-{example['id']}"
            sampled_trajectories = sample_stage_trajectories_for_example(
                self.model,
                self.tokenizer,
                example,
                rollout_id=rollout_id,
                generation_config=self.config.rollout,
                reward_config=self.reward_config,
                invalid_terminal_reward=self.config.invalid_terminal_reward,
                device=self.device,
                rng=self.rollout_rng,
                return_last_valid_trajectory_only=return_last_valid_trajectory_only,
            )
            remaining_slots = max_retained_trajectories - len(trajectories)
            if remaining_slots <= 0:
                break
            trajectories.extend(sampled_trajectories[:remaining_slots])
            if len(trajectories) >= max_retained_trajectories:
                break
        return trajectories

    def _target_guidance_example_for_sample(
        self,
        examples: Sequence[dict[str, object]],
    ) -> dict[str, object]:
        if not examples:
            raise ValueError("Target-guided trajectory collection requires examples.")
        return examples[self.target_guidance_rng.randrange(len(examples))]

    def collect_target_prefix_trajectories(
        self,
        examples: Sequence[dict[str, object]],
        *,
        iteration_index: int,
        count: int,
    ) -> list[SampledStageTrajectory]:
        if count <= 0 or not examples:
            return []
        trajectories: list[SampledStageTrajectory] = []
        for sample_index in range(count):
            example = self._target_guidance_example_for_sample(examples)
            rollout_id = (
                f"iter-{iteration_index:04d}-target-prefix-{sample_index:04d}-"
                f"{example['id']}"
            )
            trajectories.append(
                sample_target_prefix_stage_trajectory_for_example(
                    self.model,
                    self.tokenizer,
                    example,
                    rollout_id=rollout_id,
                    generation_config=self.config.rollout,
                    reward_config=self.reward_config,
                    invalid_terminal_reward=self.config.invalid_terminal_reward,
                    device=self.device,
                    rng=self.target_guidance_rng,
                    stage_strategy=self.config.target_guidance.prefix_stage_strategy,
                    shuffle_target_selfies_list=(
                        self.config.target_guidance.shuffle_target_selfies_list
                    ),
                )
            )
        return trajectories

    def collect_target_teacher_trajectories(
        self,
        examples: Sequence[dict[str, object]],
        *,
        iteration_index: int,
        count: int,
    ) -> list[SampledStageTrajectory]:
        if count <= 0 or not examples:
            return []
        trajectories: list[SampledStageTrajectory] = []
        for sample_index in range(count):
            example = self._target_guidance_example_for_sample(examples)
            rollout_id = (
                f"iter-{iteration_index:04d}-target-teacher-{sample_index:04d}-"
                f"{example['id']}"
            )
            trajectories.append(
                build_target_teacher_stage_trajectory_for_example(
                    self.tokenizer,
                    example,
                    rollout_id=rollout_id,
                    generation_config=self.config.rollout,
                    reward_config=self.reward_config,
                    invalid_terminal_reward=self.config.invalid_terminal_reward,
                    rng=self.target_guidance_rng,
                    stage_strategy=self.config.target_guidance.teacher_stage_strategy,
                    shuffle_target_selfies_list=(
                        self.config.target_guidance.shuffle_target_selfies_list
                    ),
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
        if not trajectories:
            self._last_parallel_scoring_counts = {
                str(device): 0 for device in self.parallel_devices
            }
            self._last_parallel_scored_groups = []
            return []

        microbatches = self._build_scoring_microbatches(trajectories)
        if not self.parallel_training_enabled:
            scored_items, scored_group = self._score_microbatches(
                model=self.model,
                device=self.device,
                microbatches=microbatches,
            )
            self._last_parallel_scoring_counts = {str(self.device): len(scored_group)}
            self._last_parallel_scored_groups = [scored_group]
            return [scored for _index, scored in sorted(scored_items, key=lambda item: item[0])]

        self._sync_parallel_replicas_from_primary()
        shards = self._split_scoring_microbatches(microbatches)
        scored_items: list[tuple[int, ScoredStageTrajectory]] = []
        scored_groups: list[list[ScoredStageTrajectory]] = []
        scoring_counts: dict[str, int] = {}
        for model, device, shard in zip(self.parallel_models, self.parallel_devices, shards):
            shard_items, shard_group = self._score_microbatches(
                model=model,
                device=device,
                microbatches=shard,
            )
            scored_items.extend(shard_items)
            scored_groups.append(shard_group)
            scoring_counts[str(device)] = len(shard_group)
        self._last_parallel_scoring_counts = scoring_counts
        self._last_parallel_scored_groups = scored_groups
        return [scored for _index, scored in sorted(scored_items, key=lambda item: item[0])]

    def _build_scoring_microbatches(
        self,
        trajectories: Sequence[SampledStageTrajectory],
    ) -> list[_ScoringMicrobatch]:
        microbatch_size = max(1, int(self.config.scoring_microbatch_size))
        return [
            _ScoringMicrobatch(
                start_index=start,
                trajectories=tuple(trajectories[start : start + microbatch_size]),
            )
            for start in range(0, len(trajectories), microbatch_size)
        ]

    def _split_scoring_microbatches(
        self,
        microbatches: Sequence[_ScoringMicrobatch],
    ) -> list[list[_ScoringMicrobatch]]:
        shards: list[list[_ScoringMicrobatch]] = [
            [] for _device in self.parallel_devices
        ]
        for microbatch_index, microbatch in enumerate(microbatches):
            shards[microbatch_index % len(shards)].append(microbatch)
        return shards

    def _score_microbatches(
        self,
        *,
        model: torch.nn.Module,
        device: torch.device,
        microbatches: Sequence[_ScoringMicrobatch],
    ) -> tuple[list[tuple[int, ScoredStageTrajectory]], list[ScoredStageTrajectory]]:
        scored: list[ScoredStageTrajectory] = []
        scored_items: list[tuple[int, ScoredStageTrajectory]] = []
        decoder_start_token_id = int(model.policy_model.config.decoder_start_token_id)
        stop_token_id = int(self.tokenizer.convert_tokens_to_ids(EOM_TOKEN))

        for microbatch in microbatches:
            trajectory_batch = list(microbatch.trajectories)
            prompt_inputs = self.tokenizer(
                [trajectory.prompt_text for trajectory in trajectory_batch],
                padding=True,
                truncation=True,
                max_length=self.config.rollout.max_source_length,
                return_tensors="pt",
            )
            prompt_input_ids = prompt_inputs["input_ids"].to(device)
            prompt_attention_mask = prompt_inputs.get("attention_mask")
            if prompt_attention_mask is None:
                prompt_attention_mask = torch.ones_like(prompt_input_ids)
            else:
                prompt_attention_mask = prompt_attention_mask.to(device)

            decoder_prefix_ids = [
                encode_decoder_prefix(
                    self.tokenizer,
                    trajectory.decoder_prefix_text,
                    decoder_start_token_id=decoder_start_token_id,
                    device=device,
                )
                for trajectory in trajectory_batch
            ]
            batch_scores = model.score_action_sequences(
                input_ids=prompt_input_ids,
                attention_mask=prompt_attention_mask,
                decoder_prefix_ids=decoder_prefix_ids,
                action_token_ids=[
                    trajectory.action_token_ids for trajectory in trajectory_batch
                ],
                stop_token_id=stop_token_id,
            )
            for row_offset, (trajectory, (log_pf_tokens, log_stop, log_state_flows)) in enumerate(
                zip(
                    trajectory_batch,
                    batch_scores,
                )
            ):
                scored_trajectory = ScoredStageTrajectory(
                    sampled=trajectory,
                    log_pf_tokens=log_pf_tokens,
                    log_stop=log_stop,
                    log_state_flows=log_state_flows,
                )
                scored.append(scored_trajectory)
                scored_items.append((microbatch.start_index + row_offset, scored_trajectory))
        return scored_items, scored

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

        numerator, denominator, residuals = self._compute_objective_components(scored_trajectories)
        loss = numerator / denominator.clamp_min(1.0e-12)
        return loss, self._build_objective_diagnostics(
            scored_trajectories,
            loss=loss,
            residuals=residuals,
        )

    def _compute_parallel_objective_loss(
        self,
        scored_groups: Sequence[Sequence[ScoredStageTrajectory]],
        fallback_scored_trajectories: Sequence[ScoredStageTrajectory],
    ) -> tuple[torch.Tensor, dict[str, float]]:
        non_empty_groups = [list(group) for group in scored_groups if group]
        if not self.parallel_training_enabled or len(non_empty_groups) <= 1:
            return self._compute_objective_loss(fallback_scored_trajectories)

        numerator_terms: list[torch.Tensor] = []
        denominator_terms: list[torch.Tensor] = []
        residual_sets: list[torch.Tensor] = []
        ordered_scored: list[ScoredStageTrajectory] = []
        for group in non_empty_groups:
            numerator, denominator, residuals = self._compute_objective_components(group)
            numerator_terms.append(numerator.to(self.device))
            denominator_terms.append(denominator.to(self.device))
            residual_sets.append(residuals.detach().to(self.device))
            ordered_scored.extend(group)

        numerator_total = torch.stack(numerator_terms).sum()
        denominator_total = torch.stack(denominator_terms).sum().clamp_min(1.0e-12)
        loss = numerator_total / denominator_total
        residuals = (
            torch.cat(residual_sets)
            if residual_sets
            else torch.zeros(0, dtype=torch.float32, device=self.device)
        )
        return loss, self._build_objective_diagnostics(
            ordered_scored,
            loss=loss,
            residuals=residuals,
        )

    def _compute_objective_components(
        self,
        scored_trajectories: Sequence[ScoredStageTrajectory],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.config.objective == "tb":
            residuals = torch.stack(
                [trajectory_balance_residual(trajectory) for trajectory in scored_trajectories]
            )
            numerator = residuals.pow(2).sum()
            denominator = torch.tensor(
                float(max(residuals.numel(), 1)),
                dtype=residuals.dtype,
                device=residuals.device,
            )
            return numerator, denominator, residuals

        if self.config.objective == "db":
            residual_sets = [
                detailed_balance_residuals(trajectory) for trajectory in scored_trajectories
            ]
            residuals = torch.cat(residual_sets) if residual_sets else torch.zeros((), device=self.device)
            loss_terms = [detailed_balance_loss(trajectory) for trajectory in scored_trajectories]
            if loss_terms:
                numerator = torch.stack(loss_terms).sum()
                denominator = torch.tensor(
                    float(len(loss_terms)),
                    dtype=numerator.dtype,
                    device=numerator.device,
                )
            else:
                numerator = torch.zeros((), device=self.device)
                denominator = torch.ones((), device=self.device)
            return numerator, denominator, residuals

        if self.config.objective == "subtb":
            residual_sets: list[torch.Tensor] = []
            weighted_numerators: list[torch.Tensor] = []
            weighted_denominators: list[torch.Tensor] = []
            fallback_device = self.device
            fallback_dtype = torch.float32
            for trajectory in scored_trajectories:
                residuals, weights = subtrajectory_balance_residuals(trajectory)
                if residuals.numel() == 0:
                    continue
                fallback_device = residuals.device
                fallback_dtype = residuals.dtype
                residual_sets.append(residuals)
                weighted_numerators.append((weights * residuals.pow(2)).sum())
                weighted_denominators.append(weights.sum())
            if weighted_numerators:
                numerator = torch.stack(weighted_numerators).sum()
                denominator = torch.stack(weighted_denominators).sum()
                residuals = torch.cat(residual_sets)
            else:
                numerator = torch.zeros((), device=fallback_device, dtype=fallback_dtype)
                denominator = torch.ones((), device=fallback_device, dtype=fallback_dtype)
                residuals = torch.zeros(0, device=fallback_device, dtype=fallback_dtype)
            return numerator, denominator, residuals

        raise ValueError(f"Unsupported objective: {self.config.objective!r}")

    def _build_objective_diagnostics(
        self,
        scored_trajectories: Sequence[ScoredStageTrajectory],
        *,
        loss: torch.Tensor,
        residuals: torch.Tensor,
    ) -> dict[str, float]:
        residual_values = residuals.detach().to(self.device)
        root_log_flows = stack_scalar_likes(
            [trajectory.log_state_flows[0] for trajectory in scored_trajectories],
            device=self.device,
        )
        terminal_stop_logprobs = stack_scalar_likes(
            [trajectory.log_stop[-1] for trajectory in scored_trajectories],
            device=self.device,
        )
        return {
            "objective_loss": float(loss.detach().to(self.device).item()),
            "objective_residual_mean": _tensor_mean(residual_values),
            "objective_residual_std": _tensor_std(residual_values),
            "mean_root_log_flow": _tensor_mean(root_log_flows),
            "mean_terminal_stop_logprob": _tensor_mean(terminal_stop_logprobs),
        }

    def train_iteration(
        self,
        examples: Sequence[dict[str, object]],
        *,
        iteration_index: int,
    ) -> GFlowNetTrainIterationResult:
        iteration_start = perf_counter()
        self._set_optimizer_learning_rate(
            self._scheduled_learning_rate(iteration_index)
        )
        sampling_start = perf_counter()
        raw_on_policy_trajectories = self._run_sampling_eval(
            lambda: self.collect_on_policy_trajectories(
                examples,
                iteration_index=iteration_index,
            )
        )
        max_on_policy_trajectories = max(1, int(self.config.rollout.max_molecules_per_sequence))
        on_policy_trajectories = raw_on_policy_trajectories[:max_on_policy_trajectories]
        on_policy_trimmed_count = len(raw_on_policy_trajectories) - len(on_policy_trajectories)
        sampling_duration_sec = perf_counter() - sampling_start
        on_policy_batch = OnPolicyBatch.from_trajectories(on_policy_trajectories)
        duplicate_count_on_policy = sum(
            int(trajectory.is_duplicate) for trajectory in on_policy_trajectories
        )
        novelty_metrics = on_policy_novelty_metrics(on_policy_trajectories)

        replay_sample = ReplaySampleBatch(())
        replay_trajectories: list[SampledStageTrajectory] = []
        target_prefix_trajectories: list[SampledStageTrajectory] = []
        target_teacher_trajectories: list[SampledStageTrajectory] = []
        replay_sampling_duration_sec = 0.0
        target_guidance_sampling_duration_sec = 0.0

        if self.config.target_guidance.enabled:
            target_guidance_sampling_start = perf_counter()
            anchor_count = max(len(on_policy_trajectories), len(examples))
            on_policy_fraction = float(self.config.target_guidance.on_policy_fraction)
            prefix_fraction = float(self.config.target_guidance.target_prefix_rollout_fraction)
            teacher_fraction = float(self.config.target_guidance.target_teacher_fraction)
            off_policy_fraction = prefix_fraction + teacher_fraction
            target_prefix_count = _compute_target_guided_target_count(
                anchor_count=anchor_count,
                on_policy_fraction=on_policy_fraction,
                source_fraction=prefix_fraction,
                off_policy_fraction=off_policy_fraction,
            )
            target_teacher_count = _compute_target_guided_target_count(
                anchor_count=anchor_count,
                on_policy_fraction=on_policy_fraction,
                source_fraction=teacher_fraction,
                off_policy_fraction=off_policy_fraction,
            )
            target_prefix_trajectories, target_teacher_trajectories = self._run_sampling_eval(
                lambda: (
                    self.collect_target_prefix_trajectories(
                        examples,
                        iteration_index=iteration_index,
                        count=target_prefix_count,
                    ),
                    self.collect_target_teacher_trajectories(
                        examples,
                        iteration_index=iteration_index,
                        count=target_teacher_count,
                    ),
                )
            )
            target_guidance_sampling_duration_sec = (
                perf_counter() - target_guidance_sampling_start
            )
        else:
            replay_sampling_start = perf_counter()
            replay_target_count = _compute_replay_target_count(
                on_policy_count=len(on_policy_trajectories),
                replay_fraction=self.config.replay.replay_fraction,
                legacy_replay_batch_size=self.config.replay.replay_batch_size,
            )
            if self.replay_buffer is not None and replay_target_count > 0:
                replay_sample = self.replay_buffer.sample(
                    replay_target_count,
                    rng=self.replay_rng,
                    with_replacement=self.config.replay.with_replacement,
                )
                replay_trajectories = list(replay_sample.trajectories)
            replay_sampling_duration_sec = perf_counter() - replay_sampling_start

            if self.replay_buffer is not None:
                self.replay_buffer.extend(on_policy_trajectories)

        max_optimization_trajectories_per_iter = (
            self.config.max_optimization_trajectories_per_iter
        )
        optimization_candidates, optimization_trajectories = _select_optimization_trajectories(
            target_teacher_trajectories=target_teacher_trajectories,
            on_policy_trajectories=on_policy_trajectories,
            target_prefix_trajectories=target_prefix_trajectories,
            replay_trajectories=replay_trajectories,
            max_optimization_trajectories_per_iter=(
                max_optimization_trajectories_per_iter
            ),
        )
        optimization_trimmed_count = len(optimization_candidates) - len(
            optimization_trajectories
        )
        retained_replay_count = min(
            len(replay_trajectories),
            max(
                0,
                len(optimization_trajectories)
                - len(target_teacher_trajectories)
                - len(on_policy_trajectories)
                - len(target_prefix_trajectories),
            ),
        )
        target_prefix_batch = OnPolicyBatch.from_trajectories(target_prefix_trajectories)
        target_teacher_batch = OnPolicyBatch.from_trajectories(target_teacher_trajectories)

        if not optimization_trajectories:
            self._last_parallel_scoring_counts = {
                str(device): 0 for device in self.parallel_devices
            }
            iteration_duration_sec = perf_counter() - iteration_start
            metrics: dict[str, Any] = {
                "iteration": float(iteration_index),
                "learning_rate": float(self.optimizer.param_groups[0]["lr"]),
                "objective_loss": 0.0,
                "mean_stage_reward": on_policy_batch.mean_stage_reward(),
                "stage_reward_std": _tensor_std(
                    torch.tensor(
                        [trajectory.terminal_reward for trajectory in on_policy_trajectories],
                        dtype=torch.float32,
                        device=self.device,
                    )
                ),
                "mean_training_stage_reward": 0.0,
                "training_stage_reward_std": 0.0,
                "valid_fraction": on_policy_batch.valid_fraction(),
                "duplicate_fraction": on_policy_batch.duplicate_fraction(),
                "duplicate_count_on_policy": float(duplicate_count_on_policy),
                "mean_num_actions": on_policy_batch.mean_num_actions(),
                "max_num_actions": float(
                    max(
                        (
                            trajectory.num_actions
                            for trajectory in on_policy_trajectories
                        ),
                        default=0,
                    )
                ),
                "mean_stage_index": on_policy_batch.mean_stage_index(),
                "num_on_policy_trajectories": float(len(on_policy_trajectories)),
                "num_on_policy_trajectories_raw": float(len(raw_on_policy_trajectories)),
                "num_on_policy_trajectories_trimmed": float(on_policy_trimmed_count),
                "num_optimization_trajectories_raw": float(len(optimization_candidates)),
                "num_optimization_trajectories": float(len(optimization_trajectories)),
                "num_optimization_trajectories_trimmed": float(optimization_trimmed_count),
                "max_optimization_trajectories_per_iter": float(
                    max_optimization_trajectories_per_iter or 0
                ),
                "num_target_prefix_trajectories": float(len(target_prefix_trajectories)),
                "num_target_teacher_trajectories": float(len(target_teacher_trajectories)),
                "num_target_guided_trajectories": float(
                    len(target_prefix_trajectories) + len(target_teacher_trajectories)
                ),
                "num_replay_trajectories": float(len(replay_trajectories)),
                "target_guidance_on_policy_fraction": (
                    float(self.config.target_guidance.on_policy_fraction)
                    if self.config.target_guidance.enabled
                    else 0.0
                ),
                "target_guidance_prefix_fraction": (
                    float(self.config.target_guidance.target_prefix_rollout_fraction)
                    if self.config.target_guidance.enabled
                    else 0.0
                ),
                "target_guidance_teacher_fraction": (
                    float(self.config.target_guidance.target_teacher_fraction)
                    if self.config.target_guidance.enabled
                    else 0.0
                ),
                "target_prefix_valid_fraction": target_prefix_batch.valid_fraction(),
                "target_teacher_valid_fraction": target_teacher_batch.valid_fraction(),
                "target_prefix_mean_stage_reward": target_prefix_batch.mean_stage_reward(),
                "target_teacher_mean_stage_reward": target_teacher_batch.mean_stage_reward(),
                "configured_replay_fraction": (
                    float(self.config.replay.replay_fraction or 0.0)
                    if self.replay_buffer is not None
                    else 0.0
                ),
                "replay_fraction": 0.0,
                "replay_buffer_type": (
                    str(self.config.replay.buffer_type)
                    if self.replay_buffer is not None
                    else "disabled"
                ),
                "replay_recent_count": float(replay_sample.source_counts.get("recent", 0)),
                "replay_reward_count": float(replay_sample.source_counts.get("reward", 0)),
                "replay_uniform_count": float(replay_sample.source_counts.get("uniform", 0)),
                "replay_tb_residual_count": float(
                    replay_sample.source_counts.get("tb_residual", 0)
                ),
                "replay_size": float(len(self.replay_buffer) if self.replay_buffer is not None else 0),
                "replay_total_action_tokens": float(
                    self.replay_buffer.total_action_tokens if self.replay_buffer is not None else 0
                ),
                "rollout_append_probability": float(self.config.rollout.append_probability),
                "rollout_return_last_valid_trajectory_only": float(self.config.objective == "subtb"),
                "grad_norm": 0.0,
                "sampling_duration_sec": sampling_duration_sec,
                "replay_sampling_duration_sec": replay_sampling_duration_sec,
                "target_guidance_sampling_duration_sec": target_guidance_sampling_duration_sec,
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
                **novelty_metrics,
                **termination_reason_metrics(on_policy_trajectories),
                **rollout_stage_metrics(
                    on_policy_trajectories,
                    max_molecules_per_sequence=self.config.rollout.max_molecules_per_sequence,
                    invalid_terminal_reward=self.config.invalid_terminal_reward,
                ),
                **self._parallel_training_metrics(),
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

        scoring_start = perf_counter()
        scored_trajectories = self.score_trajectories(optimization_trajectories)
        scoring_duration_sec = perf_counter() - scoring_start
        if self.replay_buffer is not None:
            self.replay_buffer.observe_scored(scored_trajectories)

        loss_start = perf_counter()
        loss, diagnostics = self._compute_parallel_objective_loss(
            self._last_parallel_scored_groups,
            scored_trajectories,
        )
        loss_duration_sec = perf_counter() - loss_start

        self.optimizer.zero_grad(set_to_none=True)
        self._zero_parallel_replica_gradients()
        backward_start = perf_counter()
        loss.backward()
        self._aggregate_parallel_replica_gradients()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.model.parameters(),
            self.config.max_grad_norm,
        )
        backward_duration_sec = perf_counter() - backward_start
        optimizer_start = perf_counter()
        self.optimizer.step()
        self._sync_parallel_replicas_from_primary()
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
            "duplicate_count_on_policy": float(duplicate_count_on_policy),
            "mean_num_actions": on_policy_batch.mean_num_actions(),
            "max_num_actions": float(action_counts.max().item()) if action_counts.numel() > 0 else 0.0,
            "mean_stage_index": on_policy_batch.mean_stage_index(),
            "num_on_policy_trajectories": float(len(on_policy_trajectories)),
            "num_on_policy_trajectories_raw": float(len(raw_on_policy_trajectories)),
            "num_on_policy_trajectories_trimmed": float(on_policy_trimmed_count),
            "num_optimization_trajectories_raw": float(len(optimization_candidates)),
            "num_optimization_trajectories": float(len(optimization_trajectories)),
            "num_optimization_trajectories_trimmed": float(optimization_trimmed_count),
            "max_optimization_trajectories_per_iter": float(
                max_optimization_trajectories_per_iter or 0
            ),
            "num_target_prefix_trajectories": float(len(target_prefix_trajectories)),
            "num_target_teacher_trajectories": float(len(target_teacher_trajectories)),
            "num_target_guided_trajectories": float(
                len(target_prefix_trajectories) + len(target_teacher_trajectories)
            ),
            "num_replay_trajectories": float(len(replay_trajectories)),
            "target_guidance_on_policy_fraction": (
                float(self.config.target_guidance.on_policy_fraction)
                if self.config.target_guidance.enabled
                else 0.0
            ),
            "target_guidance_prefix_fraction": (
                float(self.config.target_guidance.target_prefix_rollout_fraction)
                if self.config.target_guidance.enabled
                else 0.0
            ),
            "target_guidance_teacher_fraction": (
                float(self.config.target_guidance.target_teacher_fraction)
                if self.config.target_guidance.enabled
                else 0.0
            ),
            "target_prefix_valid_fraction": target_prefix_batch.valid_fraction(),
            "target_teacher_valid_fraction": target_teacher_batch.valid_fraction(),
            "target_prefix_mean_stage_reward": target_prefix_batch.mean_stage_reward(),
            "target_teacher_mean_stage_reward": target_teacher_batch.mean_stage_reward(),
            "configured_replay_fraction": (
                float(self.config.replay.replay_fraction or 0.0)
                if self.replay_buffer is not None
                else 0.0
            ),
            "replay_fraction": float(
                retained_replay_count / len(optimization_trajectories)
                if optimization_trajectories
                else 0.0
            ),
            "replay_buffer_type": (
                str(self.config.replay.buffer_type)
                if self.replay_buffer is not None
                else "disabled"
            ),
            "replay_recent_count": float(replay_sample.source_counts.get("recent", 0)),
            "replay_reward_count": float(replay_sample.source_counts.get("reward", 0)),
            "replay_uniform_count": float(replay_sample.source_counts.get("uniform", 0)),
            "replay_tb_residual_count": float(
                replay_sample.source_counts.get("tb_residual", 0)
            ),
            "replay_size": float(len(self.replay_buffer) if self.replay_buffer is not None else 0),
            "replay_total_action_tokens": float(
                self.replay_buffer.total_action_tokens if self.replay_buffer is not None else 0
            ),
            "rollout_append_probability": float(self.config.rollout.append_probability),
            "rollout_return_last_valid_trajectory_only": float(self.config.objective == "subtb"),
            "grad_norm": float(grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm),
            "mean_log_pf_token": _tensor_mean(all_log_pf_tokens),
            "mean_log_pb_token": _tensor_mean(all_log_pb_tokens),
            "mean_log_state_flow": _tensor_mean(all_log_state_flows),
            "sampling_duration_sec": sampling_duration_sec,
            "replay_sampling_duration_sec": replay_sampling_duration_sec,
            "target_guidance_sampling_duration_sec": target_guidance_sampling_duration_sec,
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
            **novelty_metrics,
            **termination_reason_metrics(on_policy_trajectories),
            **rollout_stage_metrics(
                on_policy_trajectories,
                max_molecules_per_sequence=self.config.rollout.max_molecules_per_sequence,
                invalid_terminal_reward=self.config.invalid_terminal_reward,
            ),
            **self._parallel_training_metrics(),
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
            or iteration_index == self._final_iteration_index()
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
        checkpoint_dir = save_gflownet_iteration_artifacts(
            output_dir=self.config.output_dir,
            iteration_index=iteration_index,
            model=self.model,
            tokenizer=self.tokenizer,
            config=self.config.to_dict(),
            metrics=metrics,
            trajectories=trajectories,
            create_archive=iteration_index == self._final_iteration_index(),
        )
        last_checkpoint_dir, last_checkpoint_zip = save_gflownet_last_artifacts(
            output_dir=self.config.output_dir,
            model=self.model,
            tokenizer=self.tokenizer,
            config=self.config.to_dict(),
            metrics=metrics,
            trajectories=trajectories,
        )
        self.last_checkpoint_iteration = iteration_index
        self.last_checkpoint_dir = str(last_checkpoint_dir)
        self.last_checkpoint_zip = (
            str(last_checkpoint_zip) if last_checkpoint_zip is not None else None
        )
        return checkpoint_dir

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

        sampler = EpochBatchSampler(
            train_dataset,
            batch_size=gflownet_config.batch_size,
            rng=random.Random(gflownet_config.seed),
        )
        report_accumulator = GFlowNetReportAccumulator()
        start_iteration = int(gflownet_config.start_iteration)
        final_iteration = start_iteration + int(gflownet_config.gflownet_iterations)
        for iteration in range(start_iteration + 1, final_iteration + 1):
            iteration_examples, epoch_index = sampler.next_batch()
            iteration_result = trainer.train_iteration(
                iteration_examples,
                iteration_index=iteration,
            )
            iteration_result.metrics["epoch"] = float(epoch_index)
            report_record = report_accumulator.update(iteration_result.metrics)
            history.append(iteration_result.metrics)
            write_gflownet_history(output_dir, history)
            tracker.log_metrics(
                tracker_headline_metrics(iteration_result.metrics),
                step=iteration,
                prefix="gflownet",
            )
            tracker.log_metrics(
                report_record["main"],
                step=iteration,
                prefix="gflownet_report",
            )
            tracker.log_metrics(
                report_record["appendix"],
                step=iteration,
                prefix="gflownet_report_appendix",
            )
            append_gflownet_report_metrics(output_dir, [report_record])
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
            "start_iteration": gflownet_config.start_iteration,
            "final_iteration": final_iteration,
            "history": history,
            "resolved_checkpoint_source": resolved_checkpoint_path_or_id,
            "resolved_checkpoint_path_or_id": resolved_checkpoint_path_or_id,
            "best_objective_loss": getattr(trainer, "best_objective_loss", None),
            "best_checkpoint_iteration": getattr(trainer, "best_checkpoint_iteration", None),
            "best_checkpoint_dir": getattr(trainer, "best_checkpoint_dir", None),
            "best_checkpoint_zip": getattr(trainer, "best_checkpoint_zip", None),
            "last_checkpoint_iteration": getattr(trainer, "last_checkpoint_iteration", None),
            "last_checkpoint_dir": getattr(trainer, "last_checkpoint_dir", None),
            "last_checkpoint_zip": getattr(trainer, "last_checkpoint_zip", None),
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
