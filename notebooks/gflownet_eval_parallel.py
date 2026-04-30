"""Shared helpers for sharded multi-worker GFlowNet evaluation."""

from __future__ import annotations

from dataclasses import dataclass
import random
from pathlib import Path
from typing import TYPE_CHECKING, Any, Sequence

from post_training.gflownet.trajectory import SampledStageTrajectory

if TYPE_CHECKING:
    from evaluation_metrics import EvaluationMetricConfig, EvaluationMetricsResult, GenerationGroup


@dataclass(frozen=True, slots=True)
class ContiguousShard:
    shard_index: int
    num_shards: int
    start: int
    end: int

    @property
    def size(self) -> int:
        return max(0, int(self.end) - int(self.start))

    @property
    def global_example_indices(self) -> tuple[int, ...]:
        return tuple(range(int(self.start), int(self.end)))


def select_examples(
    dataset: Sequence[dict[str, object]],
    *,
    max_examples: int | None,
    fraction: float | None,
    seed: int,
    with_replacement: bool,
) -> tuple[list[dict[str, object]], list[int]]:
    examples = [dataset[index] for index in range(len(dataset))]
    dataset_size = len(examples)
    if dataset_size == 0:
        return [], []

    if fraction is not None:
        selected_count = int(round(dataset_size * float(fraction)))
    elif max_examples is not None:
        selected_count = int(max_examples)
    else:
        selected_count = dataset_size

    selected_count = max(1, min(selected_count, dataset_size))
    rng = random.Random(int(seed))

    if selected_count >= dataset_size:
        selected_indices = list(range(dataset_size))
    elif with_replacement:
        selected_indices = [rng.randrange(dataset_size) for _ in range(selected_count)]
    else:
        selected_indices = rng.sample(range(dataset_size), selected_count)

    return [examples[index] for index in selected_indices], selected_indices


def contiguous_shard(
    total_items: int,
    *,
    shard_index: int,
    num_shards: int,
) -> ContiguousShard:
    total_items = int(total_items)
    shard_index = int(shard_index)
    num_shards = int(num_shards)
    if total_items < 0:
        raise ValueError("total_items must be non-negative.")
    if num_shards <= 0:
        raise ValueError("num_shards must be positive.")
    if shard_index < 0 or shard_index >= num_shards:
        raise ValueError("shard_index must satisfy 0 <= shard_index < num_shards.")

    base_size = total_items // num_shards
    remainder = total_items % num_shards
    start = shard_index * base_size + min(shard_index, remainder)
    shard_size = base_size + int(shard_index < remainder)
    return ContiguousShard(
        shard_index=shard_index,
        num_shards=num_shards,
        start=start,
        end=start + shard_size,
    )


def shard_selected_examples(
    examples: Sequence[dict[str, object]],
    selected_indices: Sequence[int],
    *,
    shard_index: int,
    num_shards: int,
) -> tuple[list[dict[str, object]], list[int], ContiguousShard]:
    if len(examples) != len(selected_indices):
        raise ValueError("examples and selected_indices must have the same length.")
    shard = contiguous_shard(
        len(examples),
        shard_index=shard_index,
        num_shards=num_shards,
    )
    return (
        list(examples[shard.start : shard.end]),
        [int(index) for index in selected_indices[shard.start : shard.end]],
        shard,
    )


def batched(items: Sequence[dict[str, object]], batch_size: int):
    batch_size = int(batch_size)
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    for start in range(0, len(items), batch_size):
        yield start, list(items[start : start + batch_size])


def selected_indices_snapshot(
    selected_indices: Sequence[int],
    *,
    limit: int = 1000,
) -> dict[str, object]:
    truncated = len(selected_indices) > int(limit)
    return {
        "selected_indices": [int(index) for index in selected_indices[: int(limit)]],
        "selected_indices_count": len(selected_indices),
        "selected_indices_truncated": truncated,
    }


def build_generation_group_and_row(
    example: dict[str, object],
    trajectories: Sequence[SampledStageTrajectory],
    *,
    split_name: str,
    global_example_index: int,
    selected_dataset_index: int,
    parallel_mode: bool,
    parallel_workers: int,
    worker_shard_index: int,
    worker_num_shards: int,
) -> tuple["GenerationGroup", dict[str, object]]:
    from evaluation_metrics import GenerationGroup, MoleculeInput

    generated_selfies = [
        trajectory.sampled_selfies
        for trajectory in trajectories
        if trajectory.sampled_selfies
    ]
    group = GenerationGroup(
        group_id=str(example["id"]),
        candidates=tuple(
            MoleculeInput(text=str(selfies), representation="selfies")
            for selfies in generated_selfies
        ),
        targets=tuple(
            MoleculeInput(text=str(selfies), representation="selfies")
            for selfies in example["target_selfies_list"]
        ),
    )
    row = {
        "id": str(example["id"]),
        "split": str(split_name),
        "description": str(example["description"]),
        "prompt": str(example["prompt"]),
        "target_selfies_list": [str(selfies) for selfies in example["target_selfies_list"]],
        "sampled_selfies_list": [str(selfies) for selfies in generated_selfies],
        "num_trajectories": len(trajectories),
        "num_generated_selfies": len(generated_selfies),
        "trajectory_stage_indices": [int(trajectory.stage_index) for trajectory in trajectories],
        "termination_reasons": [str(trajectory.termination_reason) for trajectory in trajectories],
        "global_example_index": int(global_example_index),
        "selected_dataset_index": int(selected_dataset_index),
        "parallel_mode": bool(parallel_mode),
        "parallel_workers": int(max(1, parallel_workers)),
        "worker_shard_index": int(worker_shard_index),
        "worker_num_shards": int(max(1, worker_num_shards)),
    }
    return group, row


def build_generation_group_from_row(row: dict[str, object]) -> "GenerationGroup":
    from evaluation_metrics import GenerationGroup, MoleculeInput

    return GenerationGroup(
        group_id=str(row["id"]),
        candidates=tuple(
            MoleculeInput(text=str(selfies), representation="selfies")
            for selfies in row.get("sampled_selfies_list", [])
        ),
        targets=tuple(
            MoleculeInput(text=str(selfies), representation="selfies")
            for selfies in row.get("target_selfies_list", [])
        ),
    )


def sort_generation_rows_by_global_example_index(
    rows: Sequence[dict[str, object]],
) -> list[dict[str, object]]:
    return sorted(rows, key=lambda row: int(row["global_example_index"]))


def evaluate_generation_rows(
    rows: Sequence[dict[str, object]],
    *,
    metric_config: "EvaluationMetricConfig",
) -> "EvaluationMetricsResult":
    from evaluation_metrics import evaluate_generation_groups

    ordered_rows = sort_generation_rows_by_global_example_index(rows)
    groups = [build_generation_group_from_row(row) for row in ordered_rows]
    return evaluate_generation_groups(groups, config=metric_config)


def compact_metrics(
    result: "EvaluationMetricsResult",
    *,
    split_name: str,
    progress_examples: int | None = None,
    mean_trajectory_length: float,
) -> dict[str, float | int]:
    prefix = f"eval/{split_name}"
    metrics: dict[str, float | int] = {
        f"{prefix}/accepted_unique_count": result.accepted_unique_count,
        f"{prefix}/n_circles": result.n_circles,
        f"{prefix}/valid_fraction": result.valid_fraction,
        f"{prefix}/internal_diversity": result.internal_diversity,
        f"{prefix}/mean_max_dice_similarity": result.mean_max_dice_similarity,
        f"{prefix}/novelty_fraction": result.novelty_fraction,
        f"{prefix}/novelty_count": result.novelty_count,
        f"{prefix}/prefix_valid_fraction": result.prefix_valid_fraction,
        f"{prefix}/prefix_duplicate_fraction": result.prefix_duplicate_fraction,
        f"{prefix}/prefix_duplicate_valid_fraction": result.prefix_duplicate_valid_fraction,
        f"{prefix}/prefix_average_max_dice_similarity": result.prefix_average_max_dice_similarity,
        f"{prefix}/prefix_accepted_unique_internal_diversity": result.prefix_accepted_unique_internal_diversity,
        f"{prefix}/prefix_valid_internal_diversity": result.prefix_valid_internal_diversity,
        f"{prefix}/mean_trajectory_length": float(mean_trajectory_length),
    }
    if progress_examples is not None:
        metrics[f"{prefix}/progress_examples"] = int(progress_examples)
    if split_name == "validation":
        metrics.update(
            {
                "eval/accepted_unique_count": result.accepted_unique_count,
                "eval/n_circles": result.n_circles,
                "eval/valid_fraction": result.valid_fraction,
                "eval/internal_diversity": result.internal_diversity,
                "eval/mean_max_dice_similarity": result.mean_max_dice_similarity,
                "eval/novelty_fraction": result.novelty_fraction,
                "eval/novelty_count": result.novelty_count,
                "eval/prefix_valid_fraction": result.prefix_valid_fraction,
                "eval/prefix_duplicate_fraction": result.prefix_duplicate_fraction,
                "eval/prefix_duplicate_valid_fraction": result.prefix_duplicate_valid_fraction,
                "eval/prefix_average_max_dice_similarity": result.prefix_average_max_dice_similarity,
                "eval/prefix_accepted_unique_internal_diversity": (
                    result.prefix_accepted_unique_internal_diversity
                ),
                "eval/prefix_valid_internal_diversity": result.prefix_valid_internal_diversity,
                "eval/mean_trajectory_length": float(mean_trajectory_length),
            }
        )
    return metrics


def build_progress_payload(
    result: "EvaluationMetricsResult",
    *,
    split_name: str,
    dataset_path: Path,
    dataset_size: int,
    num_selected_examples: int,
    evaluated_examples: int,
    selected_fraction: float | None,
    sample_with_replacement: bool,
    selected_indices: Sequence[int],
    mean_trajectory_length: float,
    parallel_mode: bool,
    parallel_workers: int,
    worker_shard_index: int,
    worker_num_shards: int,
    shard_start_index: int,
    shard_end_index: int,
    selection_seed: int,
    eval_batch_size: int,
) -> dict[str, object]:
    payload = {
        "split_name": split_name,
        "dataset_path": str(dataset_path),
        "dataset_size": int(dataset_size),
        "num_selected_examples": int(num_selected_examples),
        "evaluated_examples": int(evaluated_examples),
        "selected_fraction": selected_fraction,
        "selection_seed": int(selection_seed),
        "sample_with_replacement": bool(sample_with_replacement),
        "accepted_unique_count": result.accepted_unique_count,
        "num_candidates": result.num_candidates,
        "num_valid_candidates": result.num_valid_candidates,
        "valid_fraction": result.valid_fraction,
        "internal_diversity": result.internal_diversity,
        "mean_max_dice_similarity": result.mean_max_dice_similarity,
        "prefix_valid_fraction": result.prefix_valid_fraction,
        "prefix_duplicate_fraction": result.prefix_duplicate_fraction,
        "prefix_duplicate_valid_fraction": result.prefix_duplicate_valid_fraction,
        "prefix_average_max_dice_similarity": result.prefix_average_max_dice_similarity,
        "prefix_accepted_unique_internal_diversity": result.prefix_accepted_unique_internal_diversity,
        "prefix_valid_internal_diversity": result.prefix_valid_internal_diversity,
        "novelty_fraction": result.novelty_fraction,
        "eval_batch_size": int(eval_batch_size),
        "mean_trajectory_length": float(mean_trajectory_length),
        "parallel_mode": bool(parallel_mode),
        "parallel_workers": int(max(1, parallel_workers)),
        "worker_shard_index": int(worker_shard_index),
        "worker_num_shards": int(max(1, worker_num_shards)),
        "shard_start_index": int(shard_start_index),
        "shard_end_index": int(shard_end_index),
    }
    payload.update(selected_indices_snapshot(selected_indices))
    return payload


def build_final_metrics_payload(
    result: "EvaluationMetricsResult",
    *,
    split_name: str,
    dataset_path: Path,
    dataset_size: int,
    num_selected_examples: int,
    selected_fraction: float | None,
    sample_with_replacement: bool,
    selected_indices: Sequence[int],
    report_every_examples: int,
    eval_batch_size: int,
    checkpoint_for_eval: Path,
    rollout_diagnostics: dict[str, float],
    selection_seed: int,
    parallel_mode: bool,
    parallel_workers: int,
    worker_shard_index: int | None = None,
    worker_num_shards: int | None = None,
    shard_start_index: int | None = None,
    shard_end_index: int | None = None,
    extra_metadata: dict[str, object] | None = None,
) -> dict[str, object]:
    payload = result.to_dict(include_assessments=False)
    payload.update(
        {
            "split_name": split_name,
            "dataset_path": str(dataset_path),
            "dataset_size": int(dataset_size),
            "num_examples": int(num_selected_examples),
            "num_selected_examples": int(num_selected_examples),
            "selection_fraction": selected_fraction,
            "selection_seed": int(selection_seed),
            "sample_with_replacement": bool(sample_with_replacement),
            "report_every_examples": int(report_every_examples),
            "eval_batch_size": int(eval_batch_size),
            "checkpoint_for_eval": str(checkpoint_for_eval),
            "parallel_mode": bool(parallel_mode),
            "parallel_workers": int(max(1, parallel_workers)),
            "mean_trajectory_length": float(rollout_diagnostics.get("mean_trajectory_length", 0.0)),
        }
    )
    payload.update(selected_indices_snapshot(selected_indices))
    for metric_key, metric_value in rollout_diagnostics.items():
        payload[str(metric_key)] = float(metric_value)
    if worker_shard_index is not None:
        payload["worker_shard_index"] = int(worker_shard_index)
    if worker_num_shards is not None:
        payload["worker_num_shards"] = int(worker_num_shards)
    if shard_start_index is not None:
        payload["shard_start_index"] = int(shard_start_index)
    if shard_end_index is not None:
        payload["shard_end_index"] = int(shard_end_index)
    if extra_metadata:
        payload.update(extra_metadata)
    return payload
