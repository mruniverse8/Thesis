from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import json
from typing import Any, Sequence

import torch

from molecules.selfies import decode_biot5_selfies

from .trajectory import SampledStageTrajectory

GFLOWNET_TRACKER_HEADLINE_METRIC_KEYS = frozenset(
    {
        "iteration",
        "learning_rate",
        "objective_loss",
        "mean_stage_reward",
        "stage_reward_std",
        "mean_training_stage_reward",
        "training_stage_reward_std",
        "valid_fraction",
        "duplicate_fraction",
        "mean_num_actions",
        "max_num_actions",
        "mean_stage_index",
        "num_on_policy_trajectories",
        "num_replay_trajectories",
        "replay_fraction",
        "replay_size",
        "replay_total_action_tokens",
        "grad_norm",
        "all_finite",
    }
)


def _truncate_text(text: str, *, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3] + "..."


def _render_sequence(values: Sequence[str | None], *, max_chars: int) -> str:
    rendered = " | ".join(value or "<none>" for value in values)
    return _truncate_text(rendered, max_chars=max_chars)


def _decode_action_text(tokenizer, action_token_ids: Sequence[int]) -> str | None:
    if tokenizer is None or not hasattr(tokenizer, "decode") or not action_token_ids:
        return None
    try:
        decoded = tokenizer.decode(
            list(action_token_ids),
            skip_special_tokens=False,
            clean_up_tokenization_spaces=True,
        ).strip()
    except Exception:
        return None
    return decoded or None


def _cleanup_decode_stage_text(stage_text: str) -> dict[str, Any]:
    try:
        return decode_biot5_selfies(stage_text)
    except Exception as exc:
        return {
            "selected_selfies": None,
            "is_valid_selfies": False,
            "used_filter_selfies_fallback": False,
            "selfies_decode_error": f"{type(exc).__name__}: {exc}",
        }


def all_finite(*tensors: torch.Tensor) -> bool:
    return all(bool(torch.isfinite(tensor).all().item()) for tensor in tensors if tensor.numel() > 0)


def stack_scalar_likes(
    values: Sequence[float | torch.Tensor],
    *,
    device: torch.device,
) -> torch.Tensor:
    if not values:
        return torch.zeros(0, dtype=torch.float32, device=device)

    stacked: list[torch.Tensor] = []
    for value in values:
        if isinstance(value, torch.Tensor):
            stacked.append(value.detach().to(device=device, dtype=torch.float32).reshape(()))
            continue
        stacked.append(torch.tensor(float(value), dtype=torch.float32, device=device))
    return torch.stack(stacked)


def safe_rate(count: int | float, duration_sec: float) -> float:
    if duration_sec <= 0.0:
        return 0.0
    return float(count) / duration_sec


def tracker_headline_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    return {
        key: metrics[key]
        for key in GFLOWNET_TRACKER_HEADLINE_METRIC_KEYS
        if key in metrics
    }


def tracker_diagnostic_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    diagnostic_metrics = {
        key: value
        for key, value in metrics.items()
        if key not in GFLOWNET_TRACKER_HEADLINE_METRIC_KEYS and key != "iteration"
    }
    if "iteration" in metrics:
        return {"iteration": metrics["iteration"], **diagnostic_metrics}
    return diagnostic_metrics


def _normalize_metric_key_component(value: str) -> str:
    return value.strip().lower().replace("-", "_").replace(" ", "_")


def termination_reason_metrics(
    trajectories: Sequence[SampledStageTrajectory],
    *,
    prefix: str = "termination_fraction_",
) -> dict[str, float]:
    base_reasons = (
        "stop_token",
        "eos_token",
        "max_stage_new_tokens",
        "max_sequence_length",
    )
    if not trajectories:
        return {f"{prefix}{reason}": 0.0 for reason in base_reasons}

    counts: dict[str, int] = defaultdict(int)
    for trajectory in trajectories:
        counts[str(trajectory.termination_reason)] += 1

    total = len(trajectories)
    metrics = {f"{prefix}{reason}": counts.get(reason, 0) / total for reason in base_reasons}
    for reason, count in sorted(counts.items()):
        metric_key = f"{prefix}{_normalize_metric_key_component(reason)}"
        metrics.setdefault(metric_key, count / total)
    return metrics


def rollout_stage_metrics(
    trajectories: Sequence[SampledStageTrajectory],
    *,
    max_molecules_per_sequence: int | None = None,
    invalid_terminal_reward: float | None = None,
    stage_indices: Sequence[int] = (1, 2),
) -> dict[str, float]:
    metrics: dict[str, float] = {
        "num_rollouts": 0.0,
        "max_stage_index": 0.0,
        "mean_planned_stage_count": 0.0,
        "max_planned_stage_count": 0.0,
        "mean_realized_stage_count": 0.0,
        "max_realized_stage_count": 0.0,
        "fraction_rollouts_planned_stage_2_plus": 0.0,
        "fraction_rollouts_reaching_stage_2": 0.0,
        "fraction_rollouts_reaching_stage_3_plus": 0.0,
        "fraction_rollouts_reaching_planned_stage_count": 0.0,
        "rollout_stage_count_1_fraction": 0.0,
        "rollout_stage_count_2_fraction": 0.0,
        "rollout_stage_count_3_plus_fraction": 0.0,
        "invalid_reward_floor_fraction": 0.0,
    }
    for stage_index in stage_indices:
        stage_prefix = f"stage{stage_index}_"
        metrics.update(
            {
                f"{stage_prefix}num_trajectories": 0.0,
                f"{stage_prefix}valid_fraction": 0.0,
                f"{stage_prefix}mean_stage_reward": 0.0,
                f"{stage_prefix}mean_num_actions": 0.0,
                f"{stage_prefix}invalid_reward_floor_fraction": 0.0,
            }
        )
        metrics.update(
            termination_reason_metrics(
                (),
                prefix=f"{stage_prefix}termination_fraction_",
            )
        )

    if not trajectories:
        return metrics

    grouped_rollouts: dict[str, list[SampledStageTrajectory]] = defaultdict(list)
    for trajectory in trajectories:
        grouped_rollouts[trajectory.rollout_id].append(trajectory)

    planned_stage_counts: list[int] = []
    realized_stage_counts: list[int] = []
    reached_stage_2 = 0
    reached_stage_3_plus = 0
    planned_stage_2_plus = 0
    reached_planned_stage_count = 0

    for rollout in grouped_rollouts.values():
        ordered_rollout = sorted(rollout, key=lambda item: item.stage_index)
        realized_stage_count = len(ordered_rollout)
        target_stage_count = max(1, len(ordered_rollout[0].target_selfies_list))
        if max_molecules_per_sequence is not None:
            target_stage_count = min(target_stage_count, int(max_molecules_per_sequence))
        planned_stage_counts.append(target_stage_count)
        realized_stage_counts.append(realized_stage_count)
        if target_stage_count >= 2:
            planned_stage_2_plus += 1
        if realized_stage_count >= 2:
            reached_stage_2 += 1
        if realized_stage_count >= 3:
            reached_stage_3_plus += 1
        if realized_stage_count >= target_stage_count:
            reached_planned_stage_count += 1

    num_rollouts = len(grouped_rollouts)
    metrics.update(
        {
            "num_rollouts": float(num_rollouts),
            "max_stage_index": float(max(trajectory.stage_index for trajectory in trajectories)),
            "mean_planned_stage_count": float(sum(planned_stage_counts) / num_rollouts),
            "max_planned_stage_count": float(max(planned_stage_counts)),
            "mean_realized_stage_count": float(sum(realized_stage_counts) / num_rollouts),
            "max_realized_stage_count": float(max(realized_stage_counts)),
            "fraction_rollouts_planned_stage_2_plus": float(planned_stage_2_plus / num_rollouts),
            "fraction_rollouts_reaching_stage_2": float(reached_stage_2 / num_rollouts),
            "fraction_rollouts_reaching_stage_3_plus": float(reached_stage_3_plus / num_rollouts),
            "fraction_rollouts_reaching_planned_stage_count": float(
                reached_planned_stage_count / num_rollouts
            ),
            "rollout_stage_count_1_fraction": float(
                sum(int(count == 1) for count in realized_stage_counts) / num_rollouts
            ),
            "rollout_stage_count_2_fraction": float(
                sum(int(count == 2) for count in realized_stage_counts) / num_rollouts
            ),
            "rollout_stage_count_3_plus_fraction": float(
                sum(int(count >= 3) for count in realized_stage_counts) / num_rollouts
            ),
        }
    )

    if invalid_terminal_reward is not None:
        metrics["invalid_reward_floor_fraction"] = float(
            sum(
                int(abs(float(trajectory.terminal_reward) - float(invalid_terminal_reward)) <= 1.0e-12)
                for trajectory in trajectories
            )
            / len(trajectories)
        )

    for stage_index in stage_indices:
        stage_trajectories = [trajectory for trajectory in trajectories if trajectory.stage_index == stage_index]
        if not stage_trajectories:
            continue
        stage_prefix = f"stage{stage_index}_"
        metrics.update(
            {
                f"{stage_prefix}num_trajectories": float(len(stage_trajectories)),
                f"{stage_prefix}valid_fraction": float(
                    sum(int(trajectory.is_valid) for trajectory in stage_trajectories)
                    / len(stage_trajectories)
                ),
                f"{stage_prefix}mean_stage_reward": float(
                    sum(float(trajectory.terminal_reward) for trajectory in stage_trajectories)
                    / len(stage_trajectories)
                ),
                f"{stage_prefix}mean_num_actions": float(
                    sum(trajectory.num_actions for trajectory in stage_trajectories)
                    / len(stage_trajectories)
                ),
            }
        )
        if invalid_terminal_reward is not None:
            metrics[f"{stage_prefix}invalid_reward_floor_fraction"] = float(
                sum(
                    int(
                        abs(float(trajectory.terminal_reward) - float(invalid_terminal_reward))
                        <= 1.0e-12
                    )
                    for trajectory in stage_trajectories
                )
                / len(stage_trajectories)
            )
        metrics.update(
            termination_reason_metrics(
                stage_trajectories,
                prefix=f"{stage_prefix}termination_fraction_",
            )
        )

    return metrics


@dataclass(frozen=True, slots=True)
class GFlowNetTrainIterationResult:
    metrics: dict[str, Any]
    diagnostic_metrics: dict[str, Any] | None = None
    categorized_diagnostic_metrics: dict[str, dict[str, Any]] | None = None
    trajectory_preview: dict[str, Any] | None = None


def build_trajectory_preview_payload(
    trajectories: Sequence[SampledStageTrajectory],
    *,
    iteration_index: int,
    num_samples: int,
    max_chars: int,
    tokenizer,
) -> dict[str, Any] | None:
    if not trajectories:
        return None

    grouped_trajectories: dict[str, list[SampledStageTrajectory]] = defaultdict(list)
    for trajectory in trajectories:
        grouped_trajectories[trajectory.rollout_id].append(trajectory)

    preview_candidates: list[dict[str, Any]] = []
    for rollout in grouped_trajectories.values():
        ordered_rollout = sorted(rollout, key=lambda trajectory: trajectory.stage_index)
        raw_stage_texts = [
            str(trajectory.metadata.get("raw_stage_text", trajectory.stage_text))
            for trajectory in ordered_rollout
        ]
        cleanup_results = [
            _cleanup_decode_stage_text(raw_stage_text) for raw_stage_text in raw_stage_texts
        ]
        preview_candidates.append(
            {
                "iteration": iteration_index,
                "rollout_id": ordered_rollout[0].rollout_id,
                "example_id": ordered_rollout[0].example_id,
                "num_stages": len(ordered_rollout),
                "total_reward": float(
                    sum(float(trajectory.terminal_reward) for trajectory in ordered_rollout)
                ),
                "stage_indices": [int(trajectory.stage_index) for trajectory in ordered_rollout],
                "stage_rewards": [
                    float(trajectory.terminal_reward) for trajectory in ordered_rollout
                ],
                "generated_selfies_sequence": [
                    trajectory.sampled_selfies for trajectory in ordered_rollout
                ],
                "raw_stage_text_sequence": raw_stage_texts,
                "cleanup_selected_selfies_sequence": [
                    result.get("selected_selfies") for result in cleanup_results
                ],
                "recoverable_by_cleanup_sequence": [
                    bool(result.get("is_valid_selfies")) and trajectory.sampled_selfies is None
                    for trajectory, result in zip(ordered_rollout, cleanup_results)
                ],
                "new_action_token_ids_sequence": [
                    list(trajectory.action_token_ids) for trajectory in ordered_rollout
                ],
                "new_action_text_sequence": [
                    _decode_action_text(tokenizer, trajectory.action_token_ids)
                    for trajectory in ordered_rollout
                ],
                "valid_sequence": [bool(trajectory.is_valid) for trajectory in ordered_rollout],
                "duplicate_sequence": [
                    bool(trajectory.is_duplicate) for trajectory in ordered_rollout
                ],
                "termination_reasons": [
                    str(trajectory.termination_reason) for trajectory in ordered_rollout
                ],
            }
        )

    ordered_candidates = sorted(
        preview_candidates,
        key=lambda item: (
            float(item["total_reward"]),
            str(item["example_id"]),
            str(item["rollout_id"]),
        ),
    )

    selected_records: list[dict[str, Any]] = []
    selected_indices: set[int] = set()
    preferred_indices = [
        ("best", len(ordered_candidates) - 1),
        ("median", (len(ordered_candidates) - 1) // 2),
        ("worst", 0),
    ]
    for preview_slot, index in preferred_indices:
        if index in selected_indices:
            continue
        selected_indices.add(index)
        record = dict(ordered_candidates[index])
        record["preview_slot"] = preview_slot
        selected_records.append(record)
        if len(selected_records) >= num_samples:
            break

    if len(selected_records) < num_samples:
        for index in range(len(ordered_candidates) - 1, -1, -1):
            if index in selected_indices:
                continue
            selected_indices.add(index)
            record = dict(ordered_candidates[index])
            record["preview_slot"] = f"extra_{len(selected_records) + 1}"
            selected_records.append(record)
            if len(selected_records) >= num_samples:
                break

    rendered_sections: list[str] = []
    for record in selected_records:
        action_text_sequence = record["new_action_text_sequence"]
        if any(item is not None for item in action_text_sequence):
            rendered_actions = _render_sequence(action_text_sequence, max_chars=max_chars)
        else:
            rendered_actions = _render_sequence(
                [json.dumps(item) for item in record["new_action_token_ids_sequence"]],
                max_chars=max_chars,
            )
        rendered_sections.append(
            "\n".join(
                [
                    (
                        f"[{record['preview_slot']}] example_id={record['example_id']} "
                        f"rollout_id={record['rollout_id']} total_reward={record['total_reward']:.4f}"
                    ),
                    (
                        "generated_selfies="
                        + _render_sequence(
                            record["generated_selfies_sequence"],
                            max_chars=max_chars,
                        )
                    ),
                    (
                        "raw_stage_text="
                        + _render_sequence(
                            record["raw_stage_text_sequence"],
                            max_chars=max_chars,
                        )
                    ),
                    (
                        "cleanup_selected_selfies="
                        + _render_sequence(
                            record["cleanup_selected_selfies_sequence"],
                            max_chars=max_chars,
                        )
                    ),
                    f"new_actions={rendered_actions}",
                    f"stage_rewards={record['stage_rewards']}",
                    f"termination_reasons={record['termination_reasons']}",
                    f"valid_sequence={record['valid_sequence']}",
                    f"duplicate_sequence={record['duplicate_sequence']}",
                    f"recoverable_by_cleanup={record['recoverable_by_cleanup_sequence']}",
                ]
            )
        )

    return {
        "iteration": iteration_index,
        "records": selected_records,
        "tracker_text": "\n\n".join(rendered_sections),
    }
