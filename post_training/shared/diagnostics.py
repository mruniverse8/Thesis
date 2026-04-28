from __future__ import annotations

from collections.abc import Iterable, Mapping
import re
from typing import Any

DIAGNOSTIC_METRIC_CATEGORIES = (
    "reward_only",
    "validity",
    "termination",
    "stage_rollout",
    "sec_timer",
    "optimizer",
    "numerics",
)

_STAGE_PREFIX_PATTERN = re.compile(r"^stage\d+_")
_REWARD_METRIC_KEYS = frozenset(
    {
        "mean_reward",
        "reward_std",
        "mean_return",
        "mean_amplified_reward",
        "mean_total_reward",
        "mean_match_reward",
        "mean_diversity_reward",
        "mean_stage_reward",
        "stage_reward_std",
        "mean_training_stage_reward",
        "training_stage_reward_std",
        "invalid_reward_floor_fraction",
    }
)
_VALIDITY_METRIC_KEYS = frozenset(
    {
        "valid_fraction",
        "duplicate_fraction",
        "valid_rate",
        "duplicate_rate",
    }
)
_STAGE_ROLLOUT_METRIC_KEYS = frozenset(
    {
        "num_stage_trajectories",
        "mean_molecules_per_rollout",
        "mean_action_token_count",
        "max_action_token_count",
        "empty_action_rate",
        "num_on_policy_trajectories_raw",
        "num_on_policy_trajectories_trimmed",
        "num_rollouts",
        "mean_planned_trajectory_length",
        "max_planned_trajectory_length",
        "mean_trajectory_length",
        "max_trajectory_length",
        "rollout_append_probability",
    }
)
_OPTIMIZER_METRIC_KEYS = frozenset(
    {
        "learning_rate",
        "mean_policy_loss",
        "mean_value_loss",
        "mean_kl",
        "mean_entropy",
        "mean_old_value",
        "old_value_std",
        "advantage_raw_mean",
        "advantage_raw_std",
        "standardized_advantage_mean",
        "standardized_advantage_std",
        "mean_old_logprob",
        "mean_reference_logprob",
        "configured_replay_fraction",
        "replay_fraction",
        "replay_buffer_type",
        "replay_recent_count",
        "replay_reward_count",
        "replay_uniform_count",
        "replay_tb_residual_count",
        "replay_size",
        "replay_total_action_tokens",
        "mean_log_pf_token",
        "mean_log_pb_token",
        "mean_log_state_flow",
        "objective_loss",
        "objective_residual_mean",
        "objective_residual_std",
        "mean_root_log_flow",
        "mean_terminal_stop_logprob",
        "mini_batch_size",
        "policy_loss",
        "value_loss",
        "total_loss",
        "entropy_bonus",
        "approx_kl_mean",
        "ratio_mean",
        "ratio_std",
        "clip_fraction",
        "batch_advantage_mean",
        "batch_advantage_std",
        "batch_return_mean",
        "new_value_mean",
    }
)
_NUMERICS_METRIC_KEYS = frozenset({"grad_norm", "all_finite"})


def _is_stage_metric(key: str) -> bool:
    return bool(_STAGE_PREFIX_PATTERN.match(key))


def resolve_metric_category(metric_key: str) -> str:
    if metric_key in _NUMERICS_METRIC_KEYS:
        return "numerics"

    if "termination_fraction_" in metric_key:
        return "termination"

    if metric_key in _VALIDITY_METRIC_KEYS or metric_key.endswith(
        ("_valid_fraction", "_duplicate_fraction")
    ):
        return "validity"

    if (
        metric_key in _REWARD_METRIC_KEYS
        or metric_key.endswith(("_reward", "_reward_std"))
        or metric_key.endswith(("_mean_stage_reward", "_invalid_reward_floor_fraction"))
    ):
        return "reward_only"

    if metric_key.endswith(("_duration_sec", "_per_sec")):
        return "sec_timer"

    if (
        metric_key in _OPTIMIZER_METRIC_KEYS
        or metric_key.startswith("objective_")
        or metric_key.startswith(("advantage_", "standardized_advantage_"))
    ):
        return "optimizer"

    if (
        metric_key in _STAGE_ROLLOUT_METRIC_KEYS
        or metric_key.startswith(("fraction_rollouts_", "trajectory_length_"))
        or (_is_stage_metric(metric_key) and metric_key.endswith(("_num_trajectories", "_mean_num_actions")))
    ):
        return "stage_rollout"

    raise ValueError(f"Unsupported diagnostic metric category for key: {metric_key}")


def categorize_metric_payload(
    payload: Mapping[str, Any],
    *,
    metadata_keys: Iterable[str] = (),
) -> dict[str, dict[str, Any]]:
    metadata_key_set = frozenset(str(key) for key in metadata_keys)
    categories: dict[str, dict[str, Any]] = {
        category: {} for category in DIAGNOSTIC_METRIC_CATEGORIES
    }
    for raw_key, value in payload.items():
        key = str(raw_key)
        if key in metadata_key_set:
            continue
        categories[resolve_metric_category(key)][key] = value
    return {category: metrics for category, metrics in categories.items() if metrics}


def build_categorized_metric_record(
    payload: Mapping[str, Any],
    *,
    metadata_keys: Iterable[str] = (),
    categories: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    record = {str(key): payload[key] for key in metadata_keys if key in payload}
    record["categories"] = dict(
        categories if categories is not None else categorize_metric_payload(payload, metadata_keys=metadata_keys)
    )
    return record


def iter_categorized_tracker_payloads(
    payload: Mapping[str, Any],
    *,
    base_prefix: str,
    metadata_keys: Iterable[str] = (),
) -> tuple[tuple[str, dict[str, Any]], ...]:
    categories = categorize_metric_payload(payload, metadata_keys=metadata_keys)
    return tuple(
        (f"{base_prefix}_{category}", metrics)
        for category, metrics in categories.items()
    )
