import pytest

from post_training.shared.diagnostics import (
    build_categorized_metric_record,
    categorize_metric_payload,
    iter_categorized_tracker_payloads,
)


def test_categorize_metric_payload_groups_iteration_diagnostics_by_bucket() -> None:
    payload = {
        "iteration": 3.0,
        "mean_total_reward": 1.5,
        "valid_rate": 0.5,
        "termination_fraction_stop_token": 0.75,
        "mean_realized_stage_count": 2.0,
        "sampling_duration_sec": 0.25,
        "standardized_advantage_std": 1.0,
        "grad_norm": 0.9,
    }

    categories = categorize_metric_payload(payload, metadata_keys=("iteration",))

    assert categories == {
        "reward_only": {"mean_total_reward": 1.5},
        "validity": {"valid_rate": 0.5},
        "termination": {"termination_fraction_stop_token": 0.75},
        "stage_rollout": {"mean_realized_stage_count": 2.0},
        "sec_timer": {"sampling_duration_sec": 0.25},
        "optimizer": {"standardized_advantage_std": 1.0},
        "numerics": {"grad_norm": 0.9},
    }


def test_build_categorized_metric_record_and_tracker_payloads_preserve_metadata() -> None:
    payload = {
        "optimizer_step": 25,
        "ppo_iteration": 2,
        "policy_loss": 0.3,
        "grad_norm": 0.8,
        "all_finite": True,
    }

    record = build_categorized_metric_record(
        payload,
        metadata_keys=("optimizer_step", "ppo_iteration"),
    )
    tracker_payloads = iter_categorized_tracker_payloads(
        payload,
        base_prefix="ppo_optimizer",
        metadata_keys=("optimizer_step", "ppo_iteration"),
    )

    assert record == {
        "optimizer_step": 25,
        "ppo_iteration": 2,
        "categories": {
            "optimizer": {"policy_loss": 0.3},
            "numerics": {"grad_norm": 0.8, "all_finite": True},
        },
    }
    assert tracker_payloads == (
        ("ppo_optimizer_optimizer", {"policy_loss": 0.3}),
        ("ppo_optimizer_numerics", {"grad_norm": 0.8, "all_finite": True}),
    )


def test_categorize_metric_payload_rejects_unknown_keys() -> None:
    with pytest.raises(ValueError, match="Unsupported diagnostic metric category"):
        categorize_metric_payload({"mystery_metric": 1.0})
