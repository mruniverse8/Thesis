from types import SimpleNamespace

import pytest

from post_training.ppo.config import StageTrajectory
from post_training.ppo.diagnostics import (
    build_trajectory_preview_payload,
    rollout_stage_metrics,
    termination_reason_metrics,
)


def _make_reward_breakdown(
    *,
    amplified_reward: float,
    total_reward: float | None = None,
    match_reward: float = 0.0,
    diversity_reward: float = 0.0,
    is_duplicate: bool = False,
    is_valid: bool = True,
):
    return SimpleNamespace(
        amplified_reward=amplified_reward,
        total_reward=amplified_reward if total_reward is None else total_reward,
        match=SimpleNamespace(reward=match_reward),
        diversity=SimpleNamespace(reward=diversity_reward),
        is_duplicate=is_duplicate,
        candidate=SimpleNamespace(is_valid=is_valid, canonical_smiles="C"),
    )


def _make_stage_trajectory(
    *,
    rollout_id: str,
    example_id: str,
    stage_index: int,
    reward: float,
    action_token_ids: tuple[int, ...] = (1, 2),
    target_selfies_list: tuple[str, ...] = ("target",),
    sampled_selfies: str | None = "A",
    stage_text: str = "A",
    termination_reason: str = "stop_token",
    is_valid: bool = True,
    is_duplicate: bool = False,
) -> StageTrajectory:
    return StageTrajectory(
        rollout_id=rollout_id,
        example_id=example_id,
        prompt_text="prompt",
        description="description",
        target_selfies_list=target_selfies_list,
        stage_index=stage_index,
        decoder_prefix_text="",
        stage_text=stage_text,
        sampled_selfies=sampled_selfies,
        stop_token="<eom>" if termination_reason == "stop_token" else None,
        termination_reason=termination_reason,
        action_token_ids=action_token_ids,
        action_logprob_sum_old=-1.0,
        reference_logprob_sum=-1.1,
        value_old=0.0,
        reward_breakdown=_make_reward_breakdown(
            amplified_reward=reward,
            total_reward=reward,
            match_reward=reward / 2.0,
            diversity_reward=reward / 4.0,
            is_duplicate=is_duplicate,
            is_valid=is_valid,
        ),
        reward=reward,
        entropy_sum_old=0.0,
        is_valid=is_valid,
        is_duplicate=is_duplicate,
    )


def test_termination_reason_metrics_supports_custom_prefix() -> None:
    trajectories = (
        _make_stage_trajectory(
            rollout_id="rollout-1",
            example_id="example-1",
            stage_index=1,
            reward=2.0,
            termination_reason="stop_token",
        ),
        _make_stage_trajectory(
            rollout_id="rollout-2",
            example_id="example-2",
            stage_index=1,
            reward=1.0,
            termination_reason="max_stage_new_tokens",
            is_valid=False,
        ),
    )

    metrics = termination_reason_metrics(trajectories, prefix="stage1_termination_fraction_")

    assert metrics["stage1_termination_fraction_stop_token"] == pytest.approx(0.5)
    assert metrics["stage1_termination_fraction_max_stage_new_tokens"] == pytest.approx(0.5)
    assert metrics["stage1_termination_fraction_eos_token"] == pytest.approx(0.0)


def test_rollout_stage_metrics_capture_realized_lengths_and_stage_breakdown() -> None:
    trajectories = (
        _make_stage_trajectory(
            rollout_id="rollout-1",
            example_id="example-1",
            stage_index=1,
            reward=1.0,
            action_token_ids=(1,),
            target_selfies_list=("A",),
            sampled_selfies="A",
            stage_text="A",
        ),
        _make_stage_trajectory(
            rollout_id="rollout-2",
            example_id="example-2",
            stage_index=1,
            reward=2.0,
            action_token_ids=(1, 2),
            target_selfies_list=("A", "B"),
            sampled_selfies="B1",
            stage_text="B1",
        ),
        _make_stage_trajectory(
            rollout_id="rollout-2",
            example_id="example-2",
            stage_index=2,
            reward=3.0,
            action_token_ids=(3,),
            target_selfies_list=("A", "B"),
            sampled_selfies="B2",
            stage_text="B2",
        ),
        _make_stage_trajectory(
            rollout_id="rollout-3",
            example_id="example-3",
            stage_index=1,
            reward=0.5,
            action_token_ids=(4, 5, 6),
            target_selfies_list=("A", "B", "C", "D"),
            sampled_selfies=None,
            stage_text="C1",
            termination_reason="max_stage_new_tokens",
            is_valid=False,
            is_duplicate=True,
        ),
        _make_stage_trajectory(
            rollout_id="rollout-3",
            example_id="example-3",
            stage_index=2,
            reward=0.75,
            action_token_ids=(),
            target_selfies_list=("A", "B", "C", "D"),
            sampled_selfies=None,
            stage_text="C2",
            termination_reason="max_sequence_length",
            is_valid=False,
        ),
        _make_stage_trajectory(
            rollout_id="rollout-3",
            example_id="example-3",
            stage_index=3,
            reward=1.25,
            action_token_ids=(7, 8),
            target_selfies_list=("A", "B", "C", "D"),
            sampled_selfies="C3",
            stage_text="C3",
        ),
    )

    metrics = rollout_stage_metrics(
        trajectories,
        max_molecules_per_sequence=8,
    )

    assert metrics["num_rollouts"] == pytest.approx(3.0)
    assert metrics["max_stage_index"] == pytest.approx(3.0)
    assert metrics["mean_planned_stage_count"] == pytest.approx(7.0 / 3.0)
    assert metrics["max_planned_stage_count"] == pytest.approx(4.0)
    assert metrics["mean_realized_stage_count"] == pytest.approx(2.0)
    assert metrics["max_realized_stage_count"] == pytest.approx(3.0)
    assert metrics["fraction_rollouts_planned_stage_2_plus"] == pytest.approx(2.0 / 3.0)
    assert metrics["fraction_rollouts_reaching_stage_2"] == pytest.approx(2.0 / 3.0)
    assert metrics["fraction_rollouts_reaching_stage_3_plus"] == pytest.approx(1.0 / 3.0)
    assert metrics["fraction_rollouts_reaching_planned_stage_count"] == pytest.approx(2.0 / 3.0)
    assert metrics["rollout_stage_count_1_fraction"] == pytest.approx(1.0 / 3.0)
    assert metrics["rollout_stage_count_2_fraction"] == pytest.approx(1.0 / 3.0)
    assert metrics["rollout_stage_count_3_plus_fraction"] == pytest.approx(1.0 / 3.0)
    assert metrics["stage1_num_trajectories"] == pytest.approx(3.0)
    assert metrics["stage1_valid_fraction"] == pytest.approx(2.0 / 3.0)
    assert metrics["stage1_duplicate_fraction"] == pytest.approx(1.0 / 3.0)
    assert metrics["stage1_mean_stage_reward"] == pytest.approx((1.0 + 2.0 + 0.5) / 3.0)
    assert metrics["stage1_mean_num_actions"] == pytest.approx(2.0)
    assert metrics["stage1_termination_fraction_stop_token"] == pytest.approx(2.0 / 3.0)
    assert metrics["stage1_termination_fraction_max_stage_new_tokens"] == pytest.approx(1.0 / 3.0)
    assert metrics["stage2_num_trajectories"] == pytest.approx(2.0)
    assert metrics["stage2_valid_fraction"] == pytest.approx(0.5)
    assert metrics["stage2_mean_stage_reward"] == pytest.approx((3.0 + 0.75) / 2.0)
    assert metrics["stage2_mean_num_actions"] == pytest.approx(0.5)
    assert metrics["stage2_termination_fraction_stop_token"] == pytest.approx(0.5)
    assert metrics["stage2_termination_fraction_max_sequence_length"] == pytest.approx(0.5)
    assert metrics["stage3_num_trajectories"] == pytest.approx(1.0)
    assert metrics["stage3_valid_fraction"] == pytest.approx(1.0)
    assert metrics["stage3_mean_stage_reward"] == pytest.approx(1.25)
    assert metrics["stage3_termination_fraction_stop_token"] == pytest.approx(1.0)


def test_rollout_stage_metrics_return_zeroed_defaults_for_empty_batches() -> None:
    metrics = rollout_stage_metrics(
        (),
        max_molecules_per_sequence=8,
    )

    assert metrics["num_rollouts"] == pytest.approx(0.0)
    assert metrics["mean_realized_stage_count"] == pytest.approx(0.0)
    assert metrics["fraction_rollouts_reaching_stage_2"] == pytest.approx(0.0)
    assert metrics["fraction_rollouts_reaching_stage_3_plus"] == pytest.approx(0.0)
    assert metrics["stage1_num_trajectories"] == pytest.approx(0.0)
    assert metrics["stage1_termination_fraction_stop_token"] == pytest.approx(0.0)
    assert metrics["stage2_num_trajectories"] == pytest.approx(0.0)
    assert metrics["stage2_termination_fraction_max_stage_new_tokens"] == pytest.approx(0.0)
    assert metrics["stage3_num_trajectories"] == pytest.approx(0.0)
    assert metrics["stage3_termination_fraction_max_sequence_length"] == pytest.approx(0.0)


def test_build_trajectory_preview_payload_includes_num_stages_and_rendered_summary() -> None:
    trajectories = (
        _make_stage_trajectory(
            rollout_id="rollout-1",
            example_id="example-1",
            stage_index=1,
            reward=0.5,
            action_token_ids=(1,),
            target_selfies_list=("A",),
            sampled_selfies="A1",
            stage_text="A1",
        ),
        _make_stage_trajectory(
            rollout_id="rollout-2",
            example_id="example-2",
            stage_index=1,
            reward=1.5,
            action_token_ids=(2,),
            target_selfies_list=("A", "B"),
            sampled_selfies="B1",
            stage_text="B1",
        ),
        _make_stage_trajectory(
            rollout_id="rollout-2",
            example_id="example-2",
            stage_index=2,
            reward=2.0,
            action_token_ids=(3,),
            target_selfies_list=("A", "B"),
            sampled_selfies="B2",
            stage_text="B2",
        ),
        _make_stage_trajectory(
            rollout_id="rollout-3",
            example_id="example-3",
            stage_index=1,
            reward=0.75,
            action_token_ids=(4,),
            target_selfies_list=("A", "B", "C"),
            sampled_selfies="C1",
            stage_text="C1",
        ),
        _make_stage_trajectory(
            rollout_id="rollout-3",
            example_id="example-3",
            stage_index=2,
            reward=0.75,
            action_token_ids=(5,),
            target_selfies_list=("A", "B", "C"),
            sampled_selfies="C2",
            stage_text="C2",
        ),
        _make_stage_trajectory(
            rollout_id="rollout-3",
            example_id="example-3",
            stage_index=3,
            reward=0.75,
            action_token_ids=(6,),
            target_selfies_list=("A", "B", "C"),
            sampled_selfies="C3",
            stage_text="C3",
        ),
    )

    preview = build_trajectory_preview_payload(
        trajectories,
        iteration_index=5,
        num_samples=3,
        max_chars=160,
    )

    assert preview is not None
    assert [record["preview_slot"] for record in preview["records"]] == ["best", "median", "worst"]
    assert [record["rollout_id"] for record in preview["records"]] == [
        "rollout-2",
        "rollout-3",
        "rollout-1",
    ]
    assert [record["num_stages"] for record in preview["records"]] == [2, 3, 1]
    assert preview["records"][0]["stage_indices"] == [1, 2]
    assert "num_stages=2" in preview["tracker_text"]
    assert "stage_indices=[1, 2]" in preview["tracker_text"]
