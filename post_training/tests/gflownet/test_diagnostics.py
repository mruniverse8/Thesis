import pytest

from src.constants import EOM_TOKEN

from post_training.gflownet.diagnostics import rollout_stage_metrics, termination_reason_metrics
from post_training.gflownet.trajectory import SampledStageTrajectory


def _make_sampled_trajectory(
    *,
    rollout_id: str,
    stage_index: int,
    terminal_reward: float,
    termination_reason: str = "stop_token",
    is_valid: bool = True,
    action_token_ids: tuple[int, ...] = (1, 2),
    target_selfies_list: tuple[str, ...] = ("[C][C][O]",),
) -> SampledStageTrajectory:
    return SampledStageTrajectory(
        rollout_id=rollout_id,
        example_id=f"example-{rollout_id}",
        prompt_text="prompt",
        description="description",
        target_selfies_list=target_selfies_list,
        stage_index=stage_index,
        decoder_prefix_text="",
        previous_sampled_selfies=(),
        stage_text="<bom>[C][C][O]<eom>",
        sampled_selfies="[C][C][O]" if is_valid else None,
        action_token_ids=action_token_ids,
        reward_breakdown={"amplified_reward": terminal_reward},
        prefix_rewards=tuple([1.0e-4] * len(action_token_ids) + [terminal_reward]),
        terminal_reward=terminal_reward,
        stop_token=EOM_TOKEN,
        termination_reason=termination_reason,
        is_valid=is_valid,
        is_duplicate=False,
    )


def test_termination_reason_metrics_supports_custom_prefix() -> None:
    trajectories = (
        _make_sampled_trajectory(
            rollout_id="rollout-1",
            stage_index=1,
            terminal_reward=2.0,
            termination_reason="stop_token",
        ),
        _make_sampled_trajectory(
            rollout_id="rollout-2",
            stage_index=1,
            terminal_reward=1.0e-4,
            termination_reason="max_stage_new_tokens",
            is_valid=False,
        ),
    )

    metrics = termination_reason_metrics(trajectories, prefix="stage1_termination_fraction_")

    assert metrics["stage1_termination_fraction_stop_token"] == pytest.approx(0.5)
    assert metrics["stage1_termination_fraction_max_stage_new_tokens"] == pytest.approx(0.5)
    assert metrics["stage1_termination_fraction_eos_token"] == pytest.approx(0.0)


def test_rollout_stage_metrics_capture_realized_lengths_and_stage_split_breakdown() -> None:
    trajectories = (
        _make_sampled_trajectory(
            rollout_id="rollout-1",
            stage_index=1,
            terminal_reward=2.0,
            target_selfies_list=("[C][C][O]", "[C][C][N]"),
        ),
        _make_sampled_trajectory(
            rollout_id="rollout-1",
            stage_index=2,
            terminal_reward=3.0,
            target_selfies_list=("[C][C][O]", "[C][C][N]"),
        ),
        _make_sampled_trajectory(
            rollout_id="rollout-2",
            stage_index=1,
            terminal_reward=1.0e-4,
            termination_reason="max_stage_new_tokens",
            is_valid=False,
            action_token_ids=(1, 2, 3, 4),
            target_selfies_list=("[C][C][O]", "[C][C][N]", "[C][O][O]"),
        ),
        _make_sampled_trajectory(
            rollout_id="rollout-3",
            stage_index=1,
            terminal_reward=4.0,
            target_selfies_list=("[C][C][O]",),
        ),
    )

    metrics = rollout_stage_metrics(
        trajectories,
        max_molecules_per_sequence=8,
        invalid_terminal_reward=1.0e-4,
    )

    assert metrics["num_rollouts"] == pytest.approx(3.0)
    assert metrics["max_stage_index"] == pytest.approx(2.0)
    assert metrics["mean_planned_stage_count"] == pytest.approx(8.0)
    assert metrics["max_planned_stage_count"] == pytest.approx(8.0)
    assert metrics["mean_realized_stage_count"] == pytest.approx(4.0 / 3.0)
    assert metrics["max_realized_stage_count"] == pytest.approx(2.0)
    assert metrics["fraction_rollouts_planned_stage_2_plus"] == pytest.approx(1.0)
    assert metrics["fraction_rollouts_reaching_stage_2"] == pytest.approx(1.0 / 3.0)
    assert metrics["fraction_rollouts_reaching_stage_3_plus"] == pytest.approx(0.0)
    assert metrics["fraction_rollouts_reaching_planned_stage_count"] == pytest.approx(0.0)
    assert metrics["rollout_stage_count_1_fraction"] == pytest.approx(2.0 / 3.0)
    assert metrics["rollout_stage_count_2_fraction"] == pytest.approx(1.0 / 3.0)
    assert metrics["rollout_stage_count_3_plus_fraction"] == pytest.approx(0.0)
    assert metrics["invalid_reward_floor_fraction"] == pytest.approx(0.25)
    assert metrics["stage1_num_trajectories"] == pytest.approx(3.0)
    assert metrics["stage1_valid_fraction"] == pytest.approx(2.0 / 3.0)
    assert metrics["stage1_mean_stage_reward"] == pytest.approx((2.0 + 1.0e-4 + 4.0) / 3.0)
    assert metrics["stage1_mean_num_actions"] == pytest.approx(8.0 / 3.0)
    assert metrics["stage1_invalid_reward_floor_fraction"] == pytest.approx(1.0 / 3.0)
    assert metrics["stage1_termination_fraction_stop_token"] == pytest.approx(2.0 / 3.0)
    assert metrics["stage1_termination_fraction_max_stage_new_tokens"] == pytest.approx(1.0 / 3.0)
    assert metrics["stage2_num_trajectories"] == pytest.approx(1.0)
    assert metrics["stage2_valid_fraction"] == pytest.approx(1.0)
    assert metrics["stage2_mean_stage_reward"] == pytest.approx(3.0)
    assert metrics["stage2_mean_num_actions"] == pytest.approx(2.0)
    assert metrics["stage2_invalid_reward_floor_fraction"] == pytest.approx(0.0)
    assert metrics["stage2_termination_fraction_stop_token"] == pytest.approx(1.0)


def test_rollout_stage_metrics_return_zeroed_defaults_for_empty_batches() -> None:
    metrics = rollout_stage_metrics(
        (),
        max_molecules_per_sequence=8,
        invalid_terminal_reward=1.0e-4,
    )

    assert metrics["num_rollouts"] == pytest.approx(0.0)
    assert metrics["mean_realized_stage_count"] == pytest.approx(0.0)
    assert metrics["fraction_rollouts_reaching_stage_2"] == pytest.approx(0.0)
    assert metrics["invalid_reward_floor_fraction"] == pytest.approx(0.0)
    assert metrics["stage1_num_trajectories"] == pytest.approx(0.0)
    assert metrics["stage1_termination_fraction_stop_token"] == pytest.approx(0.0)
    assert metrics["stage2_num_trajectories"] == pytest.approx(0.0)
    assert metrics["stage2_termination_fraction_max_stage_new_tokens"] == pytest.approx(0.0)
