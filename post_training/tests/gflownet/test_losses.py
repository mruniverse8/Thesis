import math

import pytest
import torch

from src.constants import EOM_TOKEN

from post_training.gflownet.losses import (
    detailed_balance_residuals,
    subtrajectory_balance_loss,
    subtrajectory_balance_residuals,
    trajectory_balance_residual,
)
from post_training.gflownet.trajectory import SampledStageTrajectory, ScoredStageTrajectory


def _make_sampled() -> SampledStageTrajectory:
    return SampledStageTrajectory(
        rollout_id="rollout-1",
        example_id="example-1",
        prompt_text="prompt",
        description="description",
        target_selfies_list=("[C][C][O]",),
        stage_index=1,
        decoder_prefix_text="",
        previous_valid_selfies=(),
        stage_text="<bom>[C][C][O]<eom>",
        sampled_selfies="[C][C][O]",
        action_token_ids=(10, 11),
        reward_breakdown={"amplified_reward": 2.0},
        prefix_rewards=(1.0e-4, 1.0e-4, 2.0),
        terminal_reward=2.0,
        stop_token=EOM_TOKEN,
        termination_reason="stop_token",
        is_valid=True,
        is_duplicate=False,
    )


def _make_scored(
    *,
    action_token_ids: tuple[int, ...],
    log_pf_tokens: tuple[float, ...],
    log_stop: tuple[float, ...],
    log_state_flows: tuple[float, ...],
    terminal_reward: float,
    log_pb_tokens: tuple[float, ...] = (),
) -> ScoredStageTrajectory:
    sampled = SampledStageTrajectory(
        rollout_id="rollout-1",
        example_id="example-1",
        prompt_text="prompt",
        description="description",
        target_selfies_list=("[C][C][O]",),
        stage_index=1,
        decoder_prefix_text="",
        previous_valid_selfies=(),
        stage_text="<bom>[C][C][O]<eom>",
        sampled_selfies="[C][C][O]",
        action_token_ids=action_token_ids,
        reward_breakdown={"amplified_reward": terminal_reward},
        prefix_rewards=tuple([1.0e-4] * len(action_token_ids) + [terminal_reward]),
        terminal_reward=terminal_reward,
        stop_token=EOM_TOKEN,
        termination_reason="stop_token",
        is_valid=True,
        is_duplicate=False,
    )
    return ScoredStageTrajectory(
        sampled=sampled,
        log_pf_tokens=log_pf_tokens,
        log_stop=log_stop,
        log_state_flows=log_state_flows,
        log_pb_tokens=log_pb_tokens,
    )


def test_trajectory_balance_residual_uses_root_flow_and_terminal_stop() -> None:
    sampled = _make_sampled()
    trajectory = ScoredStageTrajectory(
        sampled=sampled,
        log_pf_tokens=(torch.tensor(0.2), torch.tensor(0.3)),
        log_stop=(torch.tensor(-1.0), torch.tensor(-1.1), torch.tensor(-0.5)),
        log_state_flows=(torch.tensor(0.7), torch.tensor(0.8), torch.tensor(0.9)),
    )

    residual = trajectory_balance_residual(trajectory)
    expected = 0.7 + 0.2 + 0.3 - 0.5 - torch.log(torch.tensor(2.0))

    assert residual.item() == pytest.approx(float(expected.item()), abs=1.0e-6)


def test_detailed_balance_residuals_returns_one_transition_and_terminal_term_per_action() -> None:
    sampled = _make_sampled()
    trajectory = ScoredStageTrajectory(
        sampled=sampled,
        log_pf_tokens=(0.2, 0.3),
        log_stop=(-1.0, -1.1, -0.5),
        log_state_flows=(0.7, 0.8, 0.9),
    )

    residuals = detailed_balance_residuals(trajectory)

    assert residuals.shape == (3,)


def test_subtrajectory_balance_residuals_use_learned_state_flows_until_terminal_reward() -> None:
    sampled = _make_sampled()
    trajectory = ScoredStageTrajectory(
        sampled=sampled,
        log_pf_tokens=(0.2, 0.3),
        log_stop=(-1.0, -1.1, -0.5),
        log_state_flows=(0.7, 0.8, 0.9),
    )

    residuals, weights = subtrajectory_balance_residuals(trajectory)

    log_reward = torch.log(torch.tensor(2.0))
    expected = torch.tensor(
        [
            0.7 + 0.2 - 0.8,
            0.7 + 0.2 + 0.3 - 0.9,
            0.7 + 0.2 + 0.3 - 0.5 - float(log_reward),
            0.8 + 0.3 - 0.9,
            0.8 + 0.3 - 0.5 - float(log_reward),
            0.9 - 0.5 - float(log_reward),
        ],
        dtype=torch.float32,
    )

    assert residuals.shape == (6,)
    assert torch.allclose(residuals, expected, atol=1.0e-6)
    assert torch.allclose(weights, torch.ones_like(weights))


def test_subtrajectory_balance_residuals_match_hand_computation_with_backward_terms() -> None:
    trajectory = _make_scored(
        action_token_ids=(10, 11),
        log_pf_tokens=(0.2, 0.3),
        log_stop=(-1.0, -1.1, -0.4),
        log_state_flows=(0.7, 0.8, 1.1),
        log_pb_tokens=(-0.05, -0.15),
        terminal_reward=2.5,
    )

    residuals, weights = subtrajectory_balance_residuals(trajectory, lambda_decay=0.5)

    log_reward = math.log(2.5)
    expected_residuals = torch.tensor(
        [
            0.7 + 0.2 - (-0.05) - 0.8,
            0.7 + 0.2 + 0.3 - (-0.05) - (-0.15) - 1.1,
            0.7 + 0.2 + 0.3 - 0.4 - (-0.05) - (-0.15) - log_reward,
            0.8 + 0.3 - (-0.15) - 1.1,
            0.8 + 0.3 - 0.4 - (-0.15) - log_reward,
            1.1 - 0.4 - log_reward,
        ],
        dtype=torch.float32,
    )
    expected_weights = torch.tensor([1.0, 0.5, 0.25, 1.0, 0.5, 1.0], dtype=torch.float32)

    assert torch.allclose(residuals, expected_residuals, atol=1.0e-6)
    assert torch.allclose(weights, expected_weights, atol=1.0e-6)


def test_subtrajectory_balance_loss_matches_weighted_squared_residual_average() -> None:
    first = _make_scored(
        action_token_ids=(10, 11),
        log_pf_tokens=(0.2, 0.3),
        log_stop=(-1.0, -1.1, -0.4),
        log_state_flows=(0.7, 0.8, 1.1),
        log_pb_tokens=(-0.05, -0.15),
        terminal_reward=2.5,
    )
    second = _make_scored(
        action_token_ids=(12,),
        log_pf_tokens=(0.1,),
        log_stop=(-0.7, -0.2),
        log_state_flows=(0.4, 0.6),
        log_pb_tokens=(-0.05,),
        terminal_reward=1.5,
    )

    first_residuals, first_weights = subtrajectory_balance_residuals(first, lambda_decay=0.5)
    second_residuals, second_weights = subtrajectory_balance_residuals(second, lambda_decay=0.5)
    manual_loss = (
        (first_weights * first_residuals.pow(2)).sum()
        + (second_weights * second_residuals.pow(2)).sum()
    ) / (first_weights.sum() + second_weights.sum())

    loss = subtrajectory_balance_loss((first, second), lambda_decay=0.5)

    assert torch.allclose(loss, manual_loss, atol=1.0e-6)


def test_subtrajectory_terminal_sink_uses_reward_not_terminal_state_flow() -> None:
    trajectory = _make_scored(
        action_token_ids=(10, 11),
        log_pf_tokens=(0.2, 0.3),
        log_stop=(-1.0, -1.1, -0.5),
        log_state_flows=(0.7, 0.8, 5.0),
        terminal_reward=2.0,
    )

    residuals, _weights = subtrajectory_balance_residuals(trajectory)

    reward_based = 0.7 + 0.2 + 0.3 - 0.5 - math.log(2.0)
    wrong_flow_endpoint = 0.7 + 0.2 + 0.3 - 5.0

    assert residuals[2].item() == pytest.approx(reward_based, abs=1.0e-6)
    assert residuals[2].item() != pytest.approx(wrong_flow_endpoint, abs=1.0e-6)
