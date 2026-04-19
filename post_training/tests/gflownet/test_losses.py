import pytest
import torch

from src.constants import EOM_TOKEN

from post_training.gflownet.losses import (
    detailed_balance_residuals,
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
