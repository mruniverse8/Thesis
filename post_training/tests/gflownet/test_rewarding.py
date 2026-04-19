import pytest

pytest.importorskip("selfies")
pytest.importorskip("rdkit")

from reward_utils.defaults import CHEBI20_REWARD_CONFIG

from post_training.gflownet.rewarding import score_stage_terminal_reward


def test_score_stage_terminal_reward_uses_current_stage_reward_for_valid_stage() -> None:
    summary = score_stage_terminal_reward(
        "[C][C][O]",
        targets=["[C][C][O]"],
        previous_candidates=[],
        num_prefix_states=3,
        reward_config=CHEBI20_REWARD_CONFIG,
        invalid_terminal_reward=1.0e-4,
    )

    assert summary.is_valid_terminal is True
    assert summary.terminal_reward == pytest.approx(summary.reward_breakdown["amplified_reward"])
    assert summary.prefix_rewards == pytest.approx((1.0e-4, 1.0e-4, summary.terminal_reward))


def test_score_stage_terminal_reward_floors_invalid_stage() -> None:
    summary = score_stage_terminal_reward(
        None,
        targets=["[C][C][O]"],
        previous_candidates=[],
        num_prefix_states=2,
        reward_config=CHEBI20_REWARD_CONFIG,
        invalid_terminal_reward=5.0e-4,
    )

    assert summary.is_valid_terminal is False
    assert summary.terminal_reward == pytest.approx(5.0e-4)
    assert summary.prefix_rewards == pytest.approx((5.0e-4, 5.0e-4))
