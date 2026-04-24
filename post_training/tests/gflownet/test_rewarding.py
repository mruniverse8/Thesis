import pytest

pytest.importorskip("selfies")
pytest.importorskip("rdkit")

from reward_utils.defaults import CHEBI20_REWARD_CONFIG, RewardConfig

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
    assert summary.reward_breakdown["total_reward"] == pytest.approx(1.8)
    assert summary.reward_breakdown["diversity_reward"] == pytest.approx(0.0)
    assert summary.terminal_reward == pytest.approx(summary.reward_breakdown["amplified_reward"])
    assert summary.terminal_reward == pytest.approx(14.4)
    assert summary.prefix_rewards == pytest.approx((1.0e-4, 1.0e-4, summary.terminal_reward))


def test_score_stage_terminal_reward_penalizes_invalid_stage_floor() -> None:
    summary = score_stage_terminal_reward(
        None,
        targets=["[C][C][O]"],
        previous_candidates=[],
        num_prefix_states=2,
        reward_config=CHEBI20_REWARD_CONFIG,
        invalid_terminal_reward=5.0e-4,
    )

    assert summary.is_valid_terminal is False
    assert summary.terminal_reward == pytest.approx(2.5e-4)
    assert summary.prefix_rewards == pytest.approx((5.0e-4, 2.5e-4))


def test_score_stage_terminal_reward_uses_configured_invalid_penalty() -> None:
    summary = score_stage_terminal_reward(
        None,
        targets=["[C][C][O]"],
        previous_candidates=[],
        num_prefix_states=2,
        reward_config=RewardConfig(penalty_invalid=0.25),
        invalid_terminal_reward=1.0e-4,
    )

    assert summary.is_valid_terminal is False
    assert summary.terminal_reward == pytest.approx(2.5e-5)
    assert summary.prefix_rewards == pytest.approx((1.0e-4, 2.5e-5))


def test_score_stage_terminal_reward_keeps_invalid_terminal_penalty_after_similarity_fallback() -> None:
    summary = score_stage_terminal_reward(
        "[C][C][Bad]",
        targets=["[C][C][O]"],
        previous_candidates=[],
        num_prefix_states=2,
        reward_config=RewardConfig(
            match_alpha=1.0,
            penalty_invalid=0.25,
            invalid_similarity_ngram_size=3,
        ),
        invalid_terminal_reward=1.0e-4,
    )

    assert summary.is_valid_terminal is False
    assert summary.reward_breakdown["match_reward"] == pytest.approx(0.125)
    assert summary.reward_breakdown["amplified_reward"] == pytest.approx(1.0)
    assert summary.terminal_reward == pytest.approx(0.25)
    assert summary.prefix_rewards == pytest.approx((1.0e-4, 0.25))


def test_score_stage_terminal_reward_uses_invalid_candidate_text_for_ngram_fallback() -> None:
    summary = score_stage_terminal_reward(
        None,
        invalid_candidate_text="[C][C][Bad]",
        targets=["[C][C][O]"],
        previous_candidates=[],
        num_prefix_states=2,
        reward_config=RewardConfig(
            match_alpha=1.0,
            penalty_invalid=0.25,
            invalid_similarity_ngram_size=3,
            reward_variant="reward_var2",
        ),
        invalid_terminal_reward=1.0e-4,
    )

    assert summary.is_valid_terminal is False
    assert summary.reward_breakdown["is_valid"] is False
    assert summary.reward_breakdown["match_reward"] == pytest.approx(0.125)
    assert summary.reward_breakdown["total_reward"] == pytest.approx(0.125)
    assert summary.reward_breakdown["amplified_reward"] == pytest.approx(1.0)
    assert summary.terminal_reward == pytest.approx(0.25)


def test_score_stage_terminal_reward_applies_reward_var2_duplicate_penalty_when_configured() -> None:
    summary = score_stage_terminal_reward(
        "[C][C][O]",
        targets=["[C][C][O]"],
        previous_candidates=["[C][C][O]"],
        num_prefix_states=2,
        reward_config=RewardConfig(
            reward_variant="reward_var2",
            duplicate_penalty_factor=0.1,
        ),
        invalid_terminal_reward=1.0e-4,
    )

    assert summary.is_valid_terminal is True
    assert summary.is_duplicate_terminal is True
    assert summary.reward_breakdown["total_reward"] == pytest.approx(0.18)
    assert summary.terminal_reward == pytest.approx(1.44)
    assert summary.prefix_rewards == pytest.approx((1.0e-4, 1.44))


def test_score_stage_terminal_reward_supports_reward_var1_ablation() -> None:
    summary = score_stage_terminal_reward(
        "[C][C][O]",
        targets=["[C][C][O]"],
        previous_candidates=["[C][C][O]"],
        num_prefix_states=2,
        reward_config=RewardConfig(
            diversity_beta=2.0,
            reward_variant="reward_var1",
        ),
        invalid_terminal_reward=1.0e-4,
    )

    assert summary.is_valid_terminal is True
    assert summary.is_duplicate_terminal is True
    assert summary.reward_breakdown["total_reward"] == pytest.approx(1.0)
    assert summary.terminal_reward == pytest.approx(8.0)


def test_score_stage_terminal_reward_supports_reward_var3_combination() -> None:
    summary = score_stage_terminal_reward(
        "[C][C][O]",
        targets=["[C][C][O]"],
        previous_candidates=["[C][C][C]"],
        num_prefix_states=2,
        reward_config=RewardConfig(
            diversity_beta=2.0,
            reward_variant="reward_var3",
        ),
        invalid_terminal_reward=1.0e-4,
    )

    assert summary.is_valid_terminal is True
    assert summary.is_duplicate_terminal is False
    assert summary.reward_breakdown["match_reward"] == pytest.approx(1.0)
    assert summary.reward_breakdown["diversity_reward"] > 0.0
    assert summary.reward_breakdown["total_reward"] == pytest.approx(
        summary.reward_breakdown["match_reward"]
        + summary.reward_breakdown["diversity_reward"]
        + 0.8
    )
    assert summary.terminal_reward == pytest.approx(8.0 * summary.reward_breakdown["total_reward"])
