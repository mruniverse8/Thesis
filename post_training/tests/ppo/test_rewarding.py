import pytest

pytest.importorskip("selfies")
pytest.importorskip("rdkit")

from reward_utils.defaults import CHEBI20_DIVERSITY_BETA, CHEBI20_REWARD_CONFIG, RewardConfig

from post_training.ppo.rewarding import build_reward_config, score_generated_sequence, score_stage_candidate


def test_build_reward_config_uses_chebi_default_when_not_overridden() -> None:
    config = build_reward_config({}, dataset_hint="data/post_training/processed/train_multimol.jsonl")

    assert config.diversity_beta == CHEBI20_DIVERSITY_BETA
    assert config.reward_variant == "reward_var2"
    assert config.plus_valid == pytest.approx(0.8)
    assert config.invalid_similarity_ngram_size == 3
    assert config.enable_invalid_similarity_ngram_fallback is True


def test_build_reward_config_accepts_invalid_similarity_ngram_size() -> None:
    config = build_reward_config({"invalid_similarity_ngram_size": 2})

    assert config.invalid_similarity_ngram_size == 2


def test_build_reward_config_accepts_disabled_invalid_similarity_ngram_fallback() -> None:
    config = build_reward_config({"enable_invalid_similarity_ngram_fallback": False})

    assert config.enable_invalid_similarity_ngram_fallback is False


def test_score_stage_candidate_handles_first_stage() -> None:
    breakdown = score_stage_candidate(
        "[C][C][O]",
        targets=["[C][C][O]", "[C][C][C]"],
        previous_candidates=[],
        config=CHEBI20_REWARD_CONFIG,
    )

    assert breakdown.match.reward == pytest.approx(1.0)
    assert breakdown.diversity.reward == pytest.approx(0.0)
    assert breakdown.total_reward == pytest.approx(1.8)


def test_score_generated_sequence_flags_duplicate_later_stage() -> None:
    results = score_generated_sequence(
        ["[C][C][O]", "[C][C][O]"],
        targets=["[C][C][O]"],
        config=CHEBI20_REWARD_CONFIG,
    )

    assert results[1].is_duplicate is True
    assert results[1].diversity.reward == pytest.approx(0.0)
    assert results[1].total_reward == pytest.approx(1.8)


def test_score_stage_candidate_applies_reward_var2_duplicate_penalty_when_configured() -> None:
    breakdown = score_stage_candidate(
        "[C][C][O]",
        targets=["[C][C][O]"],
        previous_candidates=["[C][C][O]"],
        config=RewardConfig(
            reward_variant="reward_var2",
            duplicate_penalty_factor=0.1,
        ),
    )

    assert breakdown.is_duplicate is True
    assert breakdown.candidate.is_valid is True
    assert breakdown.total_reward == pytest.approx(0.18)
    assert breakdown.amplified_reward == pytest.approx(1.44)


def test_score_stage_candidate_supports_reward_var1_ablation() -> None:
    breakdown = score_stage_candidate(
        "[C][C][O]",
        targets=["[C][C][O]"],
        previous_candidates=["[C][C][O]"],
        config=RewardConfig(
            diversity_beta=CHEBI20_DIVERSITY_BETA,
            reward_variant="reward_var1",
        ),
    )

    assert breakdown.is_duplicate is True
    assert breakdown.total_reward == pytest.approx(1.0)
    assert breakdown.amplified_reward == pytest.approx(8.0)


def test_score_stage_candidate_supports_reward_var3_combination() -> None:
    breakdown = score_stage_candidate(
        "[C][C][O]",
        targets=["[C][C][O]"],
        previous_candidates=["[C][C][C]"],
        config=RewardConfig(
            diversity_beta=CHEBI20_DIVERSITY_BETA,
            reward_variant="reward_var3",
        ),
    )

    assert breakdown.is_duplicate is False
    assert breakdown.match.reward == pytest.approx(1.0)
    assert breakdown.diversity.reward > 0.0
    assert breakdown.total_reward == pytest.approx(
        breakdown.match.reward + breakdown.diversity.reward + 0.8
    )
