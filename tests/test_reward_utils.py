from __future__ import annotations

import pytest

pytest.importorskip("rdkit")

from reward_utils import (
    CHEBI20_REWARD_CONFIG,
    RewardConfig,
    compute_dice_similarity,
    compute_rdiv,
    compute_rmatch,
    compute_tanimoto_similarity,
    compute_total_reward,
    is_duplicate_candidate,
    parse_molecule_text,
    score_candidate_sequence,
)


def test_reward_config_defaults_to_reward_var2_with_valid_bonus() -> None:
    config = RewardConfig()

    assert config.reward_variant == "reward_var2"
    assert config.plus_valid == pytest.approx(0.8)
    assert config.duplicate_penalty_factor == pytest.approx(1.0)
    assert config.penalty_invalid == pytest.approx(0.5)
    assert config.invalid_similarity_ngram_size == 3
    assert config.enable_invalid_similarity_ngram_fallback is True


def test_reward_config_accepts_reward_var3() -> None:
    config = RewardConfig(reward_variant="reward_var3")

    assert config.reward_variant == "reward_var3"


def test_reward_config_accepts_duplicate_penalty_factor() -> None:
    config = RewardConfig(duplicate_penalty_factor=0.1)

    assert config.duplicate_penalty_factor == pytest.approx(0.1)


def test_reward_config_accepts_penalty_invalid() -> None:
    config = RewardConfig(penalty_invalid=0.25)

    assert config.penalty_invalid == pytest.approx(0.25)


def test_reward_config_accepts_invalid_similarity_ngram_size() -> None:
    config = RewardConfig(invalid_similarity_ngram_size=2)

    assert config.invalid_similarity_ngram_size == 2


def test_reward_config_accepts_disabled_invalid_similarity_ngram_fallback() -> None:
    config = RewardConfig(enable_invalid_similarity_ngram_fallback=False)

    assert config.enable_invalid_similarity_ngram_fallback is False


@pytest.mark.parametrize("penalty_factor", (-0.1, 1.1))
def test_reward_config_rejects_out_of_range_duplicate_penalty_factor(penalty_factor: float) -> None:
    with pytest.raises(ValueError, match="duplicate_penalty_factor"):
        RewardConfig(duplicate_penalty_factor=penalty_factor)


@pytest.mark.parametrize("penalty_invalid", (-0.1, 0.0, 1.1))
def test_reward_config_rejects_out_of_range_penalty_invalid(penalty_invalid: float) -> None:
    with pytest.raises(ValueError, match="penalty_invalid"):
        RewardConfig(penalty_invalid=penalty_invalid)


@pytest.mark.parametrize("ngram_size", (-1, 0))
def test_reward_config_rejects_invalid_similarity_ngram_size(ngram_size: int) -> None:
    with pytest.raises(ValueError, match="invalid_similarity_ngram_size"):
        RewardConfig(invalid_similarity_ngram_size=ngram_size)


def test_parse_smiles_candidate_to_canonical_smiles() -> None:
    record = parse_molecule_text("CCO", representation="smiles")

    assert record.is_valid is True
    assert record.canonical_smiles == "CCO"
    assert record.used_selfies_decoder is False


def test_parse_selfies_candidate_to_canonical_smiles() -> None:
    record = parse_molecule_text("[C][C][O]", representation="selfies")

    assert record.is_valid is True
    assert record.canonical_smiles == "CCO"
    assert record.used_selfies_decoder is True


def test_invalid_candidate_returns_invalid_record() -> None:
    record = parse_molecule_text("this is not a molecule", representation="auto")

    assert record.is_valid is False
    assert record.canonical_smiles is None
    assert record.error is not None


def test_self_similarity_is_one_for_dice_and_tanimoto() -> None:
    assert compute_dice_similarity("CCO", "CCO") == pytest.approx(1.0)
    assert compute_tanimoto_similarity("CCO", "CCO") == pytest.approx(1.0)
    assert compute_dice_similarity("CCO", "CCO", invalid_fallback_penalty=0.1) == pytest.approx(1.0)


def test_similarity_returns_zero_when_fingerprint_and_fallback_features_do_not_overlap() -> None:
    assert compute_dice_similarity("not a molecule", "CCO") == pytest.approx(0.0)
    assert compute_tanimoto_similarity("not a molecule", "CCO") == pytest.approx(0.0)


def test_invalid_similarity_uses_penalized_character_ngram_fallback() -> None:
    dice = compute_dice_similarity("CCO?", "CCO", invalid_fallback_penalty=0.5)
    tanimoto = compute_tanimoto_similarity("CCO?", "CCO", invalid_fallback_penalty=0.5)

    assert dice == pytest.approx(1.0 / 3.0)
    assert tanimoto == pytest.approx(0.25)


def test_invalid_similarity_can_disable_ngram_fallback() -> None:
    dice = compute_dice_similarity(
        "CCO?",
        "CCO",
        invalid_fallback_penalty=0.5,
        enable_invalid_similarity_ngram_fallback=False,
    )
    tanimoto = compute_tanimoto_similarity(
        "CCO?",
        "CCO",
        invalid_fallback_penalty=0.5,
        enable_invalid_similarity_ngram_fallback=False,
    )

    assert dice == pytest.approx(0.0)
    assert tanimoto == pytest.approx(0.0)


def test_invalid_similarity_fallback_changes_with_penalty() -> None:
    low_penalty = compute_dice_similarity("CCO?", "CCO", invalid_fallback_penalty=0.25)
    high_penalty = compute_dice_similarity("CCO?", "CCO", invalid_fallback_penalty=0.5)

    assert low_penalty == pytest.approx(high_penalty / 2.0)


def test_invalid_character_fallback_changes_with_ngram_size() -> None:
    trigram = compute_dice_similarity(
        "CCO?",
        "CCO",
        invalid_fallback_penalty=0.5,
        invalid_similarity_ngram_size=3,
    )
    bigram = compute_dice_similarity(
        "CCO?",
        "CCO",
        invalid_fallback_penalty=0.5,
        invalid_similarity_ngram_size=2,
    )

    assert trigram == pytest.approx(1.0 / 3.0)
    assert bigram == pytest.approx(0.4)


def test_invalid_selfies_like_similarity_uses_bracket_token_fallback() -> None:
    similarity = compute_dice_similarity(
        "[C][C][Bad]",
        "[C][O][Bad]",
        representation_a="selfies",
        representation_b="selfies",
        invalid_fallback_penalty=0.5,
    )

    assert similarity == pytest.approx(0.4)


def test_rmatch_uses_maximum_dice_similarity_over_targets() -> None:
    component = compute_rmatch(
        "CCO",
        ["CCC", "CCO", "c1ccccc1"],
        alpha=0.5,
    )

    assert component.max_similarity == pytest.approx(1.0)
    assert component.reward == pytest.approx(1.0)
    assert component.best_index == 1
    assert component.best_reference == "CCO"


def test_rdiv_is_zero_for_first_candidate() -> None:
    component = compute_rdiv("CCO", [], beta=1.0)

    assert component.max_similarity == pytest.approx(0.0)
    assert component.reward == pytest.approx(0.0)
    assert component.best_index is None


def test_rdiv_goes_to_zero_for_duplicate_candidate() -> None:
    component = compute_rdiv("CCO", ["CCO"], beta=2.0)

    assert component.max_similarity == pytest.approx(1.0)
    assert component.reward == pytest.approx(0.0)


def test_reward_var1_preserves_legacy_duplicate_penalty() -> None:
    breakdown = compute_total_reward(
        "CCO",
        targets=["CCO", "CCC"],
        previous_candidates=["CCO"],
        config=RewardConfig(
            diversity_beta=2.0,
            reward_variant="reward_var1",
        ),
    )

    assert breakdown.is_duplicate is True
    assert breakdown.match.reward == pytest.approx(1.0)
    assert breakdown.diversity.reward == pytest.approx(0.0)
    assert breakdown.total_reward == pytest.approx(1.0)
    assert breakdown.amplified_reward == pytest.approx(8.0)


def test_reward_var2_flags_duplicates_without_penalizing_total_reward() -> None:
    breakdown = compute_total_reward(
        "CCO",
        targets=["CCO", "CCC"],
        previous_candidates=["CCO"],
        config=CHEBI20_REWARD_CONFIG,
    )

    assert breakdown.is_duplicate is True
    assert breakdown.match.reward == pytest.approx(1.0)
    assert breakdown.diversity.reward == pytest.approx(0.0)
    assert breakdown.total_reward == pytest.approx(1.8)
    assert breakdown.amplified_reward == pytest.approx(14.4)


def test_reward_var2_applies_duplicate_penalty_when_configured() -> None:
    breakdown = compute_total_reward(
        "CCO",
        targets=["CCO", "CCC"],
        previous_candidates=["CCO"],
        config=RewardConfig(
            reward_variant="reward_var2",
            duplicate_penalty_factor=0.1,
        ),
    )

    assert breakdown.candidate.is_valid is True
    assert breakdown.is_duplicate is True
    assert breakdown.match.reward == pytest.approx(1.0)
    assert breakdown.diversity.reward == pytest.approx(0.0)
    assert breakdown.total_reward == pytest.approx(0.18)
    assert breakdown.amplified_reward == pytest.approx(1.44)


def test_reward_var2_unfingerprintable_invalid_candidate_without_overlap_scores_zero() -> None:
    breakdown = compute_total_reward(
        "not a molecule",
        targets=["CCO"],
        previous_candidates=["CCC"],
    )

    assert breakdown.candidate.is_valid is False
    assert breakdown.match.reward == pytest.approx(0.0)
    assert breakdown.diversity.reward == pytest.approx(0.0)
    assert breakdown.total_reward == pytest.approx(0.0)


def test_total_reward_passes_invalid_similarity_fallback_config_to_rmatch() -> None:
    breakdown = compute_total_reward(
        "CCO?",
        targets=["CCO"],
        previous_candidates=[],
        config=RewardConfig(
            reward_variant="reward_var2",
            match_alpha=1.0,
            penalty_invalid=0.25,
            invalid_similarity_ngram_size=2,
        ),
    )

    assert breakdown.candidate.is_valid is False
    assert breakdown.match.max_similarity == pytest.approx(0.2)
    assert breakdown.match.reward == pytest.approx(0.2)
    assert breakdown.total_reward == pytest.approx(0.2)


def test_total_reward_can_disable_invalid_similarity_fallback_for_rmatch() -> None:
    breakdown = compute_total_reward(
        "CCO?",
        targets=["CCO"],
        previous_candidates=[],
        config=RewardConfig(
            reward_variant="reward_var2",
            match_alpha=1.0,
            penalty_invalid=0.25,
            invalid_similarity_ngram_size=2,
            enable_invalid_similarity_ngram_fallback=False,
        ),
    )

    assert breakdown.candidate.is_valid is False
    assert breakdown.match.max_similarity == pytest.approx(0.0)
    assert breakdown.match.reward == pytest.approx(0.0)
    assert breakdown.total_reward == pytest.approx(0.0)


def test_rmatch_passes_invalid_similarity_fallback_options() -> None:
    component = compute_rmatch(
        "CCO?",
        ["CCO"],
        alpha=1.0,
        invalid_fallback_penalty=0.25,
        invalid_similarity_ngram_size=2,
    )

    assert component.max_similarity == pytest.approx(0.2)
    assert component.reward == pytest.approx(0.2)


def test_rmatch_can_disable_invalid_similarity_fallback() -> None:
    component = compute_rmatch(
        "CCO?",
        ["CCO"],
        alpha=1.0,
        invalid_fallback_penalty=0.25,
        invalid_similarity_ngram_size=2,
        enable_invalid_similarity_ngram_fallback=False,
    )

    assert component.max_similarity == pytest.approx(0.0)
    assert component.reward == pytest.approx(0.0)


def test_reward_var1_ignores_duplicate_penalty_factor() -> None:
    breakdown = compute_total_reward(
        "CCO",
        targets=["CCO", "CCC"],
        previous_candidates=["CCO"],
        config=RewardConfig(
            diversity_beta=2.0,
            reward_variant="reward_var1",
            duplicate_penalty_factor=0.1,
        ),
    )

    assert breakdown.is_duplicate is True
    assert breakdown.total_reward == pytest.approx(1.0)
    assert breakdown.amplified_reward == pytest.approx(8.0)


def test_reward_var3_combines_match_diversity_and_valid_bonus() -> None:
    breakdown = compute_total_reward(
        "CCO",
        targets=["CCO"],
        previous_candidates=["CCC"],
        config=RewardConfig(
            diversity_beta=2.0,
            reward_variant="reward_var3",
        ),
    )

    assert breakdown.candidate.is_valid is True
    assert breakdown.match.reward == pytest.approx(1.0)
    assert breakdown.diversity.reward > 0.0
    assert breakdown.total_reward == pytest.approx(
        breakdown.match.reward + breakdown.diversity.reward + 0.8
    )
    assert breakdown.amplified_reward == pytest.approx(8.0 * breakdown.total_reward)


def test_reward_var3_ignores_duplicate_penalty_factor() -> None:
    breakdown = compute_total_reward(
        "CCO",
        targets=["CCO"],
        previous_candidates=["CCO"],
        config=RewardConfig(
            diversity_beta=2.0,
            reward_variant="reward_var3",
            duplicate_penalty_factor=0.1,
        ),
    )

    assert breakdown.candidate.is_valid is True
    assert breakdown.is_duplicate is True
    assert breakdown.match.reward == pytest.approx(1.0)
    assert breakdown.diversity.reward == pytest.approx(0.0)
    assert breakdown.total_reward == pytest.approx(1.8)
    assert breakdown.amplified_reward == pytest.approx(14.4)


def test_sequence_scoring_tracks_first_step_and_duplicates() -> None:
    results = score_candidate_sequence(
        candidates=["CCO", "CCO", "CCC"],
        targets=["CCO", "CCC"],
        config=CHEBI20_REWARD_CONFIG,
    )

    assert len(results) == 3
    assert results[0].diversity.reward == pytest.approx(0.0)
    assert results[0].total_reward == pytest.approx(1.8)
    assert results[1].is_duplicate is True
    assert results[1].diversity.reward == pytest.approx(0.0)
    assert results[1].total_reward == pytest.approx(1.8)
    assert results[2].candidate.is_valid is True
    assert results[2].total_reward == pytest.approx(1.8)


def test_auto_parser_accepts_spaced_selfies() -> None:
    record = parse_molecule_text("[C] [C] [O]", representation="auto")

    assert record.is_valid is True
    assert record.used_selfies_decoder is True
    assert record.canonical_smiles == "CCO"


def test_forcing_selfies_representation_on_plain_smiles_fails() -> None:
    record = parse_molecule_text("CCO", representation="selfies")

    assert record.is_valid is False
    assert record.canonical_smiles is None
    assert record.error is not None


def test_smiles_and_selfies_for_same_molecule_are_duplicates() -> None:
    smiles_record = parse_molecule_text("CCO", representation="smiles")
    selfies_record = parse_molecule_text("[C][C][O]", representation="selfies")

    assert smiles_record.canonical_smiles == selfies_record.canonical_smiles == "CCO"
    assert is_duplicate_candidate(selfies_record, [smiles_record]) is True


def test_similarity_is_one_across_smiles_and_selfies_for_same_molecule() -> None:
    dice = compute_dice_similarity(
        "[C][C][O]",
        "CCO",
        representation_a="selfies",
        representation_b="smiles",
    )
    tanimoto = compute_tanimoto_similarity(
        "[C][C][O]",
        "CCO",
        representation_a="selfies",
        representation_b="smiles",
    )

    assert dice == pytest.approx(1.0)
    assert tanimoto == pytest.approx(1.0)


def test_total_reward_supports_cross_representation_inputs() -> None:
    breakdown = compute_total_reward(
        "[C][C][O]",
        targets=["CCO", "CCC"],
        previous_candidates=["CCC"],
        config=CHEBI20_REWARD_CONFIG,
        candidate_representation="selfies",
        target_representation="smiles",
        previous_representation="smiles",
    )

    assert breakdown.candidate.is_valid is True
    assert breakdown.match.reward == pytest.approx(1.0)
    assert breakdown.diversity.reward == pytest.approx(0.0)
    assert breakdown.total_reward == pytest.approx(1.8)
    assert breakdown.is_duplicate is False


def test_rmatch_supports_cross_representation_inputs() -> None:
    component = compute_rmatch(
        "[C][C][O]",
        ["CCC", "CCO"],
        alpha=0.5,
        candidate_representation="selfies",
        target_representation="smiles",
    )

    assert component.max_similarity == pytest.approx(1.0)
    assert component.reward == pytest.approx(1.0)
    assert component.best_index == 1
    assert component.best_reference == "CCO"


def test_rdiv_supports_cross_representation_inputs() -> None:
    component = compute_rdiv(
        "[C][C][O]",
        ["CCO", "CCC"],
        beta=2.0,
        candidate_representation="selfies",
        previous_representation="smiles",
    )

    assert component.max_similarity == pytest.approx(1.0)
    assert component.reward == pytest.approx(0.0)
    assert component.best_index == 0
    assert component.best_reference == "CCO"


def test_sequence_scoring_handles_mixed_representations_and_invalid_inputs() -> None:
    results = score_candidate_sequence(
        candidates=["[C][C][O]", "CCO", "not a molecule"],
        targets=["CCO"],
        config=CHEBI20_REWARD_CONFIG,
        candidate_representation="auto",
        target_representation="smiles",
    )

    assert len(results) == 3
    assert results[0].candidate.is_valid is True
    assert results[1].is_duplicate is True
    assert results[2].candidate.is_valid is False
    assert results[2].total_reward == pytest.approx(0.0)


def test_curated_sequence_scoring_is_reproducible() -> None:
    candidates = ["[C][C][O]", "CCC", "c1ccccc1", "CCO"]
    targets = ["CCO", "CCC"]

    first_run = score_candidate_sequence(
        candidates=candidates,
        targets=targets,
        config=CHEBI20_REWARD_CONFIG,
        candidate_representation="auto",
        target_representation="smiles",
    )
    second_run = score_candidate_sequence(
        candidates=candidates,
        targets=targets,
        config=CHEBI20_REWARD_CONFIG,
        candidate_representation="auto",
        target_representation="smiles",
    )

    assert [item.candidate.canonical_smiles for item in first_run] == [
        item.candidate.canonical_smiles for item in second_run
    ]
    assert [item.total_reward for item in first_run] == pytest.approx(
        [item.total_reward for item in second_run]
    )
    assert [item.amplified_reward for item in first_run] == pytest.approx(
        [item.amplified_reward for item in second_run]
    )
    assert [item.is_duplicate for item in first_run] == [item.is_duplicate for item in second_run]
