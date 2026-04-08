from __future__ import annotations

import pytest

pytest.importorskip("rdkit")

from reward_utils import (
    CHEBI20_REWARD_CONFIG,
    compute_dice_similarity,
    compute_rdiv,
    compute_rmatch,
    compute_tanimoto_similarity,
    compute_total_reward,
    is_duplicate_candidate,
    parse_molecule_text,
    score_candidate_sequence,
)


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


def test_total_reward_flags_duplicates_and_applies_amplification() -> None:
    breakdown = compute_total_reward(
        "CCO",
        targets=["CCO", "CCC"],
        previous_candidates=["CCO"],
        config=CHEBI20_REWARD_CONFIG,
    )

    assert breakdown.is_duplicate is True
    assert breakdown.match.reward == pytest.approx(1.0)
    assert breakdown.diversity.reward == pytest.approx(0.0)
    assert breakdown.total_reward == pytest.approx(1.0)
    assert breakdown.amplified_reward == pytest.approx(8.0)


def test_invalid_candidate_yields_zero_reward() -> None:
    breakdown = compute_total_reward(
        "not a molecule",
        targets=["CCO"],
        previous_candidates=["CCC"],
    )

    assert breakdown.candidate.is_valid is False
    assert breakdown.match.reward == pytest.approx(0.0)
    assert breakdown.diversity.reward == pytest.approx(0.0)
    assert breakdown.total_reward == pytest.approx(0.0)


def test_sequence_scoring_tracks_first_step_and_duplicates() -> None:
    results = score_candidate_sequence(
        candidates=["CCO", "CCO", "CCC"],
        targets=["CCO", "CCC"],
        config=CHEBI20_REWARD_CONFIG,
    )

    assert len(results) == 3
    assert results[0].diversity.reward == pytest.approx(0.0)
    assert results[1].is_duplicate is True
    assert results[1].diversity.reward == pytest.approx(0.0)
    assert results[2].candidate.is_valid is True


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
    assert breakdown.diversity.reward < 1.0
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
