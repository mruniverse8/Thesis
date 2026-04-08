import pytest

pytest.importorskip("selfies")
pytest.importorskip("rdkit")

from reward_utils.defaults import CHEBI20_REWARD_CONFIG

from post_training.reward_adapter import score_generated_sequence, score_stage_candidate


def test_score_stage_candidate_handles_first_stage() -> None:
    breakdown = score_stage_candidate(
        "[C][C][O]",
        targets=["[C][C][O]", "[C][C][C]"],
        previous_candidates=[],
        config=CHEBI20_REWARD_CONFIG,
    )

    assert breakdown.match.reward == pytest.approx(1.0)
    assert breakdown.diversity.reward == pytest.approx(0.0)


def test_score_generated_sequence_flags_duplicate_later_stage() -> None:
    results = score_generated_sequence(
        ["[C][C][O]", "[C][C][O]"],
        targets=["[C][C][O]"],
        config=CHEBI20_REWARD_CONFIG,
    )

    assert results[1].is_duplicate is True
    assert results[1].diversity.reward == pytest.approx(0.0)
