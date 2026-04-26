from __future__ import annotations

import json

import pytest

from evaluation_metrics.cli import load_generation_groups_from_jsonl
from evaluation_metrics import (
    EvaluationMetricConfig,
    GenerationGroup,
    MoleculeInput,
    evaluate_generated_molecules,
    evaluate_generation_groups,
)


def test_accepted_unique_counts_only_valid_molecules_above_dice_threshold() -> None:
    result = evaluate_generated_molecules(
        candidates=("[C][C][O]", "[C][C][O]", "[C][C][C]", "[Bad]"),
        targets=("[C][C][O]",),
        config=EvaluationMetricConfig(acceptance_dice_threshold=0.7),
    )

    assert result.num_candidates == 4
    assert result.num_valid_candidates == 3
    assert result.num_unique_valid_molecules == 2
    assert result.num_accepted == 2
    assert result.accepted_unique_count == 1
    assert result.accepted_unique_smiles == ("CCO",)


def test_grouped_acceptance_uses_each_examples_own_targets() -> None:
    result = evaluate_generation_groups(
        (
            GenerationGroup(
                group_id="a",
                candidates=(MoleculeInput("[C][C][O]", "selfies"),),
                targets=(MoleculeInput("[C][C][O]", "selfies"),),
            ),
            GenerationGroup(
                group_id="b",
                candidates=(MoleculeInput("[C][C][C]", "selfies"),),
                targets=(MoleculeInput("[C][C][C]", "selfies"),),
            ),
        ),
        config=EvaluationMetricConfig(acceptance_dice_threshold=0.7),
    )

    accepted = [row for row in result.candidate_assessments if row.is_accepted]
    assert result.accepted_unique_count == 2
    assert {row.group_id for row in accepted} == {"a", "b"}


def test_novelty_counts_accepted_unique_molecules_not_in_targets() -> None:
    result = evaluate_generated_molecules(
        candidates=("[C][C][O]", "[C][C][N]"),
        targets=("[C][C][O]",),
        config=EvaluationMetricConfig(acceptance_dice_threshold=0.0),
    )

    assert result.accepted_unique_count == 2
    assert result.novelty_count == 1
    assert result.novelty_fraction == pytest.approx(0.5)
    assert result.novel_accepted_unique_smiles == ("CCN",)


def test_n_circles_is_disabled_by_default() -> None:
    result = evaluate_generated_molecules(
        candidates=("[C]", "[C][C]", "[C][C][C]", "[O]"),
        targets=("[C]", "[C][C]", "[C][C][C]", "[O]"),
        config=EvaluationMetricConfig(
            acceptance_dice_threshold=0.7,
            n_circles_tanimoto_threshold=0.6,
        ),
    )

    assert result.accepted_unique_count == 4
    assert result.n_circles == 0
    assert result.n_circles_exact is False
    assert result.to_dict()["compute_n_circles"] is False


def test_n_circles_counts_largest_mutually_dissimilar_accepted_subset_when_enabled() -> None:
    result = evaluate_generated_molecules(
        candidates=("[C]", "[C][C]", "[C][C][C]", "[O]"),
        targets=("[C]", "[C][C]", "[C][C][C]", "[O]"),
        config=EvaluationMetricConfig(
            acceptance_dice_threshold=0.7,
            compute_n_circles=True,
            n_circles_tanimoto_threshold=0.6,
        ),
    )

    assert result.accepted_unique_count == 4
    assert result.n_circles_exact is True
    assert result.n_circles >= 2
    assert result.n_circles <= result.accepted_unique_count
    assert result.to_dict()["compute_n_circles"] is True


def test_internal_diversity_is_average_pairwise_tanimoto_distance() -> None:
    same = evaluate_generated_molecules(
        candidates=("[C][C][O]", "[C][C][O]"),
        targets=("[C][C][O]",),
    )
    different = evaluate_generated_molecules(
        candidates=("[C][C][O]", "[O]"),
        targets=("[C][C][O]", "[O]"),
    )

    assert same.internal_diversity == pytest.approx(0.0)
    assert 0.0 <= different.internal_diversity <= 1.0
    assert different.internal_diversity > same.internal_diversity


def test_result_to_dict_can_include_assessments() -> None:
    result = evaluate_generated_molecules(
        candidates=("[C][O]",),
        targets=("[C][O]",),
    )

    payload = result.to_dict(include_assessments=True)
    assert payload["accepted_unique_count"] == 1
    assert payload["novelty_count"] == 0
    assert payload["novelty_fraction"] == 0.0
    assert payload["novel_accepted_unique_smiles"] == []
    assert payload["candidate_assessments"][0]["is_accepted"] is True


def test_load_generation_groups_from_jsonl_uses_target_lookup(tmp_path) -> None:
    generated_path = tmp_path / "generated.jsonl"
    targets_path = tmp_path / "targets.jsonl"
    generated_path.write_text(
        json.dumps(
            {
                "example_id": "ex-1",
                "generated_selfies_list": ["[C][C][O]"],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    targets_path.write_text(
        json.dumps(
            {
                "id": "ex-1",
                "target_selfies_list": ["[C][C][O]"],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    groups = load_generation_groups_from_jsonl(
        generated_path,
        targets_jsonl=targets_path,
    )

    assert len(groups) == 1
    assert groups[0].group_id == "ex-1"
    assert groups[0].candidates[0].text == "[C][C][O]"
    assert groups[0].targets[0].text == "[C][C][O]"


def test_load_generation_groups_from_jsonl_treats_canonical_smiles_as_smiles(tmp_path) -> None:
    generated_path = tmp_path / "generated.jsonl"
    generated_path.write_text(
        json.dumps(
            {
                "id": "ex-1",
                "canonical_smiles": "CCO",
                "target_selfies_list": ["[C][C][O]"],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    groups = load_generation_groups_from_jsonl(generated_path)
    result = evaluate_generation_groups(groups)

    assert groups[0].candidates[0].representation == "smiles"
    assert result.accepted_unique_count == 1
