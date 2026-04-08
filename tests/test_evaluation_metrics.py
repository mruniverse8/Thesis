from __future__ import annotations

import json

import pytest

pytest.importorskip("rdkit")

from evaluation import (
    MoleculeMetricConfig,
    ReferenceMolecule,
    compute_group_molecule_metrics,
    compute_internal_diversity,
    compute_ncircles,
)
from reward_utils import compute_tanimoto_similarity
from src.evaluation import evaluate_prediction_molecules
from src.io_utils import write_jsonl


def test_ncircles_and_internal_diversity_use_accepted_unique_molecules() -> None:
    molecules = ["CCO", "CCCCO", "CCCCCO"]

    expected_intdiv = (
        (1.0 - compute_tanimoto_similarity("CCO", "CCCCO"))
        + (1.0 - compute_tanimoto_similarity("CCO", "CCCCCO"))
        + (1.0 - compute_tanimoto_similarity("CCCCO", "CCCCCO"))
    ) / 3.0

    assert compute_ncircles(molecules, tanimoto_threshold=0.6) == 2
    assert compute_internal_diversity(molecules) == pytest.approx(expected_intdiv)


def test_group_metrics_collapse_duplicate_smiles_and_selfies() -> None:
    references = [
        ReferenceMolecule(
            example_id="ref-1",
            description="desc-a",
            molecule_text="CCO",
            representation="smiles",
        )
    ]
    predictions = [
        {
            "id": "pred-1",
            "description": "desc-a",
            "prediction_smiles": "CCO",
            "prediction_selfies": "",
            "prediction_text": "CCO",
        },
        {
            "id": "pred-2",
            "description": "desc-a",
            "prediction_smiles": None,
            "prediction_selfies": "[C][C][O]",
            "prediction_text": "[C][C][O]",
        },
    ]

    summary, assessments = compute_group_molecule_metrics(predictions, references)

    assert len(assessments) == 2
    assert summary["num_predictions"] == 2
    assert summary["num_valid"] == 2
    assert summary["num_accepted"] == 2
    assert summary["num_unique_accepted"] == 1
    assert summary["accepted_unique_smiles"] == ["CCO"]
    assert summary["ncircles"] == 1
    assert summary["intdiv"] == pytest.approx(0.0)


def test_evaluate_prediction_molecules_writes_split_and_group_reports(tmp_path) -> None:
    dataset_path = tmp_path / "validation.jsonl"
    summary_path = tmp_path / "validation_molecule_metrics.json"
    group_path = tmp_path / "validation_molecule_metrics_by_group.jsonl"

    write_jsonl(
        dataset_path,
        [
            {
                "id": "ref-a1",
                "description": "desc-a",
                "selfies": "[C][C][O]",
                "source_smiles": "CCO",
            },
            {
                "id": "ref-a2",
                "description": "desc-a",
                "selfies": "[C][C][C][C][O]",
                "source_smiles": "CCCCO",
            },
            {
                "id": "ref-b1",
                "description": "desc-b",
                "selfies": "[N][C][=C][C][=C][Branch1][C][N][C][=C][Ring1][#Branch1]",
                "source_smiles": "Nc1ccc(N)cc1",
            },
        ],
    )

    predictions = [
        {
            "id": "pred-a1",
            "description": "desc-a",
            "prediction_smiles": "CCO",
            "prediction_selfies": "",
            "prediction_text": "CCO",
        },
        {
            "id": "pred-a2",
            "description": "desc-a",
            "prediction_smiles": "CCCCO",
            "prediction_selfies": "",
            "prediction_text": "CCCCO",
        },
        {
            "id": "pred-a3",
            "description": "desc-a",
            "prediction_smiles": "CCCCCO",
            "prediction_selfies": "",
            "prediction_text": "CCCCCO",
        },
        {
            "id": "pred-a4",
            "description": "desc-a",
            "prediction_smiles": None,
            "prediction_selfies": "[C][C][O]",
            "prediction_text": "[C][C][O]",
        },
        {
            "id": "pred-a5",
            "description": "desc-a",
            "prediction_smiles": None,
            "prediction_selfies": "not a molecule",
            "prediction_text": "not a molecule",
        },
        {
            "id": "pred-b1",
            "description": "desc-b",
            "prediction_smiles": "CCN",
            "prediction_selfies": "",
            "prediction_text": "CCN",
        },
    ]

    summary, by_group, written_summary_path, written_group_path = evaluate_prediction_molecules(
        predictions,
        dataset_path=dataset_path,
        metric_config=MoleculeMetricConfig(),
        summary_path=summary_path,
        by_group_path=group_path,
    )

    expected_intdiv = (
        (1.0 - compute_tanimoto_similarity("CCO", "CCCCO"))
        + (1.0 - compute_tanimoto_similarity("CCO", "CCCCCO"))
        + (1.0 - compute_tanimoto_similarity("CCCCO", "CCCCCO"))
    ) / 3.0

    assert written_summary_path == summary_path
    assert written_group_path == group_path
    assert summary["num_generated"] == 6
    assert summary["num_groups"] == 2
    assert summary["num_valid"] == 5
    assert summary["num_accepted"] == 4
    assert summary["num_unique_accepted"] == 3
    assert summary["accepted_unique_smiles"] == ["CCO", "CCCCO", "CCCCCO"]
    assert summary["ncircles"] == 2
    assert summary["intdiv"] == pytest.approx(expected_intdiv)

    desc_a = next(item for item in by_group if item["description"] == "desc-a")
    desc_b = next(item for item in by_group if item["description"] == "desc-b")

    assert desc_a["num_predictions"] == 5
    assert desc_a["num_accepted"] == 4
    assert desc_a["num_unique_accepted"] == 3
    assert desc_a["accepted_unique_smiles"] == ["CCO", "CCCCO", "CCCCCO"]
    assert desc_a["ncircles"] == 2
    assert desc_b["num_predictions"] == 1
    assert desc_b["num_accepted"] == 0
    assert desc_b["num_unique_accepted"] == 0

    with summary_path.open("r", encoding="utf-8") as handle:
        written_summary = json.load(handle)
    with group_path.open("r", encoding="utf-8") as handle:
        written_groups = [json.loads(line) for line in handle if line.strip()]

    assert written_summary["num_unique_accepted"] == 3
    assert len(written_groups) == 2
