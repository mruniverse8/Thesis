from __future__ import annotations

import pytest

pytest.importorskip("selfies")

from data_collection.lpm24 import build_lpm24_grouped_records


def test_build_lpm24_grouped_records_groups_descriptions_and_deduplicates() -> None:
    raw_records = [
        {"description": "dual inhibitor", "SMILES": "CCO"},
        {"description": "dual inhibitor", "SMILES": "CO"},
        {"description": "dual inhibitor", "SMILES": "CCO"},
        {"description": "small hydrocarbon", "smiles": "CC"},
    ]

    processed = build_lpm24_grouped_records(raw_records, split="train", max_molecules_per_example=8)

    assert processed == [
        {
            "id": "train-000000",
            "description": "dual inhibitor",
            "target_selfies_list": ["[C][C][O]", "[C][O]"],
            "target_smiles_list": ["CCO", "CO"],
        },
        {
            "id": "train-000001",
            "description": "small hydrocarbon",
            "target_selfies_list": ["[C][C]"],
            "target_smiles_list": ["CC"],
        },
    ]


def test_build_lpm24_grouped_records_skips_groups_without_valid_molecules() -> None:
    raw_records = [
        {"caption": "valid group", "molecule": "CCO"},
        {"caption": "invalid-only group", "molecule": "not-a-smiles"},
        {"caption": "missing-smiles"},
    ]

    processed = build_lpm24_grouped_records(raw_records, split="test", max_molecules_per_example=8)

    assert processed == [
        {
            "id": "test-000000",
            "description": "valid group",
            "target_selfies_list": ["[C][C][O]"],
            "target_smiles_list": ["CCO"],
        }
    ]
