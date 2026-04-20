from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("selfies")

from data_collection.lpm24 import export_lpm24_training_splits
from post_training.sft_multi.dataset import MultiMoleculeDataset
from src.io_utils import load_yaml, read_jsonl, write_jsonl
from src.runtime_bootstrap import infer_managed_dataset, resolve_stage_spec


def _write_processed_split(path: Path, records: list[dict[str, object]]) -> None:
    write_jsonl(path, records)


def test_export_lpm24_training_splits_creates_grouped_layout_and_filters_outliers(tmp_path: Path) -> None:
    dataset_root = tmp_path / "data" / "lpm24"
    processed_dir = dataset_root / "processed"

    train_records = [
        {
            "id": "train-000000",
            "description": "kept train one",
            "target_selfies_list": ["[C][O]"],
            "target_smiles_list": ["CO"],
        },
        {
            "id": "train-000001",
            "description": "kept train two",
            "target_selfies_list": ["[C]"],
            "target_smiles_list": ["C"],
        },
        {
            "id": "train-000002",
            "description": "kept train three",
            "target_selfies_list": ["[C]", "[O]"],
            "target_smiles_list": ["C", "O"],
        },
        {
            "id": "train-000003",
            "description": "stage outlier",
            "target_selfies_list": ["[C][C][C][C]"],
            "target_smiles_list": ["CCCC"],
        },
        {
            "id": "train-000004",
            "description": "target outlier",
            "target_selfies_list": ["[C][C]", "[O][O]", "[N][N]"],
            "target_smiles_list": ["CC", "OO", "NN"],
        },
    ]
    test_records = [
        {
            "id": "test-000000",
            "description": "kept test one",
            "target_selfies_list": ["[C][N]"],
            "target_smiles_list": ["CN"],
        },
        {
            "id": "test-000001",
            "description": "filtered test one",
            "target_selfies_list": ["[C][C][C][C]"],
            "target_smiles_list": ["CCCC"],
        },
    ]

    _write_processed_split(processed_dir / "train_multimol.jsonl", train_records)
    _write_processed_split(processed_dir / "test_multimol.jsonl", test_records)

    summary = export_lpm24_training_splits(
        input_dir=dataset_root,
        validation_fraction=0.34,
        seed=7,
        max_target_symbols=5,
        max_stage_symbols=3,
    )

    grouped_splits_dir = dataset_root / "grouped_splits"
    train_output = read_jsonl(grouped_splits_dir / "train_multimol.jsonl")
    validation_output = read_jsonl(grouped_splits_dir / "validation_multimol.jsonl")
    test_output = read_jsonl(grouped_splits_dir / "test_multimol.jsonl")

    train_output_ids = {record["id"] for record in train_output}
    validation_output_ids = {record["id"] for record in validation_output}
    test_output_ids = {record["id"] for record in test_output}

    assert summary["counts"] == {
        "train_source_records": 5,
        "train_records": 2,
        "validation_records": 1,
        "test_source_records": 2,
        "test_records": 1,
    }
    assert train_output_ids | validation_output_ids == {
        "train-000000",
        "train-000001",
        "train-000002",
    }
    assert train_output_ids.isdisjoint(validation_output_ids)
    assert test_output_ids == {"test-000000"}
    assert summary["filtering"]["train_source"]["removed_by_reason"] == {
        "stage_symbol_limit": 1,
        "target_symbol_limit": 1,
    }
    assert summary["filtering"]["test"]["removed_by_reason"] == {"stage_symbol_limit": 1}
    assert train_output[0]["target_smiles_list"] or validation_output[0]["target_smiles_list"]

    dataset = MultiMoleculeDataset.from_jsonl(grouped_splits_dir / "train_multimol.jsonl")
    assert len(dataset) == 2
    assert sorted(dataset[0].keys()) == [
        "description",
        "id",
        "prompt",
        "target_selfies_list",
        "target_smiles_list",
        "target_text",
    ]


def test_lpm24_configs_target_grouped_splits_and_manual_bootstrap_mode(tmp_path: Path) -> None:
    sft_config = load_yaml("configs/multi_molecule_sft_lpm24.yaml")
    gflownet_config = load_yaml("configs/multi_molecule_gflownet_lpm24.yaml")

    assert sft_config["model"]["molecule_separator_token"] == " "
    assert sft_config["data"] == {
        "train_file": "data/lpm24/grouped_splits/train_multimol.jsonl",
        "validation_file": "data/lpm24/grouped_splits/validation_multimol.jsonl",
        "test_file": "data/lpm24/grouped_splits/test_multimol.jsonl",
        "max_source_length": 512,
        "max_target_length": 1024,
        "max_molecules_per_sequence": 8,
        "num_workers": 0,
    }
    assert gflownet_config["model"]["checkpoint"] == "outputs/multi_molecule_sft_lpm24/checkpoints/best"
    assert gflownet_config["gflownet"]["rollout"]["stage_separator"] == " "

    config_path = tmp_path / "multi_molecule_gflownet_lpm24.yaml"
    config_path.write_text(
        "\n".join(
            [
                "data:",
                "  train_file: data/lpm24/grouped_splits/train_multimol.jsonl",
                "  validation_file: data/lpm24/grouped_splits/validation_multimol.jsonl",
                "  test_file: data/lpm24/grouped_splits/test_multimol.jsonl",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    dataset_kind = infer_managed_dataset(
        stage_spec=resolve_stage_spec("gflownet"),
        config_path=config_path,
        repo_dir=tmp_path,
    )
    assert dataset_kind == "manual"
