from __future__ import annotations

import pytest

pytest.importorskip("selfies")
pytest.importorskip("rdkit")

from data_collection.biot5_collection import (
    StaticCandidateGenerator,
    build_contrastive_generation_kwargs,
    collect_biot5_training_data,
)
from data_collection.config_utils import resolve_biot5_collection_config_paths
from src.io_utils import load_yaml, read_jsonl, write_jsonl


def test_build_contrastive_generation_kwargs_matches_paper_style_defaults() -> None:
    kwargs = build_contrastive_generation_kwargs(
        {
            "max_new_tokens": 384,
            "penalty_alpha": 0.5,
            "top_k": 4,
        },
        bad_words_ids=[[1, 2, 3]],
    )

    assert kwargs["do_sample"] is False
    assert kwargs["num_beams"] == 1
    assert kwargs["penalty_alpha"] == pytest.approx(0.5)
    assert kwargs["top_k"] == 4
    assert kwargs["bad_words_ids"] == [[1, 2, 3]]


def test_resolve_biot5_collection_config_paths_resolves_relative_paths(tmp_path) -> None:
    project_root = tmp_path / "project"
    project_root.mkdir()
    config_path = project_root / "config.yaml"
    config_path.write_text(
        (
            "model:\n"
            "  selfies_vocab_path: vocab.txt\n"
            "data:\n"
            "  train_file: data/train.jsonl\n"
            "  staging_dir: outputs/run\n"
            "  derived_train_file: data/train_multimol.jsonl\n"
        ),
        encoding="utf-8",
    )

    resolved = resolve_biot5_collection_config_paths(load_yaml(config_path), project_root=project_root)

    assert resolved["model"]["selfies_vocab_path"] == str(project_root / "vocab.txt")
    assert resolved["data"]["train_file"] == str(project_root / "data" / "train.jsonl")
    assert resolved["data"]["staging_dir"] == str(project_root / "outputs" / "run")
    assert resolved["data"]["derived_train_file"] == str(project_root / "data" / "train_multimol.jsonl")


def test_collect_biot5_training_data_filters_and_derives_grouped_records(tmp_path) -> None:
    train_file = tmp_path / "train.jsonl"
    write_jsonl(
        train_file,
        [
            {
                "id": "desc-1-a",
                "description": "small alcohol molecule",
                "selfies": "[C][C][O]",
                "source_smiles": "CCO",
            },
            {
                "id": "desc-1-b",
                "description": "small alcohol molecule",
                "selfies": "[C][O]",
                "source_smiles": "CO",
            },
            {
                "id": "desc-2",
                "description": "simple hydrocarbon fuel molecule",
                "selfies": "[C][C][C][C]",
                "source_smiles": "CCCC",
            },
        ],
    )

    config = {
        "seed": 42,
        "model": {
            "model_name_or_path": "unused-in-test",
            "tokenizer_name": "unused-in-test",
            "base_tokenizer_name": "unused-in-test",
            "selfies_vocab_path": "unused-in-test",
            "device": "cpu",
        },
        "data": {
            "train_file": str(train_file),
            "staging_dir": str(tmp_path / "staging"),
            "derived_train_file": str(tmp_path / "derived" / "train_multimol.jsonl"),
        },
        "generation": {
            "target_molecules_per_description": 4,
            "max_source_length": 512,
            "max_new_tokens": 384,
            "penalty_alpha": 0.5,
            "top_k": 4,
        },
        "filtering": {
            "acceptance_dice_threshold": 0.7,
            "fingerprint_radius": 2,
            "fingerprint_num_bits": 2048,
            "max_molecules_per_example": 1,
        },
        "runtime": {
            "description_offset": 0,
            "max_descriptions": 2,
        },
    }

    generator = StaticCandidateGenerator(
        {
            "small alcohol molecule": [
                "<bom>[C][C][O]<eom>",
                "noise [C][O] noise",
                "noise [O][C][C] trailing",
                "[O]",
            ],
            "simple hydrocarbon fuel molecule": [
                "[C][C][C][C]",
                "[bad_token]",
                "[O]",
                "<bom>[C][C][C][C]<eom>",
            ],
        }
    )

    summary = collect_biot5_training_data(config, generator=generator)

    grouped_records = read_jsonl(tmp_path / "staging" / "accepted_grouped.jsonl")
    derived_records = read_jsonl(tmp_path / "derived" / "train_multimol.jsonl")
    assessments = read_jsonl(tmp_path / "staging" / "candidate_assessments.jsonl")

    assert summary["selected_descriptions"] == 2
    assert summary["descriptions_with_accepted_molecules"] == 2
    assert summary["rejections_by_reason"] == {
        "duplicate_smiles": 2,
        "invalid_selfies": 1,
        "unaccepted_molecule": 2,
    }

    assert grouped_records == [
        {
            "id": "desc-1-a",
            "description": "small alcohol molecule",
            "reference_smiles_list": ["CCO", "CO"],
            "accepted_target_selfies_list": ["[C][C][O]", "[C][O]"],
            "accepted_target_smiles_list": ["CCO", "CO"],
            "accepted_candidate_count": 2,
            "raw_candidate_count": 4,
        },
        {
            "id": "desc-2",
            "description": "simple hydrocarbon fuel molecule",
            "reference_smiles_list": ["CCCC"],
            "accepted_target_selfies_list": ["[C][C][C][C]"],
            "accepted_target_smiles_list": ["CCCC"],
            "accepted_candidate_count": 1,
            "raw_candidate_count": 4,
        },
    ]
    assert derived_records == [
        {
            "id": "desc-1-a",
            "description": "small alcohol molecule",
            "target_selfies_list": ["[C][C][O]"],
            "target_smiles_list": ["CCO"],
            "accepted_candidate_count": 2,
        },
        {
            "id": "desc-2",
            "description": "simple hydrocarbon fuel molecule",
            "target_selfies_list": ["[C][C][C][C]"],
            "target_smiles_list": ["CCCC"],
            "accepted_candidate_count": 1,
        },
    ]
    assert [item["rejection_reason"] for item in assessments if item["rejection_reason"]] == [
        "duplicate_smiles",
        "unaccepted_molecule",
        "invalid_selfies",
        "unaccepted_molecule",
        "duplicate_smiles",
    ]
