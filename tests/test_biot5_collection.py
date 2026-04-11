from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("selfies")
pytest.importorskip("rdkit")

from data_collection import merge_biot5_collection_parts
from data_collection.biot5_collection import StaticCandidateGenerator, collect_biot5_training_data
from data_collection.biot5_generation import (
    BioT5DiverseBeamGenerator,
    build_diverse_beam_generation_kwargs,
)
from data_collection.config_utils import resolve_biot5_collection_config_paths
from src.io_utils import load_yaml, read_jsonl, write_jsonl


def test_build_diverse_beam_generation_kwargs_matches_alt_defaults() -> None:
    kwargs = build_diverse_beam_generation_kwargs(
        {
            "max_length": 512,
            "num_beams": 30,
            "num_return_sequences": 30,
            "num_beam_groups": 6,
            "diversity_penalty": 0.5,
            "early_stopping": True,
            "length_penalty": 1.0,
        },
        target_count=30,
    )

    assert kwargs["do_sample"] is False
    assert kwargs["use_cache"] is True
    assert kwargs["max_length"] == 512
    assert kwargs["num_beams"] == 30
    assert kwargs["num_return_sequences"] == 30
    assert kwargs["num_beam_groups"] == 6
    assert kwargs["diversity_penalty"] == pytest.approx(0.5)
    assert kwargs["early_stopping"] is True
    assert kwargs["length_penalty"] == pytest.approx(1.0)


def test_build_diverse_beam_generation_kwargs_rejects_target_mismatch() -> None:
    with pytest.raises(ValueError, match="target_count=4"):
        build_diverse_beam_generation_kwargs(
            {
                "max_length": 512,
                "num_beams": 30,
                "num_return_sequences": 30,
            },
            target_count=4,
        )


def test_build_diverse_beam_generation_kwargs_rejects_invalid_beam_groups() -> None:
    with pytest.raises(ValueError, match="divisible by num_beam_groups"):
        build_diverse_beam_generation_kwargs(
            {
                "max_length": 512,
                "num_beams": 30,
                "num_return_sequences": 30,
                "num_beam_groups": 4,
                "diversity_penalty": 0.5,
            },
            target_count=30,
        )


def test_biot5_diverse_beam_generator_remote_group_beam_helpers() -> None:
    exc = ValueError("Group Beam Search requires `trust_remote_code=True`")

    assert BioT5DiverseBeamGenerator._needs_remote_group_beam_search(exc) is True
    assert BioT5DiverseBeamGenerator._needs_remote_group_beam_search(ValueError("other")) is False
    assert BioT5DiverseBeamGenerator._remote_group_beam_generation_kwargs({"num_beams": 30}) == {
        "num_beams": 30,
        "custom_generate": "transformers-community/group-beam-search",
        "trust_remote_code": True,
    }


def test_biot5_diverse_beam_generator_accepts_trust_remote_code_via_kwargs() -> None:
    class DummyModel:
        def generate(self, *, custom_generate=None, **kwargs):
            return custom_generate, kwargs

    generator = BioT5DiverseBeamGenerator.__new__(BioT5DiverseBeamGenerator)
    generator.model = DummyModel()

    assert generator._supports_remote_group_beam_search() is True


def test_biot5_diverse_beam_generator_loads_checkpoint_native_tokenizer(monkeypatch) -> None:
    import src.training
    import transformers

    tokenizer_calls: list[tuple[str, int]] = []
    model_calls: list[str] = []

    class DummyTokenizer:
        @staticmethod
        def from_pretrained(model_name_or_path: str, model_max_length: int):
            tokenizer_calls.append((model_name_or_path, model_max_length))
            return DummyTokenizer()

    class DummyModel:
        @staticmethod
        def from_pretrained(model_name_or_path: str):
            model_calls.append(model_name_or_path)
            return DummyModel()

        def to(self, device):
            self.device = device
            return self

        def eval(self):
            return self

        def generate(self, *, custom_generate=None, **kwargs):
            return []

    monkeypatch.setattr(transformers, "T5Tokenizer", DummyTokenizer)
    monkeypatch.setattr(transformers, "T5ForConditionalGeneration", DummyModel)
    monkeypatch.setattr(src.training, "choose_device", lambda device_name: f"resolved:{device_name}")

    generator = BioT5DiverseBeamGenerator(
        model_name_or_path="QizhiPei/biot5-plus-base-chebi20",
        device_name="auto",
        model_max_length=512,
        generation_config={
            "max_length": 512,
            "num_beams": 1,
            "num_return_sequences": 1,
        },
    )

    assert tokenizer_calls == [("QizhiPei/biot5-plus-base-chebi20", 512)]
    assert model_calls == ["QizhiPei/biot5-plus-base-chebi20"]
    assert generator.model_max_length == 512
    assert generator.max_source_length == 512
    assert generator.device == "resolved:auto"


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
            "model_max_length": 512,
            "device": "cpu",
        },
        "data": {
            "train_file": str(train_file),
            "staging_dir": str(tmp_path / "staging"),
            "derived_train_file": str(tmp_path / "derived" / "train_multimol.jsonl"),
        },
        "generation": {
            "target_molecules_per_description": 4,
            "max_length": 512,
            "num_beams": 4,
            "num_return_sequences": 4,
            "num_beam_groups": 2,
            "diversity_penalty": 0.5,
            "early_stopping": True,
            "length_penalty": 1.0,
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
                "<pad><bom>[C][C][O]<eom></s>",
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
    raw_candidates = read_jsonl(tmp_path / "staging" / "raw_candidates.jsonl")

    assert summary["generation_strategy"] == "diverse_beam_search"
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

    first_raw_candidate = raw_candidates[0]
    assert first_raw_candidate["cleaned_prediction_selfies"] == "[C][C][O]"
    assert first_raw_candidate["used_filter_selfies_fallback"] is False

    first_assessment = assessments[0]
    assert first_assessment["cleaned_selfies"] == "[C][C][O]"
    assert first_assessment["selected_selfies"] == "[C][C][O]"
    assert first_assessment["used_filter_selfies_fallback"] is False


def _build_partition_train_records() -> list[dict[str, str]]:
    return [
        {
            "id": "desc-1",
            "description": "molecule one",
            "selfies": "[C]",
            "source_smiles": "C",
        },
        {
            "id": "desc-2",
            "description": "molecule two",
            "selfies": "[C][O]",
            "source_smiles": "CO",
        },
        {
            "id": "desc-3",
            "description": "molecule three",
            "selfies": "[C][C]",
            "source_smiles": "CC",
        },
        {
            "id": "desc-4",
            "description": "molecule four",
            "selfies": "[C][C][O]",
            "source_smiles": "CCO",
        },
        {
            "id": "desc-5",
            "description": "molecule five",
            "selfies": "[C][C][C]",
            "source_smiles": "CCC",
        },
    ]


def _build_partition_config(tmp_path: Path, train_file: Path, *, part_index: int) -> dict[str, object]:
    return {
        "seed": 42,
        "model": {
            "model_name_or_path": "unused-in-test",
            "model_max_length": 512,
            "device": "cpu",
        },
        "data": {
            "train_file": str(train_file),
            "staging_dir": str(tmp_path / f"staging_part_{part_index}"),
            "derived_train_file": str(
                tmp_path / "derived" / f"train_multimol_part_{part_index}.jsonl"
            ),
        },
        "generation": {
            "target_molecules_per_description": 1,
            "max_length": 512,
            "num_beams": 1,
            "num_return_sequences": 1,
        },
        "filtering": {
            "acceptance_dice_threshold": 0.7,
            "fingerprint_radius": 2,
            "fingerprint_num_bits": 2048,
            "max_molecules_per_example": 1,
        },
        "runtime": {
            "description_offset": 0,
            "max_descriptions": None,
            "num_parts": 3,
            "part_index": part_index,
        },
    }


def _build_partition_generator() -> StaticCandidateGenerator:
    return StaticCandidateGenerator(
        {
            "molecule one": ["[C]"],
            "molecule two": ["[C][O]"],
            "molecule three": ["[C][C]"],
            "molecule four": ["[C][C][O]"],
            "molecule five": ["[C][C][C]"],
        }
    )


def test_collect_biot5_training_data_partitions_selected_descriptions(tmp_path) -> None:
    train_file = tmp_path / "train.jsonl"
    write_jsonl(train_file, _build_partition_train_records())

    summary = collect_biot5_training_data(
        _build_partition_config(tmp_path, train_file, part_index=2),
        generator=_build_partition_generator(),
    )

    assert summary["selected_descriptions"] == 2
    assert summary["partition"] == {
        "num_parts": 3,
        "part_index": 2,
        "part_description_count": 2,
        "part_start_index": 2,
        "part_end_index_exclusive": 4,
        "total_selected_descriptions_before_partition": 5,
    }

    inputs = read_jsonl(tmp_path / "staging_part_2" / "inputs.jsonl")
    assert [record["id"] for record in inputs] == ["desc-3", "desc-4"]


def test_merge_biot5_collection_parts_merges_partition_outputs(tmp_path) -> None:
    train_file = tmp_path / "train.jsonl"
    write_jsonl(train_file, _build_partition_train_records())

    part_summaries = []
    part_staging_dirs = []
    part_derived_files = []
    for part_index in (1, 2, 3):
        config = _build_partition_config(tmp_path, train_file, part_index=part_index)
        summary = collect_biot5_training_data(config, generator=_build_partition_generator())
        part_summaries.append(summary)
        part_staging_dirs.append(config["data"]["staging_dir"])
        part_derived_files.append(config["data"]["derived_train_file"])

    merged_summary = merge_biot5_collection_parts(
        part_staging_dirs=part_staging_dirs,
        part_derived_train_files=part_derived_files,
        output_dir=tmp_path / "merged" / "collection_outputs",
        merged_derived_train_file=tmp_path / "merged" / "post_training" / "train_multimol.jsonl",
    )

    assert [item["partition"]["part_index"] for item in part_summaries] == [1, 2, 3]
    assert merged_summary["selected_descriptions"] == 5
    assert merged_summary["descriptions_with_accepted_molecules"] == 5
    assert merged_summary["derived_examples"] == 5
    assert merged_summary["partition"] == {
        "num_parts": 3,
        "merged_part_indices": [1, 2, 3],
        "total_selected_descriptions_before_partition": 5,
    }

    merged_inputs = read_jsonl(tmp_path / "merged" / "collection_outputs" / "inputs.jsonl")
    merged_derived = read_jsonl(tmp_path / "merged" / "post_training" / "train_multimol.jsonl")

    assert [record["id"] for record in merged_inputs] == [
        "desc-1",
        "desc-2",
        "desc-3",
        "desc-4",
        "desc-5",
    ]
    assert [record["id"] for record in merged_derived] == [
        "desc-1",
        "desc-2",
        "desc-3",
        "desc-4",
        "desc-5",
    ]


def test_merge_biot5_collection_parts_rejects_incomplete_partition_set(tmp_path) -> None:
    train_file = tmp_path / "train.jsonl"
    write_jsonl(train_file, _build_partition_train_records())

    part_staging_dirs = []
    part_derived_files = []
    for part_index in (1, 3):
        config = _build_partition_config(tmp_path, train_file, part_index=part_index)
        collect_biot5_training_data(config, generator=_build_partition_generator())
        part_staging_dirs.append(config["data"]["staging_dir"])
        part_derived_files.append(config["data"]["derived_train_file"])

    with pytest.raises(ValueError, match="Expected complete part indices"):
        merge_biot5_collection_parts(
            part_staging_dirs=part_staging_dirs,
            part_derived_train_files=part_derived_files,
            output_dir=tmp_path / "merged" / "collection_outputs",
            merged_derived_train_file=tmp_path / "merged" / "post_training" / "train_multimol.jsonl",
        )
