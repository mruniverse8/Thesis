from __future__ import annotations

import pytest

pytest.importorskip("selfies")
pytest.importorskip("rdkit")

from molecules import parse_molecule_text as canonical_parse_molecule_text
from molecules.datasets.multi import (
    build_multi_molecule_processed_record as canonical_build_multi_molecule_processed_record,
)
from molecules.datasets.single import build_processed_record as canonical_build_processed_record
from molecules.selfies import decode_selfies_to_smiles as canonical_decode_selfies_to_smiles
from post_training.sft_multi.dataset import (
    build_multi_molecule_processed_record as wrapped_build_multi_molecule_processed_record,
)
from reward_utils import parse_molecule_text as wrapped_parse_molecule_text
from src.datasets import build_processed_record as wrapped_build_processed_record
from src.selfies_utils import decode_selfies_to_smiles as wrapped_decode_selfies_to_smiles


def test_wrapper_and_canonical_parse_paths_match() -> None:
    canonical_record = canonical_parse_molecule_text("[C][C][O]", representation="selfies")
    wrapped_record = wrapped_parse_molecule_text("[C][C][O]", representation="selfies")

    assert canonical_record.is_valid is True
    assert canonical_record.canonical_smiles == wrapped_record.canonical_smiles == "CCO"


def test_wrapper_and_canonical_selfies_decode_paths_match() -> None:
    canonical = canonical_decode_selfies_to_smiles("noise [C][O] noise")
    wrapped = wrapped_decode_selfies_to_smiles("noise [C][O] noise")

    assert canonical == wrapped == ("CO", True)


def test_wrapper_and_canonical_single_record_builders_match() -> None:
    raw_record = {
        "CID": 42,
        "description": " small alcohol ",
        "SELFIES": None,
        "SMILES": "CO",
    }

    canonical = canonical_build_processed_record(raw_record, split="train", index=0)
    wrapped = wrapped_build_processed_record(raw_record, split="train", index=0)

    assert canonical == wrapped


def test_wrapper_and_canonical_multi_record_builders_match() -> None:
    raw_record = {
        "id": "example-1",
        "description": "Example description",
        "targets": [
            {"selfies": "[C][O]"},
            {"smiles": "CC"},
            {"selfies": "[C][O]"},
        ],
    }

    canonical = canonical_build_multi_molecule_processed_record(raw_record, split="train", index=0)
    wrapped = wrapped_build_multi_molecule_processed_record(raw_record, split="train", index=0)

    assert canonical == wrapped
