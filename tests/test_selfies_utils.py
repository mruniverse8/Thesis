import pytest

pytest.importorskip("selfies")

from src.selfies_utils import decode_selfies_to_smiles, parse_generated_selfies


def test_decode_selfies_to_smiles_repairs_noisy_output():
    smiles, repaired = decode_selfies_to_smiles("prefix [C][O] suffix")

    assert smiles == "CO"
    assert repaired is True


def test_parse_generated_selfies_preserves_valid_wrapper_output():
    selfies_text, smiles, repaired = parse_generated_selfies("<bom>[C][O]<eom>")

    assert selfies_text == "[C][O]"
    assert smiles == "CO"
    assert repaired is False
