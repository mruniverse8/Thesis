import pytest

pytest.importorskip("selfies")

from src.selfies_utils import (
    clean_biot5_selfies_text,
    decode_biot5_selfies,
    decode_selfies_to_smiles,
    parse_generated_selfies,
)


def test_clean_biot5_selfies_text_strips_t5_wrappers_and_bom_eom():
    cleaned = clean_biot5_selfies_text(" <pad> <bom> [C] [O] <eom> </s> ")

    assert cleaned == "[C][O]"


def test_decode_selfies_to_smiles_repairs_noisy_output():
    smiles, repaired = decode_selfies_to_smiles("prefix [C][O] suffix")

    assert smiles == "CO"
    assert repaired is True


def test_decode_biot5_selfies_uses_filter_fallback_when_needed():
    decoded = decode_biot5_selfies("Answer:[C][O]trailing")

    assert decoded["cleaned_selfies"] == "Answer:[C][O]trailing"
    assert decoded["parsed_selfies"] == "[C][O]"
    assert decoded["filtered_selfies"] == "[C][O]"
    assert decoded["selected_selfies"] == "[C][O]"
    assert decoded["used_filter_selfies_fallback"] is True
    assert decoded["decoded_smiles"] == "CO"
    assert decoded["is_valid_selfies"] is True


def test_parse_generated_selfies_preserves_valid_wrapper_output():
    selfies_text, smiles, repaired = parse_generated_selfies("<pad><bom>[C][O]<eom></s>")

    assert selfies_text == "[C][O]"
    assert smiles == "CO"
    assert repaired is False
