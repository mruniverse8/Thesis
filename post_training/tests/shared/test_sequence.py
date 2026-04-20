from src.constants import EOM_TOKEN

from post_training.shared.sequence import (
    STAGE_SEPARATOR,
    append_stage_to_prefix,
    build_stage_prefix,
    parse_single_staged_molecule,
    parse_staged_target,
    serialize_staged_target,
)


def test_serialize_and_parse_staged_target_round_trip() -> None:
    sequence = serialize_staged_target(["[C][O]", "[C][C]"])

    assert sequence == "<bom>[C][O]<eom> <bom>[C][C]<eom>"
    assert parse_staged_target(sequence) == ["[C][O]", "[C][C]"]


def test_build_stage_prefix_and_append_stage() -> None:
    prefix = build_stage_prefix(["[C][O]"])
    completed = append_stage_to_prefix(prefix, "[C][C]", EOM_TOKEN)
    continued = append_stage_to_prefix(prefix, "[C][C]", STAGE_SEPARATOR)

    assert prefix == "<bom>[C][O]<eom> "
    assert completed == "<bom>[C][O]<eom> <bom>[C][C]<eom>"
    assert continued == "<bom>[C][O]<eom> <bom>[C][C]<eom> "


def test_parse_single_staged_molecule_requires_wrappers() -> None:
    assert parse_single_staged_molecule("<bom>[C][O]<eom>") == "[C][O]"
    assert parse_single_staged_molecule("[C][O]") is None


def test_parse_staged_target_rejects_wrapped_words_and_unwrapped_tail() -> None:
    wrapped_plain_words = "<bom>alpha beta gamma<eom>"
    wrapped_malformed_selfies = "<bom>[C] alpha [O]<eom>"
    unwrapped_tail = "[N][C]"
    text = f"{wrapped_plain_words} {wrapped_malformed_selfies} {unwrapped_tail}"

    assert parse_single_staged_molecule(wrapped_plain_words) is None
    assert parse_single_staged_molecule(wrapped_malformed_selfies) is None
    assert parse_staged_target(text) == []


def test_parse_staged_target_rejects_valid_stage_with_unwrapped_suffix() -> None:
    text = "<bom>[C][O]<eom> trailing_text"

    assert parse_single_staged_molecule(text) is None
    assert parse_staged_target(text) == []
