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
