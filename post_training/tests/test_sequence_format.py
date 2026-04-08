from post_training.sequence_format import (
    MOL_SEPARATOR_TOKEN,
    append_stage_to_prefix,
    build_stage_prefix,
    parse_molecule_sequence,
    serialize_molecule_sequence,
)


def test_serialize_and_parse_molecule_sequence_round_trip() -> None:
    sequence = serialize_molecule_sequence(["[C][O]", "[C][C]"])

    assert sequence == "<bom>[C][O]<mol_sep>[C][C]<eom>"
    assert parse_molecule_sequence(sequence) == ["[C][O]", "[C][C]"]


def test_build_stage_prefix_and_append_stage() -> None:
    prefix = build_stage_prefix(["[C][O]"])
    extended = append_stage_to_prefix(prefix, "[C][C]", MOL_SEPARATOR_TOKEN)

    assert prefix == "<bom>[C][O]<mol_sep>"
    assert extended == "<bom>[C][O]<mol_sep>[C][C]<mol_sep>"
