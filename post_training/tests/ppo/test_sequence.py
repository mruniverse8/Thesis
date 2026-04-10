from post_training.ppo.sequence import build_completed_sequence, decode_stage_text


def test_decode_stage_text_handles_wrapped_stage() -> None:
    molecule, stop_token = decode_stage_text("<bom>[C][O]<eom>")
    partial_molecule, partial_stop_token = decode_stage_text("[C][C]")

    assert molecule == "[C][O]"
    assert stop_token == "<eom>"
    assert partial_molecule == "[C][C]"
    assert partial_stop_token is None


def test_build_completed_sequence_serializes_staged_molecules() -> None:
    assert build_completed_sequence(["[C][O]", "[C][C]"]) == "<bom>[C][O]<eom> <bom>[C][C]<eom>"
