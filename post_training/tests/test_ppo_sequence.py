from post_training.ppo_sequence import build_completed_sequence, decode_stage_text


def test_decode_stage_text_handles_separator_and_eom() -> None:
    molecule, stop_token = decode_stage_text("[C][O]<mol_sep>")
    final_molecule, final_stop_token = decode_stage_text("[C][C]<eom>")

    assert molecule == "[C][O]"
    assert stop_token == "<mol_sep>"
    assert final_molecule == "[C][C]"
    assert final_stop_token == "<eom>"


def test_build_completed_sequence_serializes_molecules() -> None:
    assert build_completed_sequence(["[C][O]", "[C][C]"]) == "<bom>[C][O]<mol_sep>[C][C]<eom>"
