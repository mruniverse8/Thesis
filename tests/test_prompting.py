from src.prompting import build_text2mol_prompt, unwrap_selfies_target, wrap_selfies_target


def test_build_text2mol_prompt_matches_expected_shape():
    prompt = build_text2mol_prompt("A simple alcohol.")
    assert "Definition: You are given a molecule description in English." in prompt
    assert "Input: A simple alcohol." in prompt
    assert prompt.endswith("Output: ")


def test_selfies_target_round_trip_normalizes_spaces():
    wrapped = wrap_selfies_target("[C] [O]")
    assert wrapped == "<bom>[C][O]<eom>"
    assert unwrap_selfies_target(wrapped) == "[C][O]"


def test_build_text2mol_prompt_accepts_custom_definition():
    prompt = build_text2mol_prompt("A simple alcohol.", definition="Custom definition.")

    assert prompt.startswith("Custom definition.")
