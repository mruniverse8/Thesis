from post_training.sft_prompting import build_diverse_text2mol_prompt


def test_build_diverse_text2mol_prompt_mentions_diverse_set() -> None:
    prompt = build_diverse_text2mol_prompt("An alcohol")

    assert "diverse set of molecule SELFIES" in prompt
    assert "Input: An alcohol" in prompt
    assert prompt.endswith("Output: ")
