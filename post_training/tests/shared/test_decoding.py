from __future__ import annotations

from pathlib import Path

from post_training.shared.decoding import build_stage_token_constraints


class GreedyTokenizer:
    def __init__(self, vocab: dict[str, int]) -> None:
        self._vocab = dict(vocab)
        self._tokens = sorted(self._vocab, key=len, reverse=True)

    def convert_tokens_to_ids(self, token: str) -> int:
        return int(self._vocab.get(str(token), -1))

    def __call__(
        self,
        text: str,
        *,
        add_special_tokens: bool = False,
        return_attention_mask: bool = False,
    ) -> dict[str, list[int]]:
        del add_special_tokens, return_attention_mask
        remaining = str(text)
        token_ids: list[int] = []
        while remaining:
            matched_token = None
            for token in self._tokens:
                if remaining.startswith(token):
                    matched_token = token
                    break
            if matched_token is None:
                raise ValueError(f"Tokenizer could not encode: {text!r}")
            token_ids.append(int(self._vocab[matched_token]))
            remaining = remaining[len(matched_token) :]
        return {"input_ids": token_ids}


def test_build_stage_token_constraints_unions_dictionary_and_dataset_ids(
    tmp_path: Path,
) -> None:
    selfies_dict_path = tmp_path / "selfies_dict.txt"
    selfies_dict_path.write_text("[C]\n[N]\n", encoding="utf-8")
    tokenizer = GreedyTokenizer(
        {
            "<bom>": 0,
            "<eom>": 1,
            "[C]": 2,
            "[N]": 3,
            "[C][O]": 4,
            " ": 5,
        }
    )

    constraints = build_stage_token_constraints(
        tokenizer,
        examples=[{"target_selfies_list": ["[C][O]"]}],
        selfies_dict_path=selfies_dict_path,
        separator_token=" ",
    )

    assert constraints.bom_token_id == 0
    assert constraints.eom_token_id == 1
    assert set(constraints.content_token_ids) == {2, 3, 4}
    assert constraints.separator_token_ids == (5,)
