from __future__ import annotations

import torch
import pytest

import src.tokenizer_utils as tokenizer_utils


class DummyTokenizer:
    def __init__(self, vocab: dict[str, int]) -> None:
        self._vocab = dict(vocab)
        self.model_max_length: int | None = None

    def get_vocab(self) -> dict[str, int]:
        return dict(self._vocab)

    def __len__(self) -> int:
        return len(self._vocab)

    def __call__(self, text: str, add_special_tokens: bool = False) -> dict[str, list[int]]:
        del add_special_tokens
        token_ids = [self._vocab[token] for token in str(text).split() if token in self._vocab]
        return {"input_ids": token_ids}

    def convert_ids_to_tokens(self, token_ids: list[int]) -> list[str]:
        reverse_vocab = {value: key for key, value in self._vocab.items()}
        return [reverse_vocab[int(token_id)] for token_id in token_ids]


class DummyModel:
    def __init__(self, vocab_size: int) -> None:
        self._embedding = type("Embedding", (), {"weight": torch.zeros((vocab_size, 1))})()

    def get_input_embeddings(self):
        return self._embedding


def test_prepare_training_tokenizer_uses_original_checkpoint_tokenizer(monkeypatch) -> None:
    tokenizer = DummyTokenizer({"<bom>": 0, "<eom>": 1, "[C]": 2})

    def fake_from_pretrained(model_name_or_path: str, use_fast: bool = True):
        assert model_name_or_path == "demo-model"
        assert use_fast is True
        return tokenizer

    monkeypatch.setattr(tokenizer_utils.AutoTokenizer, "from_pretrained", fake_from_pretrained)

    loaded_tokenizer, metadata = tokenizer_utils.prepare_training_tokenizer("demo-model")

    assert loaded_tokenizer is tokenizer
    assert tokenizer.model_max_length == int(1e9)
    assert metadata == {
        "vocab_size": 3,
        "added_selfies_tokens": 0,
        "added_special_tokens": 0,
    }


def test_prepare_training_tokenizer_requires_staged_selfies_tokens(monkeypatch) -> None:
    tokenizer = DummyTokenizer({"<bom>": 0, "[C]": 1})
    monkeypatch.setattr(
        tokenizer_utils.AutoTokenizer,
        "from_pretrained",
        lambda *args, **kwargs: tokenizer,
    )

    with pytest.raises(ValueError, match="<eom>"):
        tokenizer_utils.prepare_training_tokenizer("demo-model")


def test_assert_tokenizer_matches_model_vocab_rejects_mismatch() -> None:
    tokenizer = DummyTokenizer({"<bom>": 0, "<eom>": 1, "[C]": 2})
    model = DummyModel(vocab_size=5)

    with pytest.raises(ValueError, match="vocab mismatch"):
        tokenizer_utils.assert_tokenizer_matches_model_vocab(
            tokenizer,
            model,
            context="unit-test",
        )


def test_summarize_tokenizer_encoding_reports_ids_and_tokens() -> None:
    tokenizer = DummyTokenizer({"<bom>": 0, "[C][O]": 1, "<eom>": 2})

    summary = tokenizer_utils.summarize_tokenizer_encoding(
        tokenizer,
        "<bom> [C][O] <eom>",
    )

    assert summary == {
        "text": "<bom> [C][O] <eom>",
        "num_tokens": 3,
        "input_ids": [0, 1, 2],
        "tokens": ["<bom>", "[C][O]", "<eom>"],
    }


def test_build_tokenizer_comparison_report_groups_results_by_tokenizer_name() -> None:
    report = tokenizer_utils.build_tokenizer_comparison_report(
        tokenizers={
            "fast": DummyTokenizer({"<bom>": 0, "<eom>": 1}),
            "slow": DummyTokenizer({"<bom>": 4, "<eom>": 5}),
        },
        texts=["<bom> <eom>"],
    )

    assert set(report) == {"fast", "slow"}
    assert report["fast"][0]["input_ids"] == [0, 1]
    assert report["slow"][0]["input_ids"] == [4, 5]
