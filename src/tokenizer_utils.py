from __future__ import annotations

from typing import Any, Sequence

from transformers import AutoTokenizer

from .constants import BOM_TOKEN, EOM_TOKEN


def ensure_tokenizer_supports_tokens(
    tokenizer: Any,
    required_tokens: Sequence[str] = (BOM_TOKEN, EOM_TOKEN),
) -> None:
    vocabulary = tokenizer.get_vocab()
    missing_tokens = [str(token) for token in required_tokens if str(token) not in vocabulary]
    if missing_tokens:
        missing_text = ", ".join(repr(token) for token in missing_tokens)
        raise ValueError(
            f"Tokenizer is missing required tokens: {missing_text}. "
            "Use a checkpoint whose original tokenizer already includes the staged SELFIES tokens."
        )


def assert_tokenizer_matches_model_vocab(
    tokenizer: Any,
    model: Any,
    *,
    context: str,
) -> None:
    tokenizer_size = len(tokenizer)
    embedding_size = int(model.get_input_embeddings().weight.size(0))
    if embedding_size != tokenizer_size:
        raise ValueError(
            f"{context} tokenizer/model vocab mismatch: "
            f"model embeddings={embedding_size}, tokenizer={tokenizer_size}. "
            "The active training path no longer resizes embeddings; use a checkpoint with the original matching tokenizer."
        )


def prepare_training_tokenizer(
    model_name_or_path: str,
) -> tuple[Any, dict[str, int]]:
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)
    tokenizer.model_max_length = int(1e9)
    ensure_tokenizer_supports_tokens(tokenizer)

    metadata = {
        "vocab_size": len(tokenizer),
        "added_selfies_tokens": 0,
        "added_special_tokens": 0,
    }
    return tokenizer, metadata
