from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

from transformers import AutoTokenizer

from .constants import BOM_TOKEN, EOM_TOKEN


def load_selfies_vocab(path_value: str | Path) -> list[str]:
    path = Path(path_value)
    with path.open("r", encoding="utf-8") as handle:
        return [line.strip() for line in handle if line.strip()]


def get_existing_additional_special_tokens(tokenizer: Any) -> list[str]:
    candidates = []

    attr_tokens = getattr(tokenizer, "additional_special_tokens", None)
    if attr_tokens:
        candidates.extend(attr_tokens)

    special_tokens_map = getattr(tokenizer, "special_tokens_map", {}) or {}
    map_tokens = special_tokens_map.get("additional_special_tokens")
    if map_tokens:
        candidates.extend(map_tokens)

    special_tokens_map_extended = getattr(tokenizer, "special_tokens_map_extended", {}) or {}
    extended_tokens = special_tokens_map_extended.get("additional_special_tokens")
    if extended_tokens:
        candidates.extend(extended_tokens)

    deduped: list[str] = []
    seen: set[str] = set()
    for token in candidates:
        token_text = str(token)
        if token_text not in seen:
            deduped.append(token_text)
            seen.add(token_text)
    return deduped


def prepare_training_tokenizer(
    tokenizer_name: str,
    selfies_vocab_path: str | Path,
    extra_special_tokens: Sequence[str] | None = None,
) -> tuple[Any, dict[str, int]]:
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)
    tokenizer.model_max_length = int(1e9)

    vocabulary = tokenizer.get_vocab()
    selfies_vocab = load_selfies_vocab(selfies_vocab_path)
    missing_selfies_tokens = [token for token in selfies_vocab if token not in vocabulary]
    added_selfies = 0
    if missing_selfies_tokens:
        added_selfies = tokenizer.add_tokens(missing_selfies_tokens, special_tokens=True)

    additional_special_tokens = get_existing_additional_special_tokens(tokenizer)
    special_tokens = [BOM_TOKEN, EOM_TOKEN]
    if extra_special_tokens:
        special_tokens.extend(str(token) for token in extra_special_tokens)

    for token in special_tokens:
        if token not in additional_special_tokens:
            additional_special_tokens.append(token)

    added_special = tokenizer.add_special_tokens(
        {"additional_special_tokens": additional_special_tokens}
    )

    metadata = {
        "vocab_size": len(tokenizer),
        "added_selfies_tokens": added_selfies,
        "added_special_tokens": added_special,
    }
    return tokenizer, metadata


def build_decoder_tokenizer(base_tokenizer_name: str, training_tokenizer: Any) -> Any:
    decoder_tokenizer = AutoTokenizer.from_pretrained(base_tokenizer_name, use_fast=True)
    decoder_tokenizer.model_max_length = int(1e9)

    start_index = len(decoder_tokenizer)
    extra_tokens = [
        training_tokenizer.convert_ids_to_tokens(token_id)
        for token_id in range(start_index, len(training_tokenizer))
    ]
    if extra_tokens:
        decoder_tokenizer.add_tokens(extra_tokens, special_tokens=False)

    if len(decoder_tokenizer) != len(training_tokenizer):
        raise ValueError(
            "Decoder tokenizer does not match training tokenizer size: "
            f"{len(decoder_tokenizer)} != {len(training_tokenizer)}"
        )

    for token_id in range(len(training_tokenizer)):
        train_token = training_tokenizer.convert_ids_to_tokens(token_id)
        decode_token = decoder_tokenizer.convert_ids_to_tokens(token_id)
        if train_token != decode_token:
            raise ValueError(
                "Decoder tokenizer does not preserve token ids at "
                f"index {token_id}: {train_token!r} != {decode_token!r}"
            )

    return decoder_tokenizer
