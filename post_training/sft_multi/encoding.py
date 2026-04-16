from __future__ import annotations

from typing import Any

from src.tokenizer_utils import prepare_training_tokenizer


def prepare_sft_tokenizers(model_config: dict[str, Any]) -> tuple[Any, dict[str, int]]:
    training_tokenizer, tokenizer_metadata = prepare_training_tokenizer(
        model_name_or_path=str(model_config["name"]),
    )
    return training_tokenizer, tokenizer_metadata
