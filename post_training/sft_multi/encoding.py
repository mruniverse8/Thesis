from __future__ import annotations

from pathlib import Path
from typing import Any

from src.tokenizer_utils import build_decoder_tokenizer, prepare_training_tokenizer


def prepare_sft_tokenizers(model_config: dict[str, Any]) -> tuple[Any, Any, dict[str, int]]:
    training_tokenizer, tokenizer_metadata = prepare_training_tokenizer(
        tokenizer_name=str(model_config["tokenizer_name"]),
        selfies_vocab_path=Path(model_config["selfies_vocab_path"]),
        extra_special_tokens=None,
    )
    decoder_tokenizer = build_decoder_tokenizer(
        base_tokenizer_name=str(model_config["base_tokenizer_name"]),
        training_tokenizer=training_tokenizer,
    )
    return training_tokenizer, decoder_tokenizer, tokenizer_metadata
