from __future__ import annotations

from pathlib import Path
from typing import Any

from src.io_utils import write_json
from src.training import save_checkpoint

from post_training.shared.io import initialize_run_output, write_history_json


def prepare_sft_output_dir(
    output_dir: str | Path,
    *,
    config: dict[str, Any],
    tokenizer_metadata: dict[str, Any],
) -> Path:
    resolved_output_dir = initialize_run_output(output_dir, config=config)
    write_json(resolved_output_dir / "tokenizer_metadata.json", tokenizer_metadata)
    return resolved_output_dir


def write_sft_history(output_dir: str | Path, history: list[dict[str, Any]]) -> None:
    write_history_json(output_dir, history)


def save_sft_checkpoint(
    *,
    checkpoint_dir: str | Path,
    model,
    training_tokenizer,
    decoder_tokenizer,
    config: dict[str, Any],
    metrics: dict[str, Any],
) -> None:
    save_checkpoint(
        checkpoint_dir=checkpoint_dir,
        model=model,
        training_tokenizer=training_tokenizer,
        decoder_tokenizer=decoder_tokenizer,
        config=config,
        metrics=metrics,
    )
