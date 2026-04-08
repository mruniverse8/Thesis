from __future__ import annotations

from src.constants import EOM_TOKEN
from src.prompting import normalize_selfies_text

from .sequence_format import (
    MOL_SEPARATOR_TOKEN,
    append_stage_to_prefix,
    build_stage_prefix,
    serialize_molecule_sequence,
)


def decode_stage_text(
    stage_text: str,
    separator_token: str = MOL_SEPARATOR_TOKEN,
) -> tuple[str, str | None]:
    cleaned = str(stage_text).strip()
    stop_token = None

    if cleaned.endswith(separator_token):
        stop_token = separator_token
        cleaned = cleaned[: -len(separator_token)]
    elif cleaned.endswith(EOM_TOKEN):
        stop_token = EOM_TOKEN
        cleaned = cleaned[: -len(EOM_TOKEN)]

    return normalize_selfies_text(cleaned), stop_token


def build_completed_sequence(
    molecules: list[str],
    separator_token: str = MOL_SEPARATOR_TOKEN,
) -> str:
    return serialize_molecule_sequence(molecules, separator_token=separator_token)


__all__ = [
    "MOL_SEPARATOR_TOKEN",
    "append_stage_to_prefix",
    "build_completed_sequence",
    "build_stage_prefix",
    "decode_stage_text",
]
