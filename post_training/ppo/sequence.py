from __future__ import annotations

from src.constants import EOM_TOKEN

from post_training.shared.sequence import build_stage_prefix, parse_single_staged_molecule, serialize_staged_target


def decode_stage_text(stage_text: str) -> tuple[str, str | None]:
    molecule = parse_single_staged_molecule(stage_text)
    if molecule is not None:
        return molecule, EOM_TOKEN
    return str(stage_text).strip(), None


def build_completed_sequence(molecules: list[str]) -> str:
    return serialize_staged_target(molecules)


__all__ = [
    "build_completed_sequence",
    "build_stage_prefix",
    "decode_stage_text",
]
