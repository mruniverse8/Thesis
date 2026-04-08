from __future__ import annotations

import re
from collections.abc import Sequence

from src.constants import BOM_TOKEN, EOM_TOKEN
from src.prompting import normalize_selfies_text


MOL_SEPARATOR_TOKEN = "<mol_sep>"


def get_sequence_special_tokens(separator_token: str = MOL_SEPARATOR_TOKEN) -> list[str]:
    return [BOM_TOKEN, EOM_TOKEN, separator_token]


def _separator_pattern(separator_token: str) -> re.Pattern[str]:
    return re.compile(rf"\s*{re.escape(separator_token)}\s*")


def serialize_molecule_sequence(
    selfies_list: Sequence[str],
    separator_token: str = MOL_SEPARATOR_TOKEN,
) -> str:
    normalized = [normalize_selfies_text(item) for item in selfies_list]
    if not normalized:
        raise ValueError("Expected at least one molecule when serializing a sequence.")
    if any(not item for item in normalized):
        raise ValueError("Encountered an empty molecule while serializing a sequence.")
    return f"{BOM_TOKEN}{separator_token.join(normalized)}{EOM_TOKEN}"


def strip_sequence_wrappers(text: str) -> str:
    return str(text).strip().replace(BOM_TOKEN, "").replace(EOM_TOKEN, "").strip()


def parse_molecule_sequence(
    text: str,
    separator_token: str = MOL_SEPARATOR_TOKEN,
) -> list[str]:
    stripped = strip_sequence_wrappers(text)
    if not stripped:
        return []

    parts = _separator_pattern(separator_token).split(stripped)
    molecules = [normalize_selfies_text(part) for part in parts]
    return [molecule for molecule in molecules if molecule]


def build_stage_prefix(
    previous_selfies_list: Sequence[str],
    separator_token: str = MOL_SEPARATOR_TOKEN,
) -> str:
    normalized = [normalize_selfies_text(item) for item in previous_selfies_list if item]
    if not normalized:
        return BOM_TOKEN
    return f"{BOM_TOKEN}{separator_token.join(normalized)}{separator_token}"


def append_stage_to_prefix(
    prefix_text: str,
    molecule_selfies: str,
    stop_token: str,
    separator_token: str = MOL_SEPARATOR_TOKEN,
) -> str:
    if stop_token not in {separator_token, EOM_TOKEN}:
        raise ValueError(f"Unsupported stop token: {stop_token!r}")

    normalized_prefix = prefix_text.strip()
    normalized_molecule = normalize_selfies_text(molecule_selfies)
    if not normalized_molecule:
        raise ValueError("Expected a non-empty molecule when extending a sequence prefix.")

    return f"{normalized_prefix}{normalized_molecule}{stop_token}"
