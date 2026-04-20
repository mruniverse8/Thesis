from __future__ import annotations

import re
from collections.abc import Sequence

from molecules.selfies import looks_like_selfies, normalize_selfies_text
from src.constants import BOM_TOKEN, EOM_TOKEN


STAGE_SEPARATOR = " "
MOL_SEPARATOR_TOKEN = STAGE_SEPARATOR
_STAGED_MOLECULE_PATTERN = re.compile(
    rf"{re.escape(BOM_TOKEN)}(.*?){re.escape(EOM_TOKEN)}",
    flags=re.DOTALL,
)


def get_sequence_special_tokens(separator_token: str = STAGE_SEPARATOR) -> list[str]:
    del separator_token
    return [BOM_TOKEN, EOM_TOKEN]


def serialize_staged_molecule(selfies_text: str) -> str:
    normalized = normalize_selfies_text(selfies_text)
    if not normalized:
        raise ValueError("Expected a non-empty molecule when serializing a stage.")
    return f"{BOM_TOKEN}{normalized}{EOM_TOKEN}"


def serialize_staged_target(
    selfies_list: Sequence[str],
    separator_token: str = STAGE_SEPARATOR,
) -> str:
    normalized = [normalize_selfies_text(item) for item in selfies_list]
    if not normalized:
        raise ValueError("Expected at least one molecule when serializing a staged target.")
    if any(not item for item in normalized):
        raise ValueError("Encountered an empty molecule while serializing a staged target.")
    return separator_token.join(serialize_staged_molecule(item) for item in normalized)


def strip_sequence_wrappers(text: str) -> str:
    return str(text).replace(BOM_TOKEN, "").replace(EOM_TOKEN, "").strip()


def _is_valid_staged_payload(text: str) -> bool:
    normalized = normalize_selfies_text(text)
    return bool(normalized) and looks_like_selfies(normalized)


def _is_separator_segment(text: str, separator_token: str) -> bool:
    chunk = str(text)
    if not chunk:
        return True
    if not separator_token.strip():
        return not chunk.strip()
    remainder = chunk.replace(separator_token, "")
    return not remainder.strip()


def parse_staged_target(
    text: str,
    separator_token: str = STAGE_SEPARATOR,
) -> list[str]:
    raw_text = str(text)
    molecules: list[str] = []
    cursor = 0
    for match in _STAGED_MOLECULE_PATTERN.finditer(raw_text):
        if not _is_separator_segment(raw_text[cursor : match.start()], separator_token):
            return []
        molecule = normalize_selfies_text(match.group(1))
        if not _is_valid_staged_payload(molecule):
            return []
        molecules.append(molecule)
        cursor = match.end()
    if not _is_separator_segment(raw_text[cursor:], separator_token):
        return []
    return molecules


def parse_single_staged_molecule(text: str) -> str | None:
    match = re.fullmatch(
        rf"\s*{re.escape(BOM_TOKEN)}(.*?){re.escape(EOM_TOKEN)}\s*",
        str(text),
        flags=re.DOTALL,
    )
    if match is None:
        return None
    molecule = normalize_selfies_text(match.group(1))
    if not _is_valid_staged_payload(molecule):
        return None
    return molecule


def serialize_molecule_sequence(
    selfies_list: Sequence[str],
    separator_token: str = STAGE_SEPARATOR,
) -> str:
    return serialize_staged_target(selfies_list, separator_token=separator_token)


def parse_molecule_sequence(
    text: str,
    separator_token: str = STAGE_SEPARATOR,
) -> list[str]:
    return parse_staged_target(text, separator_token=separator_token)


def build_stage_prefix(
    previous_selfies_list: Sequence[str],
    separator_token: str = STAGE_SEPARATOR,
) -> str:
    if not previous_selfies_list:
        return ""
    return f"{serialize_staged_target(previous_selfies_list, separator_token=separator_token)}{separator_token}"


def append_stage_to_prefix(
    prefix_text: str,
    molecule_selfies: str,
    stop_token: str,
    separator_token: str = STAGE_SEPARATOR,
) -> str:
    stage_text = serialize_staged_molecule(molecule_selfies)
    normalized_prefix = str(prefix_text)
    if stop_token == separator_token:
        return f"{normalized_prefix}{stage_text}{separator_token}"
    if stop_token == EOM_TOKEN:
        return f"{normalized_prefix}{stage_text}"
    raise ValueError(f"Unsupported stop token: {stop_token!r}")
