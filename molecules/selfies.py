from __future__ import annotations

import re

from src.constants import BOM_TOKEN, EOM_TOKEN


SELFIES_TOKEN_PATTERN = re.compile(r"(\[[^\]]+\]\.?)")
SELFIES_FULL_PATTERN = re.compile(r"(?:\[[^\]]+\]\.?)+")


def normalize_selfies_text(text: str) -> str:
    return "".join(str(text).split())


def wrap_selfies_target(selfies_text: str) -> str:
    normalized_selfies = normalize_selfies_text(selfies_text)
    return f"{BOM_TOKEN}{normalized_selfies}{EOM_TOKEN}"


def unwrap_selfies_target(text: str) -> str:
    return normalize_selfies_text(text.replace(BOM_TOKEN, "").replace(EOM_TOKEN, ""))


def looks_like_selfies(text: str) -> bool:
    compact = normalize_selfies_text(text)
    return bool(SELFIES_FULL_PATTERN.fullmatch(compact))


def filter_selfies(text: str) -> str:
    matches = SELFIES_TOKEN_PATTERN.findall(text)
    return "".join(matches).replace("[[", "[").replace("]]", "]")


def normalize_generated_selfies(text: str) -> str:
    return unwrap_selfies_target(text.strip())


def decode_selfies_to_smiles(selfies_text: str) -> tuple[str | None, bool]:
    import selfies as sf

    normalized = normalize_selfies_text(selfies_text)
    if not normalized:
        return None, False

    try:
        return sf.decoder(normalized), False
    except Exception:
        repaired = filter_selfies(normalized)
        if not repaired:
            return None, False
        try:
            return sf.decoder(repaired), True
        except Exception:
            return None, False


def parse_generated_selfies(text: str) -> tuple[str | None, str | None, bool]:
    import selfies as sf

    normalized = normalize_generated_selfies(text)
    if not normalized:
        return None, None, False

    try:
        smiles = sf.decoder(normalized)
        return normalized, smiles, False
    except Exception:
        repaired = filter_selfies(normalized)
        if not repaired:
            return None, None, False
        try:
            smiles = sf.decoder(repaired)
            return repaired, smiles, True
        except Exception:
            return None, None, False


__all__ = [
    "SELFIES_FULL_PATTERN",
    "SELFIES_TOKEN_PATTERN",
    "decode_selfies_to_smiles",
    "filter_selfies",
    "looks_like_selfies",
    "normalize_generated_selfies",
    "normalize_selfies_text",
    "parse_generated_selfies",
    "unwrap_selfies_target",
    "wrap_selfies_target",
]
