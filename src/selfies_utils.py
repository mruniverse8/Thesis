from __future__ import annotations

import re

import selfies as sf

from .prompting import normalize_selfies_text, unwrap_selfies_target


SELFIES_TOKEN_PATTERN = re.compile(r"(\[[^\]]+\]\.?)")


def filter_selfies(text: str) -> str:
    matches = SELFIES_TOKEN_PATTERN.findall(text)
    return "".join(matches).replace("[[", "[").replace("]]", "]")


def normalize_generated_selfies(text: str) -> str:
    return unwrap_selfies_target(text.strip())


def decode_selfies_to_smiles(selfies_text: str) -> tuple[str | None, bool]:
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
