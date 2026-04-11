from __future__ import annotations

import re
from typing import Any

from src.constants import BOM_TOKEN, EOM_TOKEN


SELFIES_TOKEN_PATTERN = re.compile(r"(\[[^\]]+\]\.?)")
SELFIES_FULL_PATTERN = re.compile(r"(?:\[[^\]]+\]\.?)+")
GENERATION_WRAPPER_TOKENS = ("<pad>", "</s>", "<s>")


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


def clean_biot5_selfies_text(text: str) -> str:
    cleaned = normalize_selfies_text(text)
    for token in GENERATION_WRAPPER_TOKENS:
        cleaned = cleaned.replace(token, "")
    return unwrap_selfies_target(cleaned)


def normalize_generated_selfies(text: str) -> str:
    return clean_biot5_selfies_text(text)


def decode_biot5_selfies(
    raw_prediction_text: str,
    *,
    allow_filter_fallback: bool = True,
) -> dict[str, Any]:
    import selfies as sf

    cleaned_selfies = clean_biot5_selfies_text(raw_prediction_text)
    result: dict[str, Any] = {
        "cleaned_selfies": cleaned_selfies,
        "parsed_selfies": None,
        "selected_selfies": None,
        "filtered_selfies": None,
        "used_filter_selfies_fallback": False,
        "decoded_smiles": None,
        "is_valid_selfies": False,
        "selfies_decode_error": None,
    }

    if not cleaned_selfies:
        result["selfies_decode_error"] = "empty_selfies"
        return result

    if "[" not in cleaned_selfies or "]" not in cleaned_selfies:
        result["selfies_decode_error"] = "missing_selfies_brackets"
        if not allow_filter_fallback:
            return result

        filtered_selfies = filter_selfies(cleaned_selfies)
        result["filtered_selfies"] = filtered_selfies or None
        if not filtered_selfies:
            return result

        try:
            result["parsed_selfies"] = filtered_selfies
            result["selected_selfies"] = filtered_selfies
            result["decoded_smiles"] = sf.decoder(filtered_selfies)
            result["used_filter_selfies_fallback"] = True
            result["is_valid_selfies"] = True
            result["selfies_decode_error"] = None
            return result
        except Exception as exc:
            result["selfies_decode_error"] = f"{type(exc).__name__}: {exc}"
            return result

    try:
        result["parsed_selfies"] = cleaned_selfies
        result["selected_selfies"] = cleaned_selfies
        result["decoded_smiles"] = sf.decoder(cleaned_selfies)
        result["is_valid_selfies"] = True
        return result
    except Exception as exc:
        result["selfies_decode_error"] = f"{type(exc).__name__}: {exc}"

    if not allow_filter_fallback:
        return result

    filtered_selfies = filter_selfies(cleaned_selfies)
    result["filtered_selfies"] = filtered_selfies or None
    if not filtered_selfies or filtered_selfies == cleaned_selfies:
        return result

    try:
        result["parsed_selfies"] = filtered_selfies
        result["selected_selfies"] = filtered_selfies
        result["decoded_smiles"] = sf.decoder(filtered_selfies)
        result["used_filter_selfies_fallback"] = True
        result["is_valid_selfies"] = True
        result["selfies_decode_error"] = None
        return result
    except Exception as exc:
        result["selfies_decode_error"] = f"{type(exc).__name__}: {exc}"
        return result


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
    result = decode_biot5_selfies(text)
    return (
        result["selected_selfies"],
        result["decoded_smiles"],
        bool(result["used_filter_selfies_fallback"]),
    )


__all__ = [
    "SELFIES_FULL_PATTERN",
    "SELFIES_TOKEN_PATTERN",
    "GENERATION_WRAPPER_TOKENS",
    "clean_biot5_selfies_text",
    "decode_biot5_selfies",
    "decode_selfies_to_smiles",
    "filter_selfies",
    "looks_like_selfies",
    "normalize_generated_selfies",
    "normalize_selfies_text",
    "parse_generated_selfies",
    "unwrap_selfies_target",
    "wrap_selfies_target",
]
