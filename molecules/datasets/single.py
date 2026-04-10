from __future__ import annotations

import math
from typing import Any

import selfies as sf

from molecules.selfies import normalize_selfies_text
from src.prompting import normalize_free_text


def _string_or_empty(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    return str(value)


def build_processed_record(
    raw_record: dict[str, Any],
    split: str,
    index: int,
    convert_missing_selfies_from_smiles: bool = True,
) -> dict[str, str]:
    description = normalize_free_text(_string_or_empty(raw_record.get("description")))
    smiles = _string_or_empty(raw_record.get("SMILES")).strip()
    selfies_text = normalize_selfies_text(_string_or_empty(raw_record.get("SELFIES")))

    if not selfies_text and convert_missing_selfies_from_smiles and smiles:
        selfies_text = normalize_selfies_text(sf.encoder(smiles))

    if not description:
        raise ValueError("Missing description")
    if not selfies_text:
        raise ValueError("Missing SELFIES")

    sf.decoder(selfies_text)

    cid = _string_or_empty(raw_record.get("CID")).strip()
    processed = {
        "id": cid or f"{split}-{index:06d}",
        "description": description,
        "selfies": selfies_text,
    }
    if smiles:
        processed["source_smiles"] = smiles
    return processed


__all__ = ["build_processed_record"]
