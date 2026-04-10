from __future__ import annotations

import math
from typing import Any

import selfies as sf

from molecules.parsing import parse_molecule_text
from molecules.selfies import normalize_selfies_text
from src.prompting import normalize_free_text


def _string_or_empty(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    return str(value)


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


def _iter_candidate_fields(raw_record: dict[str, Any]) -> list[tuple[str, str]]:
    candidates: list[tuple[str, str]] = []

    for key in ("target_selfies_list", "selfies_list"):
        for item in _as_list(raw_record.get(key)):
            candidates.append((_string_or_empty(item), ""))

    for key in ("target_smiles_list", "smiles_list"):
        for item in _as_list(raw_record.get(key)):
            candidates.append(("", _string_or_empty(item)))

    for key in ("targets", "molecules"):
        for item in _as_list(raw_record.get(key)):
            if isinstance(item, dict):
                candidates.append(
                    (
                        _string_or_empty(item.get("selfies") or item.get("SELFIES")),
                        _string_or_empty(item.get("smiles") or item.get("SMILES")),
                    )
                )
            else:
                candidates.append((_string_or_empty(item), ""))

    single_selfies = _string_or_empty(raw_record.get("selfies") or raw_record.get("SELFIES"))
    single_smiles = _string_or_empty(raw_record.get("smiles") or raw_record.get("SMILES"))
    if single_selfies or single_smiles:
        candidates.append((single_selfies, single_smiles))

    return candidates


def _canonical_smiles_for_selfies(selfies_text: str) -> str | None:
    try:
        record = parse_molecule_text(selfies_text, representation="selfies")
    except ImportError:
        try:
            return sf.decoder(selfies_text)
        except Exception:
            return None

    if not record.is_valid or record.canonical_smiles is None:
        return None
    return record.canonical_smiles


def build_multi_molecule_processed_record(
    raw_record: dict[str, Any],
    split: str,
    index: int,
    max_molecules_per_sequence: int = 8,
    convert_missing_selfies_from_smiles: bool = True,
) -> dict[str, Any]:
    description = normalize_free_text(_string_or_empty(raw_record.get("description")))
    if not description:
        raise ValueError("Missing description")

    seen_keys: set[str] = set()
    target_selfies_list: list[str] = []
    target_smiles_list: list[str] = []

    for selfies_text, smiles_text in _iter_candidate_fields(raw_record):
        normalized_selfies = normalize_selfies_text(selfies_text)
        normalized_smiles = _string_or_empty(smiles_text).strip()

        if not normalized_selfies and convert_missing_selfies_from_smiles and normalized_smiles:
            try:
                normalized_selfies = normalize_selfies_text(sf.encoder(normalized_smiles))
            except Exception:
                normalized_selfies = ""

        if not normalized_selfies:
            continue

        try:
            sf.decoder(normalized_selfies)
        except Exception:
            continue

        canonical_smiles = _canonical_smiles_for_selfies(normalized_selfies)
        dedupe_key = canonical_smiles or normalized_selfies
        if dedupe_key in seen_keys:
            continue

        seen_keys.add(dedupe_key)
        target_selfies_list.append(normalized_selfies)
        if canonical_smiles:
            target_smiles_list.append(canonical_smiles)

        if len(target_selfies_list) >= max_molecules_per_sequence:
            break

    if not target_selfies_list:
        raise ValueError("Missing valid target molecules")

    cid = _string_or_empty(raw_record.get("id") or raw_record.get("CID")).strip()
    processed = {
        "id": cid or f"{split}-{index:06d}",
        "description": description,
        "target_selfies_list": target_selfies_list,
    }
    if target_smiles_list:
        processed["target_smiles_list"] = target_smiles_list
    return processed


__all__ = ["build_multi_molecule_processed_record"]
