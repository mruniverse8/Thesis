"""Shared helpers for the BioT5 local and Kaggle review notebooks.

The maintained notebooks now center on SELFIES-first BioT5 review flows.
Legacy SMILES-extraction helpers remain here only for diagnostic comparison.
"""

from __future__ import annotations

from collections import defaultdict
import re
from typing import Any, Sequence

import selfies as sf
from rdkit import DataStructs

from molecules.collection.filtering import CollectionMetricConfig, PreparedReference
from molecules.fingerprints import build_morgan_fingerprint
from molecules.parsing import parse_molecule_text
from molecules.selfies import (
    clean_biot5_selfies_text,
    decode_biot5_selfies,
    filter_selfies,
    normalize_selfies_text,
)
from src.prompting import normalize_free_text


SMILES_TAG_PATTERN = re.compile(r"<SMILES>\s*(.*?)\s*</SMILES>", re.IGNORECASE | re.DOTALL)
PREFIXED_SMILES_PATTERN = re.compile(
    r"(?:^|[\s>])SMILES\s*:\s*([A-Za-z0-9@\+\-\[\]\(\)=#$\\/%.]+)",
    re.IGNORECASE,
)
CHEMICAL_SEGMENT_PATTERN = re.compile(r"[A-Za-z0-9@\+\-\[\]\(\)=#$\\/%.]+")
LABEL_PREFIX_PATTERN = re.compile(
    r"^(?:smiles|molecule|answer|output|prediction|result)\s*[:=]\s*(.+)$",
    re.IGNORECASE,
)
SMILES_SPECIAL_CHARS = "[]=()#@\\/+-."
AROMATIC_ALPHA_CHARS = set("bcnops")


def build_prefixed_smiles_prompt(description: str) -> str:
    """Build a prompt variant that asks for a single `SMILES:` output line."""

    normalized_description = normalize_free_text(description)
    return (
        "Definition: You are given a molecule description in English. "
        "Your job is to generate exactly one molecule SMILES that fits the description. "
        "Return only one line in the format `SMILES: <molecule>`.\n\n"
        "Now complete the following example -\n"
        f"Input: {normalized_description}\n"
        "Output: SMILES: "
    )


def build_tagged_smiles_prompt(description: str) -> str:
    """Build a prompt variant that asks BioT5 for one tagged SMILES string."""

    normalized_description = normalize_free_text(description)
    return (
        "Definition: You are given a molecule description in English. "
        "Your job is to generate exactly one molecule SMILES that fits the description. "
        "Return only the molecule inside <SMILES> and </SMILES> tags.\n\n"
        "Now complete the following example -\n"
        f"Input: {normalized_description}\n"
        "Output: <SMILES>"
    )


def extract_tagged_smiles(text: str) -> str | None:
    """Return the first non-empty SMILES payload wrapped in <SMILES> tags."""

    for match in SMILES_TAG_PATTERN.finditer(str(text)):
        candidate = "".join(match.group(1).split())
        if candidate:
            return candidate
    return None


def extract_prefixed_smiles(text: str) -> str | None:
    """Return the first SMILES string emitted in a `SMILES: ...` format."""

    for match in PREFIXED_SMILES_PATTERN.finditer(str(text)):
        for candidate in _candidate_variants(match.group(1)):
            if _is_probable_smiles(candidate):
                return candidate
    return None


def extract_strict_smiles(text: str) -> tuple[str | None, str | None]:
    """Apply the strict extraction policy used for diagnosis.

    Strict extraction only trusts explicit structure:
    1. `<SMILES>...</SMILES>`
    2. `SMILES: ...`
    """

    tagged_smiles = extract_tagged_smiles(text)
    if tagged_smiles:
        return tagged_smiles, "tag_only"

    prefixed_smiles = extract_prefixed_smiles(text)
    if prefixed_smiles:
        return prefixed_smiles, "prefix_only"

    return None, None


def _strip_wrapper(candidate: str) -> str:
    """Remove one layer of outer punctuation used as display wrappers."""

    stripped = candidate.strip().strip(",;")
    pairs = {
        ('"', '"'),
        ("'", "'"),
        ("`", "`"),
        ("(", ")"),
        ("{", "}"),
    }
    if len(stripped) >= 2 and (stripped[0], stripped[-1]) in pairs:
        return stripped[1:-1].strip()
    return stripped


def _candidate_variants(segment: str) -> list[str]:
    """Generate a few cleaned variants from one raw text span."""

    raw = _strip_wrapper(segment)
    if not raw:
        return []

    variants = [raw]

    label_match = LABEL_PREFIX_PATTERN.match(raw)
    if label_match:
        variants.append(_strip_wrapper(label_match.group(1)))

    if ":" in raw:
        prefix, suffix = raw.rsplit(":", maxsplit=1)
        if prefix.isalpha() and len(prefix) >= 4:
            variants.append(_strip_wrapper(suffix))

    seen: set[str] = set()
    deduped: list[str] = []
    for item in variants:
        compact = "".join(item.split())
        if compact and compact not in seen:
            seen.add(compact)
            deduped.append(compact)
    return deduped


def _is_probable_smiles(candidate: str) -> bool:
    """Use simple shape checks to reject obvious prose fragments.

    This helper avoids any English-word dictionary. It only keeps spans that look
    structurally chemical enough to be worth sending to RDKit.
    """

    if not candidate or len(candidate) > 256:
        return False
    if any(char.isdigit() for char in candidate):
        return True
    if any(char in SMILES_SPECIAL_CHARS for char in candidate):
        return True
    if "[" in candidate or "]" in candidate:
        return True
    if candidate.isalpha():
        if candidate.islower():
            return len(candidate) <= 12 and set(candidate) <= AROMATIC_ALPHA_CHARS
        return len(candidate) <= 12 and sum(char.isupper() for char in candidate) >= 2
    return True


def _candidate_score(candidate: str, context: str) -> int:
    """Prefer valid spans that look more chemical than surrounding prose."""

    score = len(candidate)
    if any(char.isdigit() for char in candidate):
        score += 20
    if any(char in SMILES_SPECIAL_CHARS for char in candidate):
        score += 20
    if "[" in candidate and "]" in candidate:
        score += 15
    if re.search(r"[cnops]", candidate):
        score += 10
    if candidate.isalpha():
        score -= 10

    lowered_context = context.lower()
    if "smiles" in lowered_context:
        score += 10
    if "molecule" in lowered_context or "output" in lowered_context:
        score += 5
    return score


def extract_smiles_candidate(text: str) -> str | None:
    """Extract the best parseable SMILES-like span from noisy generation text.

    This is a diagnostic fallback only. It should not be treated as a strict,
    production-ready extraction policy.
    """

    source_text = str(text)
    best_candidate: str | None = None
    best_score: int | None = None

    for match in CHEMICAL_SEGMENT_PATTERN.finditer(source_text):
        context_start = max(0, match.start() - 24)
        context_end = min(len(source_text), match.end() + 24)
        context = source_text[context_start:context_end]

        for candidate in _candidate_variants(match.group(0)):
            if not _is_probable_smiles(candidate):
                continue

            record = parse_molecule_text(candidate, representation="smiles")
            if not record.is_valid:
                continue

            score = _candidate_score(candidate, context)
            if best_score is None or score > best_score:
                best_candidate = candidate
                best_score = score

    return best_candidate


def derive_selfies_from_smiles(smiles_text: str) -> str | None:
    """Canonicalize a SMILES string, then convert it into normalized SELFIES."""

    record = parse_molecule_text(smiles_text, representation="smiles")
    if not record.is_valid or record.canonical_smiles is None:
        return None
    try:
        return normalize_selfies_text(sf.encoder(record.canonical_smiles))
    except Exception:
        return None


def assess_generation_output(
    *,
    description_id: str,
    description: str,
    prompt_variant: str,
    candidate_index: int,
    raw_prediction_text: str,
    references: Sequence[PreparedReference],
    metric_config: CollectionMetricConfig,
    generation_config_name: str | None = None,
    elapsed_seconds_batch: float | None = None,
    seconds_per_sample_batch: float | None = None,
    allow_loose_fallback: bool = True,
) -> dict[str, Any]:
    """Build one notebook review row from a raw BioT5 generation."""

    tagged_smiles = extract_tagged_smiles(raw_prediction_text)
    prefixed_smiles = extract_prefixed_smiles(raw_prediction_text)
    strict_extracted_smiles, strict_extraction_mode_used = extract_strict_smiles(raw_prediction_text)
    diagnostic_loose_smiles = extract_smiles_candidate(raw_prediction_text)

    extracted_smiles = strict_extracted_smiles
    extraction_mode_used = strict_extraction_mode_used
    used_loose_fallback = False
    if extracted_smiles is None and allow_loose_fallback and diagnostic_loose_smiles is not None:
        extracted_smiles = diagnostic_loose_smiles
        extraction_mode_used = "loose_fallback"
        used_loose_fallback = True

    assessment: dict[str, Any] = {
        "description_id": description_id,
        "description": description,
        "prompt_variant": prompt_variant,
        "generation_config_name": generation_config_name,
        "candidate_index": candidate_index,
        "raw_prediction_text": raw_prediction_text,
        "elapsed_seconds_batch": elapsed_seconds_batch,
        "seconds_per_sample_batch": seconds_per_sample_batch,
        "tagged_smiles": tagged_smiles,
        "prefixed_smiles": prefixed_smiles,
        "strict_extracted_smiles": strict_extracted_smiles,
        "strict_extraction_mode_used": strict_extraction_mode_used,
        "diagnostic_loose_smiles": diagnostic_loose_smiles,
        "extracted_smiles": extracted_smiles,
        "extraction_mode_used": extraction_mode_used,
        "used_loose_fallback": used_loose_fallback,
        "canonical_smiles": None,
        "derived_selfies": None,
        "is_valid_smiles": False,
        "smiles_parse_error": None,
        "best_reference_smiles": None,
        "max_dice_similarity": 0.0,
        "passes_similarity_threshold": False,
        "rejection_reason": None,
    }

    if extracted_smiles is None:
        assessment["rejection_reason"] = "no_smiles_found"
        return assessment

    candidate_record = parse_molecule_text(extracted_smiles, representation="smiles")
    assessment["is_valid_smiles"] = candidate_record.is_valid
    assessment["smiles_parse_error"] = candidate_record.error
    assessment["canonical_smiles"] = candidate_record.canonical_smiles

    if not candidate_record.is_valid or candidate_record.canonical_smiles is None:
        assessment["rejection_reason"] = "invalid_smiles"
        return assessment

    assessment["derived_selfies"] = derive_selfies_from_smiles(candidate_record.canonical_smiles)

    fingerprint = build_morgan_fingerprint(
        candidate_record,
        radius=metric_config.fingerprint_radius,
        n_bits=metric_config.fingerprint_num_bits,
    )
    if fingerprint is None:
        assessment["rejection_reason"] = "invalid_smiles"
        return assessment

    best_similarity = 0.0
    best_reference_smiles: str | None = None
    for reference in references:
        similarity = float(DataStructs.DiceSimilarity(fingerprint, reference.fingerprint))
        if similarity > best_similarity:
            best_similarity = similarity
            best_reference_smiles = reference.canonical_smiles

    assessment["best_reference_smiles"] = best_reference_smiles
    assessment["max_dice_similarity"] = best_similarity
    assessment["passes_similarity_threshold"] = (
        best_similarity > metric_config.acceptance_dice_threshold
    )
    return assessment


def assess_biot5_native_generation_output(
    *,
    description_id: str,
    description: str,
    prompt_variant: str,
    candidate_index: int,
    raw_prediction_text: str,
    references: Sequence[PreparedReference],
    metric_config: CollectionMetricConfig,
    generation_config_name: str | None = None,
    elapsed_seconds_batch: float | None = None,
    seconds_per_sample_batch: float | None = None,
    allow_filter_fallback: bool = True,
) -> dict[str, Any]:
    """Build one review row for the native BioT5 SELFIES text2mol path."""

    decode_result = decode_biot5_selfies(
        raw_prediction_text,
        allow_filter_fallback=allow_filter_fallback,
    )
    decoded_smiles = decode_result["decoded_smiles"]

    assessment: dict[str, Any] = {
        "description_id": description_id,
        "description": description,
        "prompt_variant": prompt_variant,
        "generation_config_name": generation_config_name,
        "candidate_index": candidate_index,
        "raw_prediction_text": raw_prediction_text,
        "elapsed_seconds_batch": elapsed_seconds_batch,
        "seconds_per_sample_batch": seconds_per_sample_batch,
        "cleaned_selfies": decode_result["cleaned_selfies"],
        "parsed_selfies": decode_result["parsed_selfies"],
        "selected_selfies": decode_result["selected_selfies"],
        "filtered_selfies": decode_result["filtered_selfies"],
        "used_filter_selfies_fallback": decode_result["used_filter_selfies_fallback"],
        "decoded_smiles": decoded_smiles,
        "is_valid_selfies": decode_result["is_valid_selfies"],
        "selfies_decode_error": decode_result["selfies_decode_error"],
        "canonical_smiles": None,
        "derived_selfies": None,
        "is_valid_smiles": False,
        "smiles_parse_error": None,
        "best_reference_smiles": None,
        "max_dice_similarity": 0.0,
        "passes_similarity_threshold": False,
        "rejection_reason": None,
    }

    if decoded_smiles is None:
        assessment["rejection_reason"] = "invalid_selfies"
        return assessment

    candidate_record = parse_molecule_text(decoded_smiles, representation="smiles")
    assessment["is_valid_smiles"] = candidate_record.is_valid
    assessment["smiles_parse_error"] = candidate_record.error
    assessment["canonical_smiles"] = candidate_record.canonical_smiles

    if not candidate_record.is_valid or candidate_record.canonical_smiles is None:
        assessment["rejection_reason"] = "invalid_smiles"
        return assessment

    assessment["derived_selfies"] = derive_selfies_from_smiles(
        candidate_record.canonical_smiles
    )

    fingerprint = build_morgan_fingerprint(
        candidate_record,
        radius=metric_config.fingerprint_radius,
        n_bits=metric_config.fingerprint_num_bits,
    )
    if fingerprint is None:
        assessment["rejection_reason"] = "invalid_smiles"
        return assessment

    best_similarity = 0.0
    best_reference_smiles: str | None = None
    for reference in references:
        similarity = float(DataStructs.DiceSimilarity(fingerprint, reference.fingerprint))
        if similarity > best_similarity:
            best_similarity = similarity
            best_reference_smiles = reference.canonical_smiles

    assessment["best_reference_smiles"] = best_reference_smiles
    assessment["max_dice_similarity"] = best_similarity
    assessment["passes_similarity_threshold"] = (
        best_similarity > metric_config.acceptance_dice_threshold
    )
    return assessment


def summarize_review_records(records: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Aggregate notebook review rows by config, description, and prompt variant."""

    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[
            (
                str(record["generation_config_name"]),
                str(record["description_id"]),
                str(record["prompt_variant"]),
            )
        ].append(record)

    summaries: list[dict[str, Any]] = []
    for (generation_config_name, description_id, prompt_variant), items in sorted(grouped.items()):
        count = len(items)
        extracted_items = [item for item in items if item.get("extracted_smiles")]
        strict_extracted_items = [item for item in items if item.get("strict_extracted_smiles")]
        valid_items = [item for item in items if item.get("is_valid_smiles")]
        strict_valid_items = [
            item
            for item in items
            if item.get("strict_extracted_smiles") and item.get("is_valid_smiles")
        ]
        similarities = [float(item["max_dice_similarity"]) for item in valid_items]
        unique_canonical_smiles = {
            str(item["canonical_smiles"])
            for item in valid_items
            if item.get("canonical_smiles")
        }
        elapsed_seconds_batch = next(
            (float(item["elapsed_seconds_batch"]) for item in items if item.get("elapsed_seconds_batch") is not None),
            None,
        )
        seconds_per_sample_batch = next(
            (
                float(item["seconds_per_sample_batch"])
                for item in items
                if item.get("seconds_per_sample_batch") is not None
            ),
            None,
        )

        summaries.append(
            {
                "generation_config_name": generation_config_name,
                "description_id": description_id,
                "description": str(items[0]["description"]),
                "prompt_variant": prompt_variant,
                "sample_count": count,
                "elapsed_seconds_batch": elapsed_seconds_batch,
                "seconds_per_sample_batch": seconds_per_sample_batch,
                "strict_extraction_rate": len(strict_extracted_items) / count if count else 0.0,
                "strict_valid_molecule_rate": len(strict_valid_items) / count if count else 0.0,
                "final_extraction_rate": len(extracted_items) / count if count else 0.0,
                "final_valid_molecule_rate": len(valid_items) / count if count else 0.0,
                "tag_extraction_rate": sum(
                    int(item.get("strict_extraction_mode_used") == "tag_only") for item in items
                )
                / count
                if count
                else 0.0,
                "prefix_extraction_rate": sum(
                    int(item.get("strict_extraction_mode_used") == "prefix_only") for item in items
                )
                / count
                if count
                else 0.0,
                "loose_fallback_rate": sum(
                    int(bool(item.get("used_loose_fallback"))) for item in items
                )
                / count
                if count
                else 0.0,
                "loose_recovery_rate": sum(
                    int(bool(item.get("used_loose_fallback")) and bool(item.get("is_valid_smiles")))
                    for item in items
                )
                / count
                if count
                else 0.0,
                "unique_canonical_smiles_count": len(unique_canonical_smiles),
                "avg_max_dice_similarity": sum(similarities) / len(similarities)
                if similarities
                else 0.0,
                "best_max_dice_similarity": max(similarities) if similarities else 0.0,
                "passes_similarity_threshold_rate": sum(
                    int(bool(item.get("passes_similarity_threshold"))) for item in items
                )
                / count
                if count
                else 0.0,
                "no_smiles_found_rate": sum(
                    int(item.get("rejection_reason") == "no_smiles_found") for item in items
                )
                / count
                if count
                else 0.0,
            }
        )
    return summaries


def summarize_biot5_native_review_records(
    records: Sequence[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Aggregate native BioT5 review rows by config, description, and prompt."""

    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[
            (
                str(record["generation_config_name"]),
                str(record["description_id"]),
                str(record["prompt_variant"]),
            )
        ].append(record)

    summaries: list[dict[str, Any]] = []
    for (generation_config_name, description_id, prompt_variant), items in sorted(grouped.items()):
        count = len(items)
        valid_selfies_items = [item for item in items if item.get("is_valid_selfies")]
        valid_smiles_items = [item for item in items if item.get("is_valid_smiles")]
        similarities = [float(item["max_dice_similarity"]) for item in valid_smiles_items]
        unique_canonical_smiles = {
            str(item["canonical_smiles"])
            for item in valid_smiles_items
            if item.get("canonical_smiles")
        }
        elapsed_seconds_batch = next(
            (
                float(item["elapsed_seconds_batch"])
                for item in items
                if item.get("elapsed_seconds_batch") is not None
            ),
            None,
        )
        seconds_per_sample_batch = next(
            (
                float(item["seconds_per_sample_batch"])
                for item in items
                if item.get("seconds_per_sample_batch") is not None
            ),
            None,
        )

        summaries.append(
            {
                "generation_config_name": generation_config_name,
                "description_id": description_id,
                "description": str(items[0]["description"]),
                "prompt_variant": prompt_variant,
                "sample_count": count,
                "elapsed_seconds_batch": elapsed_seconds_batch,
                "seconds_per_sample_batch": seconds_per_sample_batch,
                "valid_selfies_rate": len(valid_selfies_items) / count if count else 0.0,
                "filter_selfies_fallback_rate": sum(
                    int(bool(item.get("used_filter_selfies_fallback"))) for item in items
                )
                / count
                if count
                else 0.0,
                "filter_selfies_recovery_rate": sum(
                    int(
                        bool(item.get("used_filter_selfies_fallback"))
                        and bool(item.get("is_valid_selfies"))
                    )
                    for item in items
                )
                / count
                if count
                else 0.0,
                "valid_smiles_rate": len(valid_smiles_items) / count if count else 0.0,
                "unique_canonical_smiles_count": len(unique_canonical_smiles),
                "avg_max_dice_similarity": sum(similarities) / len(similarities)
                if similarities
                else 0.0,
                "best_max_dice_similarity": max(similarities) if similarities else 0.0,
                "passes_similarity_threshold_rate": sum(
                    int(bool(item.get("passes_similarity_threshold"))) for item in items
                )
                / count
                if count
                else 0.0,
                "invalid_selfies_rate": sum(
                    int(item.get("rejection_reason") == "invalid_selfies") for item in items
                )
                / count
                if count
                else 0.0,
                "invalid_smiles_rate": sum(
                    int(item.get("rejection_reason") == "invalid_smiles") for item in items
                )
                / count
                if count
                else 0.0,
            }
        )
    return summaries


__all__ = [
    "assess_generation_output",
    "assess_biot5_native_generation_output",
    "build_prefixed_smiles_prompt",
    "build_tagged_smiles_prompt",
    "clean_biot5_selfies_text",
    "decode_biot5_selfies",
    "derive_selfies_from_smiles",
    "extract_prefixed_smiles",
    "extract_smiles_candidate",
    "extract_strict_smiles",
    "extract_tagged_smiles",
    "filter_selfies",
    "summarize_biot5_native_review_records",
    "summarize_review_records",
]
