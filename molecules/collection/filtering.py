from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

from rdkit import DataStructs

from molecules.fingerprints import build_morgan_fingerprint
from molecules.parsing import parse_molecule_text
from molecules.representations import MoleculeRepresentation
from molecules.selfies import decode_biot5_selfies, normalize_generated_selfies
from src.prompting import normalize_free_text


@dataclass(frozen=True)
class CollectionMetricConfig:
    fingerprint_radius: int
    fingerprint_num_bits: int
    acceptance_dice_threshold: float


@dataclass(frozen=True)
class ReferenceCandidate:
    molecule_text: str
    representation: MoleculeRepresentation


@dataclass(frozen=True)
class PreparedReference:
    canonical_smiles: str
    fingerprint: object


def prepare_reference_groups(
    records: list[dict[str, Any]],
    metric_config: CollectionMetricConfig,
) -> dict[str, tuple[PreparedReference, ...]]:
    grouped_references: dict[str, list[ReferenceCandidate]] = {}

    for index, raw_record in enumerate(records):
        description = normalize_free_text(str(raw_record.get("description", "")))
        if not description:
            raise ValueError(f"Reference record {index} is missing a description.")

        source_smiles = str(raw_record.get("source_smiles") or "").strip()
        selfies_text = str(raw_record.get("selfies") or "").strip()
        if source_smiles:
            grouped_references.setdefault(description, []).append(
                ReferenceCandidate(molecule_text=source_smiles, representation="smiles")
            )
            continue
        if selfies_text:
            grouped_references.setdefault(description, []).append(
                ReferenceCandidate(molecule_text=selfies_text, representation="selfies")
            )
            continue
        raise ValueError(
            f"Reference record {index} must include either `source_smiles` or `selfies`."
        )

    prepared_by_description: dict[str, tuple[PreparedReference, ...]] = {}
    for description, references in grouped_references.items():
        prepared: list[PreparedReference] = []
        for reference in references:
            record = parse_molecule_text(reference.molecule_text, representation=reference.representation)
            fingerprint = build_morgan_fingerprint(
                record,
                radius=metric_config.fingerprint_radius,
                n_bits=metric_config.fingerprint_num_bits,
            )
            if not record.is_valid or fingerprint is None or record.canonical_smiles is None:
                continue
            prepared.append(
                PreparedReference(
                    canonical_smiles=record.canonical_smiles,
                    fingerprint=fingerprint,
                )
            )
        if not prepared:
            raise ValueError(f"No valid reference molecules available for description: {description!r}")
        prepared_by_description[description] = tuple(prepared)

    return prepared_by_description


def assess_candidate(
    *,
    candidate_id: str,
    description_id: str,
    description: str,
    raw_prediction_text: str,
    references: Sequence[PreparedReference],
    metric_config: CollectionMetricConfig,
) -> dict[str, Any]:
    normalized_prediction = normalize_generated_selfies(raw_prediction_text)
    decode_result = decode_biot5_selfies(raw_prediction_text)
    assessment = {
        "id": candidate_id,
        "description_id": description_id,
        "description": description,
        "raw_prediction_text": raw_prediction_text,
        "normalized_prediction_selfies": normalized_prediction,
        "cleaned_selfies": decode_result["cleaned_selfies"],
        "parsed_selfies": decode_result["parsed_selfies"],
        "selected_selfies": decode_result["selected_selfies"],
        "filtered_selfies": decode_result["filtered_selfies"],
        "decoded_smiles": decode_result["decoded_smiles"],
        "canonical_smiles": None,
        "used_repair": bool(decode_result["used_filter_selfies_fallback"]),
        "used_filter_selfies_fallback": bool(decode_result["used_filter_selfies_fallback"]),
        "is_valid_selfies": bool(decode_result["is_valid_selfies"]),
        "selfies_decode_error": decode_result["selfies_decode_error"],
        "accepted": False,
        "max_dice_similarity": 0.0,
        "best_reference_smiles": None,
        "rejection_reason": None,
    }

    if not normalized_prediction:
        assessment["rejection_reason"] = "empty_output"
        return assessment

    selfies_text = decode_result["selected_selfies"]
    decoded_smiles = decode_result["decoded_smiles"]

    if not selfies_text or not decoded_smiles:
        assessment["rejection_reason"] = "invalid_selfies"
        return assessment

    candidate_record = parse_molecule_text(selfies_text, representation="selfies")
    if not candidate_record.is_valid or candidate_record.canonical_smiles is None:
        assessment["rejection_reason"] = "invalid_molecule"
        return assessment

    fingerprint = build_morgan_fingerprint(
        candidate_record,
        radius=metric_config.fingerprint_radius,
        n_bits=metric_config.fingerprint_num_bits,
    )
    if fingerprint is None:
        assessment["rejection_reason"] = "invalid_molecule"
        return assessment

    best_similarity = 0.0
    best_reference_smiles: str | None = None
    for reference in references:
        similarity = float(DataStructs.DiceSimilarity(fingerprint, reference.fingerprint))
        if similarity > best_similarity:
            best_similarity = similarity
            best_reference_smiles = reference.canonical_smiles

    assessment["canonical_smiles"] = candidate_record.canonical_smiles
    assessment["max_dice_similarity"] = best_similarity
    assessment["best_reference_smiles"] = best_reference_smiles

    if best_similarity <= metric_config.acceptance_dice_threshold:
        assessment["rejection_reason"] = "unaccepted_molecule"
        return assessment

    assessment["accepted"] = True
    return assessment


__all__ = [
    "CollectionMetricConfig",
    "PreparedReference",
    "ReferenceCandidate",
    "assess_candidate",
    "prepare_reference_groups",
]
