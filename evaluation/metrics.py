from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from reward_utils.defaults import DEFAULT_ACCEPTANCE_TANIMOTO_THRESHOLD
from reward_utils.fingerprints import build_morgan_fingerprint, ensure_molecule_record
from reward_utils.validation import MoleculeRecord, MoleculeRepresentation, ensure_rdkit, parse_molecule_text

from .config import MoleculeMetricConfig
from .grouping import ReferenceMolecule

try:
    from rdkit import DataStructs
except ImportError:  # pragma: no cover - guarded by ensure_rdkit()
    DataStructs = None


@dataclass(frozen=True)
class CandidateAssessment:
    prediction_id: str
    description: str
    prediction_input: str
    prediction_representation: MoleculeRepresentation
    record: MoleculeRecord = field(repr=False, compare=False)
    fingerprint: object | None = field(default=None, repr=False, compare=False)
    accepted: bool = False
    max_dice_similarity: float = 0.0
    best_reference_smiles: str | None = None

    @property
    def canonical_smiles(self) -> str | None:
        return self.record.canonical_smiles

    @property
    def is_valid(self) -> bool:
        return self.record.is_valid

    def to_report_dict(self) -> dict[str, Any]:
        return {
            "id": self.prediction_id,
            "description": self.description,
            "prediction_input": self.prediction_input,
            "prediction_representation": self.prediction_representation,
            "canonical_smiles": self.canonical_smiles,
            "is_valid": self.is_valid,
            "accepted": self.accepted,
            "max_dice_similarity": self.max_dice_similarity,
            "best_reference_smiles": self.best_reference_smiles,
        }


@dataclass(frozen=True)
class PreparedReference:
    reference: ReferenceMolecule
    record: MoleculeRecord = field(repr=False, compare=False)
    fingerprint: object | None = field(default=None, repr=False, compare=False)


def _prepare_reference(
    reference: ReferenceMolecule,
    *,
    config: MoleculeMetricConfig,
) -> PreparedReference:
    record = parse_molecule_text(reference.molecule_text, representation=reference.representation)
    fingerprint = build_morgan_fingerprint(
        record,
        radius=config.fingerprint_radius,
        n_bits=config.fingerprint_num_bits,
    )
    return PreparedReference(reference=reference, record=record, fingerprint=fingerprint)


def _resolve_prediction_input(
    prediction: Mapping[str, Any],
) -> tuple[str, MoleculeRepresentation]:
    prediction_selfies = str(prediction.get("prediction_selfies") or "").strip()
    if prediction_selfies:
        return prediction_selfies, "selfies"

    prediction_smiles = str(prediction.get("prediction_smiles") or "").strip()
    if prediction_smiles:
        return prediction_smiles, "smiles"

    prediction_text = str(prediction.get("prediction_text") or "").strip()
    return prediction_text, "auto"


def _assess_prediction(
    prediction: Mapping[str, Any],
    references: Sequence[PreparedReference],
    *,
    config: MoleculeMetricConfig,
) -> CandidateAssessment:
    ensure_rdkit()
    prediction_input, representation = _resolve_prediction_input(prediction)
    record = parse_molecule_text(prediction_input, representation=representation)
    fingerprint = build_morgan_fingerprint(
        record,
        radius=config.fingerprint_radius,
        n_bits=config.fingerprint_num_bits,
    )

    best_similarity = 0.0
    best_reference_smiles: str | None = None
    if fingerprint is not None:
        for reference in references:
            if reference.fingerprint is None or reference.record.canonical_smiles is None:
                continue
            similarity = float(DataStructs.DiceSimilarity(fingerprint, reference.fingerprint))
            if similarity > best_similarity:
                best_similarity = similarity
                best_reference_smiles = reference.record.canonical_smiles

    return CandidateAssessment(
        prediction_id=str(prediction.get("id") or ""),
        description=str(prediction.get("description") or "").strip(),
        prediction_input=prediction_input,
        prediction_representation=representation,
        record=record,
        fingerprint=fingerprint,
        accepted=record.is_valid and best_similarity > config.acceptance_dice_threshold,
        max_dice_similarity=best_similarity,
        best_reference_smiles=best_reference_smiles,
    )


def _unique_accepted_assessments(
    assessments: Sequence[CandidateAssessment],
) -> list[CandidateAssessment]:
    unique_assessments: dict[str, CandidateAssessment] = {}
    for assessment in assessments:
        canonical_smiles = assessment.canonical_smiles
        if not assessment.accepted or canonical_smiles is None:
            continue
        unique_assessments.setdefault(canonical_smiles, assessment)
    return list(unique_assessments.values())


def _compute_internal_diversity_from_fingerprints(fingerprints: Sequence[object]) -> float:
    ensure_rdkit()
    if len(fingerprints) < 2:
        return 0.0

    total_distance = 0.0
    total_pairs = 0
    for index, fingerprint in enumerate(fingerprints[:-1]):
        similarities = DataStructs.BulkTanimotoSimilarity(fingerprint, list(fingerprints[index + 1 :]))
        total_distance += sum(1.0 - float(similarity) for similarity in similarities)
        total_pairs += len(similarities)

    return total_distance / total_pairs if total_pairs else 0.0


def _iter_bits(mask: int):
    while mask:
        least_significant_bit = mask & -mask
        yield least_significant_bit.bit_length() - 1
        mask ^= least_significant_bit


def _color_sort(candidates_mask: int, adjacency_masks: Sequence[int]) -> tuple[list[int], list[int]]:
    order: list[int] = []
    bounds: list[int] = []
    remaining = candidates_mask
    color = 0

    while remaining:
        color += 1
        color_class = remaining
        while color_class:
            vertex = next(_iter_bits(color_class))
            vertex_mask = 1 << vertex
            order.append(vertex)
            bounds.append(color)
            remaining &= ~vertex_mask
            color_class &= ~vertex_mask
            color_class &= ~adjacency_masks[vertex]

    return order, bounds


def _maximum_clique_size(adjacency_masks: Sequence[int]) -> int:
    best_size = 0

    def expand(candidates_mask: int, size: int) -> None:
        nonlocal best_size
        if not candidates_mask:
            if size > best_size:
                best_size = size
            return

        ordered_vertices, color_bounds = _color_sort(candidates_mask, adjacency_masks)
        while ordered_vertices:
            vertex = ordered_vertices.pop()
            bound = color_bounds.pop()
            if size + bound <= best_size:
                return
            expand(candidates_mask & adjacency_masks[vertex], size + 1)
            candidates_mask &= ~(1 << vertex)

    expand((1 << len(adjacency_masks)) - 1, 0)
    return best_size


def _compute_ncircles_from_fingerprints(
    fingerprints: Sequence[object],
    *,
    tanimoto_threshold: float,
) -> int:
    ensure_rdkit()
    if not fingerprints:
        return 0
    if len(fingerprints) == 1:
        return 1

    adjacency_masks = [0 for _ in fingerprints]
    for index, fingerprint in enumerate(fingerprints[:-1]):
        similarities = DataStructs.BulkTanimotoSimilarity(fingerprint, list(fingerprints[index + 1 :]))
        for offset, similarity in enumerate(similarities, start=1):
            if float(similarity) < tanimoto_threshold:
                neighbor = index + offset
                adjacency_masks[index] |= 1 << neighbor
                adjacency_masks[neighbor] |= 1 << index

    return _maximum_clique_size(adjacency_masks)


def compute_internal_diversity(
    molecules: Sequence[MoleculeRecord | object | str],
    *,
    representation: MoleculeRepresentation = "auto",
    radius: int = 2,
    n_bits: int = 2048,
) -> float:
    fingerprints = [
        build_morgan_fingerprint(
            molecule,
            representation=representation,
            radius=radius,
            n_bits=n_bits,
        )
        for molecule in molecules
    ]
    valid_fingerprints = [fingerprint for fingerprint in fingerprints if fingerprint is not None]
    return _compute_internal_diversity_from_fingerprints(valid_fingerprints)


def compute_ncircles(
    molecules: Sequence[MoleculeRecord | object | str],
    *,
    representation: MoleculeRepresentation = "auto",
    tanimoto_threshold: float = DEFAULT_ACCEPTANCE_TANIMOTO_THRESHOLD,
    radius: int = 2,
    n_bits: int = 2048,
) -> int:
    fingerprints = [
        build_morgan_fingerprint(
            molecule,
            representation=representation,
            radius=radius,
            n_bits=n_bits,
        )
        for molecule in molecules
    ]
    valid_fingerprints = [fingerprint for fingerprint in fingerprints if fingerprint is not None]
    return _compute_ncircles_from_fingerprints(
        valid_fingerprints,
        tanimoto_threshold=tanimoto_threshold,
    )


def summarize_assessments(
    assessments: Sequence[CandidateAssessment],
    *,
    config: MoleculeMetricConfig,
) -> dict[str, Any]:
    unique_accepted = _unique_accepted_assessments(assessments)
    unique_records = [assessment.record for assessment in unique_accepted]
    unique_fingerprints = [
        assessment.fingerprint
        for assessment in unique_accepted
        if assessment.fingerprint is not None
    ]

    return {
        "num_predictions": len(assessments),
        "num_valid": sum(assessment.is_valid for assessment in assessments),
        "num_accepted": sum(assessment.accepted for assessment in assessments),
        "num_unique_accepted": len(unique_accepted),
        "accepted_unique_count": len(unique_accepted),
        "accepted_unique_smiles": [
            assessment.canonical_smiles
            for assessment in unique_accepted
            if assessment.canonical_smiles is not None
        ],
        "accepted_prediction_ids": [
            assessment.prediction_id
            for assessment in assessments
            if assessment.accepted
        ],
        "accepted_predictions": [
            assessment.to_report_dict()
            for assessment in assessments
            if assessment.accepted
        ],
        "intdiv": _compute_internal_diversity_from_fingerprints(unique_fingerprints),
        "ncircles": _compute_ncircles_from_fingerprints(
            unique_fingerprints,
            tanimoto_threshold=config.ncircles_tanimoto_threshold,
        ),
        "acceptance_dice_threshold": config.acceptance_dice_threshold,
        "ncircles_tanimoto_threshold": config.ncircles_tanimoto_threshold,
        "fingerprint_radius": config.fingerprint_radius,
        "fingerprint_num_bits": config.fingerprint_num_bits,
        "unique_records": [
            ensure_molecule_record(record)
            for record in unique_records
        ],
    }


def compute_group_molecule_metrics(
    predictions: Sequence[Mapping[str, Any]],
    references: Sequence[ReferenceMolecule],
    *,
    config: MoleculeMetricConfig | None = None,
) -> tuple[dict[str, Any], list[CandidateAssessment]]:
    metric_config = config or MoleculeMetricConfig()
    prepared_references = [
        _prepare_reference(reference, config=metric_config)
        for reference in references
    ]
    valid_references = [
        reference
        for reference in prepared_references
        if reference.record.is_valid and reference.fingerprint is not None
    ]
    if not valid_references:
        raise ValueError("At least one valid reference molecule is required to compute evaluation metrics.")

    assessments = [
        _assess_prediction(prediction, valid_references, config=metric_config)
        for prediction in predictions
    ]
    summary = summarize_assessments(assessments, config=metric_config)
    summary.pop("unique_records")
    summary["description"] = str(predictions[0].get("description") or "").strip() if predictions else ""
    summary["num_references"] = len(valid_references)
    summary["reference_smiles"] = list(
        dict.fromkeys(
            reference.record.canonical_smiles
            for reference in valid_references
            if reference.record.canonical_smiles is not None
        )
    )
    return summary, assessments
