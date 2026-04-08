from __future__ import annotations

from typing import Any, Mapping, Sequence

from .config import MoleculeMetricConfig
from .grouping import (
    SplitReferenceIndex,
    group_predictions_by_description,
    resolve_prediction_description,
)
from .metrics import CandidateAssessment, compute_group_molecule_metrics, summarize_assessments


def _normalize_predictions(
    predictions: Sequence[Mapping[str, Any]],
    reference_index: SplitReferenceIndex,
) -> list[dict[str, Any]]:
    normalized_predictions: list[dict[str, Any]] = []
    for prediction in predictions:
        normalized_prediction = dict(prediction)
        normalized_prediction["description"] = resolve_prediction_description(
            normalized_prediction,
            reference_index,
        )
        normalized_predictions.append(normalized_prediction)
    return normalized_predictions


def _build_split_summary(
    assessments: Sequence[CandidateAssessment],
    *,
    config: MoleculeMetricConfig,
    num_groups: int,
) -> dict[str, Any]:
    base_summary = summarize_assessments(assessments, config=config)
    unique_records = base_summary.pop("unique_records")

    return {
        "num_generated": base_summary.pop("num_predictions"),
        "num_groups": num_groups,
        **base_summary,
        "accepted_unique_smiles": [
            record.canonical_smiles
            for record in unique_records
            if record.canonical_smiles is not None
        ],
    }


def build_prediction_metric_report(
    predictions: Sequence[Mapping[str, Any]],
    reference_index: SplitReferenceIndex,
    *,
    config: MoleculeMetricConfig | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    metric_config = config or MoleculeMetricConfig()
    normalized_predictions = _normalize_predictions(predictions, reference_index)
    predictions_by_description = group_predictions_by_description(normalized_predictions)

    per_group: list[dict[str, Any]] = []
    all_assessments: list[CandidateAssessment] = []

    for description, grouped_predictions in predictions_by_description.items():
        references = reference_index.references_by_description.get(description, ())
        if not references:
            raise ValueError(f"No reference molecules found for description: {description!r}")

        group_summary, assessments = compute_group_molecule_metrics(
            grouped_predictions,
            references,
            config=metric_config,
        )
        per_group.append(group_summary)
        all_assessments.extend(assessments)

    split_summary = _build_split_summary(
        all_assessments,
        config=metric_config,
        num_groups=len(per_group),
    )
    return split_summary, per_group
