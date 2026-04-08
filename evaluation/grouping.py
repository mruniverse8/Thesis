from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from reward_utils.validation import MoleculeRepresentation
from src.io_utils import read_jsonl


@dataclass(frozen=True)
class ReferenceMolecule:
    example_id: str
    description: str
    molecule_text: str
    representation: MoleculeRepresentation


@dataclass(frozen=True)
class SplitReferenceIndex:
    dataset_path: Path
    description_by_id: dict[str, str]
    references_by_description: dict[str, tuple[ReferenceMolecule, ...]]


def _normalized_string(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _select_reference_molecule(record: Mapping[str, Any]) -> tuple[str, MoleculeRepresentation]:
    source_smiles = _normalized_string(record.get("source_smiles"))
    if source_smiles:
        return source_smiles, "smiles"

    selfies_text = _normalized_string(record.get("selfies"))
    if selfies_text:
        return selfies_text, "selfies"

    raise ValueError("Reference record must include either `source_smiles` or `selfies`.")


def build_reference_index(
    records: list[dict[str, Any]],
    *,
    dataset_path: str | Path | None = None,
) -> SplitReferenceIndex:
    anchor = Path(dataset_path) if dataset_path is not None else Path("<in-memory>")
    description_by_id: dict[str, str] = {}
    grouped_references: dict[str, list[ReferenceMolecule]] = {}

    for index, record in enumerate(records):
        description = _normalized_string(record.get("description"))
        if not description:
            raise ValueError(f"Reference record {index} is missing a description.")

        example_id = _normalized_string(record.get("id")) or f"reference-{index:06d}"
        molecule_text, representation = _select_reference_molecule(record)

        description_by_id[example_id] = description
        grouped_references.setdefault(description, []).append(
            ReferenceMolecule(
                example_id=example_id,
                description=description,
                molecule_text=molecule_text,
                representation=representation,
            )
        )

    return SplitReferenceIndex(
        dataset_path=anchor,
        description_by_id=description_by_id,
        references_by_description={
            description: tuple(references)
            for description, references in grouped_references.items()
        },
    )


def load_reference_index(dataset_path: str | Path) -> SplitReferenceIndex:
    path = Path(dataset_path)
    return build_reference_index(read_jsonl(path), dataset_path=path)


def resolve_prediction_description(
    prediction: Mapping[str, Any],
    reference_index: SplitReferenceIndex,
) -> str:
    description = _normalized_string(prediction.get("description"))
    if description:
        return description

    example_id = _normalized_string(prediction.get("id"))
    if example_id and example_id in reference_index.description_by_id:
        return reference_index.description_by_id[example_id]

    raise ValueError(
        "Prediction record must include a description or an id that exists in the evaluation split."
    )


def group_predictions_by_description(
    predictions: list[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    grouped_predictions: dict[str, list[dict[str, Any]]] = {}
    for prediction in predictions:
        description = _normalized_string(prediction.get("description"))
        if not description:
            raise ValueError("Prediction record is missing a description.")
        grouped_predictions.setdefault(description, []).append(prediction)
    return grouped_predictions
