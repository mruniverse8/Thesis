from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

from datasets import Dataset, DatasetDict, load_dataset

from molecules.datasets.multi import build_multi_molecule_processed_record
from src.io_utils import ensure_dir, write_json, write_jsonl
from src.prompting import normalize_free_text


DESCRIPTION_KEYS = ("description", "text", "caption", "prompt")
SMILES_KEYS = ("SMILES", "smiles", "molecule_smiles", "molecule")


def _load_primary_split(
    dataset_name: str,
    preferred_split: str | None = None,
) -> tuple[Dataset, str]:
    loaded = load_dataset(dataset_name)
    if isinstance(loaded, Dataset):
        return loaded, preferred_split or "train"
    if not isinstance(loaded, DatasetDict):
        raise TypeError(f"Unsupported dataset container type: {type(loaded)!r}")

    candidate_splits = [preferred_split, "train", "test", "validation"]
    for split_name in candidate_splits:
        if split_name and split_name in loaded:
            return loaded[split_name], split_name

    first_split = next(iter(loaded.keys()))
    return loaded[first_split], first_split


def _extract_description(raw_record: dict[str, Any]) -> str:
    for key in DESCRIPTION_KEYS:
        value = raw_record.get(key)
        if value is None:
            continue
        description = normalize_free_text(str(value))
        if description:
            return description
    raise ValueError("Missing description field in LPM-24 record.")


def _extract_smiles(raw_record: dict[str, Any]) -> str:
    for key in SMILES_KEYS:
        value = raw_record.get(key)
        if value is None:
            continue
        smiles = str(value).strip()
        if smiles:
            return smiles
    raise ValueError("Missing SMILES field in LPM-24 record.")


def build_lpm24_grouped_records(
    raw_records: Iterable[dict[str, Any]],
    *,
    split: str,
    max_molecules_per_example: int = 256,
) -> list[dict[str, Any]]:
    grouped_examples: dict[str, dict[str, Any]] = {}

    for raw_record in raw_records:
        try:
            description = _extract_description(raw_record)
            smiles = _extract_smiles(raw_record)
        except ValueError:
            continue

        grouped_record = grouped_examples.setdefault(
            description,
            {
                "id": f"{split}-{len(grouped_examples):06d}",
                "description": description,
                "targets": [],
            },
        )
        grouped_record["targets"].append({"smiles": smiles})

    processed_records: list[dict[str, Any]] = []
    for index, grouped_record in enumerate(grouped_examples.values()):
        try:
            processed = build_multi_molecule_processed_record(
                grouped_record,
                split=split,
                index=index,
                max_molecules_per_sequence=max_molecules_per_example,
                convert_missing_selfies_from_smiles=True,
            )
        except ValueError as exc:
            if str(exc) == "Missing valid target molecules":
                continue
            raise
        processed_records.append(processed)
    return processed_records


def download_and_preprocess_lpm24(
    *,
    train_dataset_name: str,
    eval_dataset_name: str | None = None,
    test_dataset_name: str | None = None,
    output_dir: str | Path,
    test_eval_description_limit: int = 1000,
    max_train_descriptions: int | None = None,
    max_test_descriptions: int | None = None,
    max_molecules_per_example: int = 256,
) -> dict[str, Any]:
    output_path = Path(output_dir).expanduser().resolve()
    raw_dir = ensure_dir(output_path / "raw")
    processed_dir = ensure_dir(output_path / "processed")

    resolved_eval_dataset_name = eval_dataset_name or test_dataset_name
    if not resolved_eval_dataset_name:
        raise ValueError("An LPM-24 evaluation dataset name is required.")

    train_dataset, train_split_name = _load_primary_split(train_dataset_name, preferred_split="train")
    eval_dataset, eval_split_name = _load_primary_split(resolved_eval_dataset_name, preferred_split="train")

    train_grouped = build_lpm24_grouped_records(
        train_dataset,
        split="train",
        max_molecules_per_example=max_molecules_per_example,
    )
    test_grouped = build_lpm24_grouped_records(
        eval_dataset,
        split="test",
        max_molecules_per_example=max_molecules_per_example,
    )

    if max_train_descriptions is not None:
        train_grouped = train_grouped[: int(max_train_descriptions)]
    if max_test_descriptions is not None:
        test_grouped = test_grouped[: int(max_test_descriptions)]

    test_eval_grouped = test_grouped[: int(test_eval_description_limit)]

    train_path = processed_dir / "train_multimol.jsonl"
    test_path = processed_dir / "test_multimol.jsonl"
    eval_path = processed_dir / "test_eval_first_1000_multimol.jsonl"

    write_jsonl(train_path, train_grouped)
    write_jsonl(test_path, test_grouped)
    write_jsonl(eval_path, test_eval_grouped)

    summary = {
        "train_dataset_name": train_dataset_name,
        "train_split_name": train_split_name,
        "eval_dataset_name": resolved_eval_dataset_name,
        "eval_split_name": eval_split_name,
        "output_dir": str(output_path),
        "max_molecules_per_example": max_molecules_per_example,
        "test_eval_description_limit": test_eval_description_limit,
        "files": {
            "train_multimol": str(train_path),
            "test_multimol": str(test_path),
            "test_eval_first_1000_multimol": str(eval_path),
            "summary": str(raw_dir / "download_metadata.json"),
        },
        "counts": {
            "train_descriptions": len(train_grouped),
            "test_descriptions": len(test_grouped),
            "test_eval_descriptions": len(test_eval_grouped),
        },
    }
    write_json(raw_dir / "download_metadata.json", summary)
    return summary
