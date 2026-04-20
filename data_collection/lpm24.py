from __future__ import annotations

import random
import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Sequence

from datasets import Dataset, DatasetDict, load_dataset

from molecules.datasets.multi import build_multi_molecule_processed_record
from src.io_utils import ensure_dir, read_jsonl, write_json, write_jsonl
from src.prompting import normalize_free_text


DESCRIPTION_KEYS = ("description", "text", "caption", "prompt")
SMILES_KEYS = ("SMILES", "smiles", "molecule_smiles", "molecule")
SELFIES_SYMBOL_PATTERN = re.compile(r"\[[^\]]+\]")


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


def count_selfies_symbols(selfies_text: str) -> int:
    return len(SELFIES_SYMBOL_PATTERN.findall(str(selfies_text)))


def estimate_staged_target_symbol_count(selfies_list: Sequence[str]) -> int:
    stage_lengths = [count_selfies_symbols(item) for item in selfies_list if str(item).strip()]
    return sum(stage_lengths) + max(0, len(stage_lengths) - 1)


def _normalize_optional_limit(limit: int | None) -> int | None:
    if limit is None:
        return None
    normalized = int(limit)
    if normalized <= 0:
        return None
    return normalized


def _record_symbol_metrics(record: dict[str, Any]) -> dict[str, int]:
    target_selfies_list = [str(item) for item in record.get("target_selfies_list", []) if str(item).strip()]
    stage_lengths = [count_selfies_symbols(item) for item in target_selfies_list]
    return {
        "target_count": len(target_selfies_list),
        "target_symbol_count": sum(stage_lengths) + max(0, len(stage_lengths) - 1),
        "max_stage_symbol_count": max(stage_lengths, default=0),
    }


def _filter_training_ready_records(
    records: Sequence[dict[str, Any]],
    *,
    split: str,
    max_target_symbols: int | None,
    max_stage_symbols: int | None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    kept_records: list[dict[str, Any]] = []
    removed_counts: Counter[str] = Counter()
    removed_preview: list[dict[str, Any]] = []

    for record in records:
        metrics = _record_symbol_metrics(record)
        exceeds_target_limit = (
            max_target_symbols is not None and metrics["target_symbol_count"] > max_target_symbols
        )
        exceeds_stage_limit = (
            max_stage_symbols is not None and metrics["max_stage_symbol_count"] > max_stage_symbols
        )

        if exceeds_target_limit or exceeds_stage_limit:
            if exceeds_target_limit and exceeds_stage_limit:
                reason = "target_and_stage_symbol_limits"
            elif exceeds_target_limit:
                reason = "target_symbol_limit"
            else:
                reason = "stage_symbol_limit"
            removed_counts[reason] += 1
            if len(removed_preview) < 10:
                removed_preview.append(
                    {
                        "id": str(record.get("id", "")),
                        "description": str(record.get("description", ""))[:160],
                        "reason": reason,
                        **metrics,
                    }
                )
            continue

        kept_records.append(record)

    summary = {
        "split": split,
        "source_records": len(records),
        "kept_records": len(kept_records),
        "removed_records": len(records) - len(kept_records),
        "removed_by_reason": dict(sorted(removed_counts.items())),
        "removed_preview": removed_preview,
        "max_target_symbols": max_target_symbols,
        "max_stage_symbols": max_stage_symbols,
    }
    return kept_records, summary


def _build_validation_count(record_count: int, validation_fraction: float) -> int:
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be between 0 and 1.")
    if record_count < 2:
        raise ValueError("At least two filtered training records are required to create train/validation splits.")

    requested = int(record_count * validation_fraction)
    return min(record_count - 1, max(1, requested))


def export_lpm24_training_splits(
    *,
    input_dir: str | Path,
    output_dir: str | Path | None = None,
    validation_fraction: float = 0.05,
    seed: int = 42,
    max_target_symbols: int | None = 1024,
    max_stage_symbols: int | None = 192,
) -> dict[str, Any]:
    input_path = Path(input_dir).expanduser().resolve()
    processed_dir = input_path / "processed"
    output_path = Path(output_dir).expanduser().resolve() if output_dir else (input_path / "grouped_splits")
    output_path = ensure_dir(output_path)

    train_source_path = processed_dir / "train_multimol.jsonl"
    test_source_path = processed_dir / "test_multimol.jsonl"
    train_source_records = read_jsonl(train_source_path)
    test_source_records = read_jsonl(test_source_path)

    normalized_target_limit = _normalize_optional_limit(max_target_symbols)
    normalized_stage_limit = _normalize_optional_limit(max_stage_symbols)

    filtered_train_records, train_filter_summary = _filter_training_ready_records(
        train_source_records,
        split="train_source",
        max_target_symbols=normalized_target_limit,
        max_stage_symbols=normalized_stage_limit,
    )
    filtered_test_records, test_filter_summary = _filter_training_ready_records(
        test_source_records,
        split="test",
        max_target_symbols=normalized_target_limit,
        max_stage_symbols=normalized_stage_limit,
    )

    shuffled_train_records = list(filtered_train_records)
    random.Random(int(seed)).shuffle(shuffled_train_records)
    validation_count = _build_validation_count(
        len(shuffled_train_records),
        validation_fraction=float(validation_fraction),
    )
    validation_records = shuffled_train_records[:validation_count]
    train_records = shuffled_train_records[validation_count:]

    train_output_path = output_path / "train_multimol.jsonl"
    validation_output_path = output_path / "validation_multimol.jsonl"
    test_output_path = output_path / "test_multimol.jsonl"
    summary_path = output_path / "summary.json"

    write_jsonl(train_output_path, train_records)
    write_jsonl(validation_output_path, validation_records)
    write_jsonl(test_output_path, filtered_test_records)

    summary = {
        "input_dir": str(input_path),
        "output_dir": str(output_path),
        "validation_fraction": float(validation_fraction),
        "seed": int(seed),
        "limits": {
            "max_target_symbols": normalized_target_limit,
            "max_stage_symbols": normalized_stage_limit,
        },
        "files": {
            "train_source": str(train_source_path),
            "test_source": str(test_source_path),
            "train_multimol": str(train_output_path),
            "validation_multimol": str(validation_output_path),
            "test_multimol": str(test_output_path),
            "summary": str(summary_path),
        },
        "counts": {
            "train_source_records": len(train_source_records),
            "train_records": len(train_records),
            "validation_records": len(validation_records),
            "test_source_records": len(test_source_records),
            "test_records": len(filtered_test_records),
        },
        "filtering": {
            "train_source": train_filter_summary,
            "test": test_filter_summary,
        },
    }
    write_json(summary_path, summary)
    return summary


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
