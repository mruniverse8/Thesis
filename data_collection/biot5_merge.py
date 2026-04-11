from __future__ import annotations

import json
from collections import Counter
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from src.io_utils import dump_yaml, ensure_dir, load_yaml, read_jsonl, write_json, write_jsonl


@dataclass(frozen=True)
class CollectionPartBundle:
    staging_dir: Path
    derived_train_file: Path
    resolved_config: dict[str, Any]
    summary: dict[str, Any]
    inputs: list[dict[str, Any]]
    raw_candidates: list[dict[str, Any]]
    candidate_assessments: list[dict[str, Any]]
    accepted_grouped: list[dict[str, Any]]
    derived_train_records: list[dict[str, Any]]


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return payload


def _load_part_bundle(staging_dir: str | Path, derived_train_file: str | Path) -> CollectionPartBundle:
    staging_path = Path(staging_dir)
    derived_path = Path(derived_train_file)
    required_paths = {
        "resolved_config": staging_path / "resolved_config.yaml",
        "summary": staging_path / "summary.json",
        "inputs": staging_path / "inputs.jsonl",
        "raw_candidates": staging_path / "raw_candidates.jsonl",
        "candidate_assessments": staging_path / "candidate_assessments.jsonl",
        "accepted_grouped": staging_path / "accepted_grouped.jsonl",
        "derived_train_file": derived_path,
    }
    missing_paths = {
        name: str(path)
        for name, path in required_paths.items()
        if not path.exists()
    }
    if missing_paths:
        raise FileNotFoundError(json.dumps({"missing_paths": missing_paths}, indent=2))

    return CollectionPartBundle(
        staging_dir=staging_path,
        derived_train_file=derived_path,
        resolved_config=load_yaml(required_paths["resolved_config"]),
        summary=_read_json(required_paths["summary"]),
        inputs=read_jsonl(required_paths["inputs"]),
        raw_candidates=read_jsonl(required_paths["raw_candidates"]),
        candidate_assessments=read_jsonl(required_paths["candidate_assessments"]),
        accepted_grouped=read_jsonl(required_paths["accepted_grouped"]),
        derived_train_records=read_jsonl(required_paths["derived_train_file"]),
    )


def _normalize_config_for_merge(config: dict[str, Any]) -> dict[str, Any]:
    normalized = deepcopy(config)
    normalized.setdefault("data", {})
    normalized["data"].pop("staging_dir", None)
    normalized["data"].pop("derived_train_file", None)
    runtime = normalized.setdefault("runtime", {})
    runtime.pop("part_index", None)
    return normalized


def _partition_info(bundle: CollectionPartBundle) -> dict[str, int]:
    summary_partition = bundle.summary.get("partition", {})
    runtime_config = bundle.resolved_config.get("runtime", {})
    num_parts = int(summary_partition.get("num_parts", runtime_config.get("num_parts", 1)))
    part_index = int(summary_partition.get("part_index", runtime_config.get("part_index", 1)))
    return {
        "num_parts": num_parts,
        "part_index": part_index,
        "part_description_count": int(
            summary_partition.get("part_description_count", len(bundle.inputs))
        ),
        "part_start_index": int(summary_partition.get("part_start_index", 0)),
        "part_end_index_exclusive": int(
            summary_partition.get("part_end_index_exclusive", len(bundle.inputs))
        ),
        "total_selected_descriptions_before_partition": int(
            summary_partition.get(
                "total_selected_descriptions_before_partition",
                bundle.summary.get("selected_descriptions", len(bundle.inputs)),
            )
        ),
    }


def _validate_part_bundles(bundles: Sequence[CollectionPartBundle]) -> list[CollectionPartBundle]:
    if not bundles:
        raise ValueError("Expected at least one collection part bundle to merge.")

    base_config = _normalize_config_for_merge(bundles[0].resolved_config)
    base_partition = _partition_info(bundles[0])
    expected_num_parts = base_partition["num_parts"]
    part_indices: set[int] = set()
    expected_total = base_partition["total_selected_descriptions_before_partition"]

    for bundle in bundles:
        normalized_config = _normalize_config_for_merge(bundle.resolved_config)
        if normalized_config != base_config:
            raise ValueError(
                "Collection part configs do not match after normalizing part-specific output paths."
            )

        partition = _partition_info(bundle)
        if partition["num_parts"] != expected_num_parts:
            raise ValueError(
                f"Expected num_parts={expected_num_parts}, got {partition['num_parts']}"
            )
        if partition["total_selected_descriptions_before_partition"] != expected_total:
            raise ValueError(
                "Collection parts report different total_selected_descriptions_before_partition values."
            )
        if partition["part_description_count"] != len(bundle.inputs):
            raise ValueError(
                "Collection part summary partition metadata does not match inputs.jsonl record count."
            )
        if partition["part_index"] in part_indices:
            raise ValueError(f"Duplicate part_index detected: {partition['part_index']}")
        part_indices.add(partition["part_index"])

    expected_indices = set(range(1, expected_num_parts + 1))
    if part_indices != expected_indices:
        raise ValueError(
            f"Expected complete part indices {sorted(expected_indices)}, got {sorted(part_indices)}"
        )

    return sorted(bundles, key=lambda bundle: _partition_info(bundle)["part_index"])


def _assert_unique_ids(records: Sequence[dict[str, Any]], *, key: str, context: str) -> None:
    seen: set[str] = set()
    duplicates: set[str] = set()
    for record in records:
        value = str(record.get(key) or "")
        if not value:
            continue
        if value in seen:
            duplicates.add(value)
        seen.add(value)
    if duplicates:
        preview = sorted(duplicates)[:5]
        raise ValueError(f"Duplicate {key} values in {context}: {preview}")


def merge_biot5_collection_parts(
    *,
    part_staging_dirs: Sequence[str | Path],
    part_derived_train_files: Sequence[str | Path],
    output_dir: str | Path,
    merged_derived_train_file: str | Path,
) -> dict[str, Any]:
    if len(part_staging_dirs) != len(part_derived_train_files):
        raise ValueError("part_staging_dirs and part_derived_train_files must have the same length.")

    bundles = _validate_part_bundles(
        [
            _load_part_bundle(staging_dir, derived_train_file)
            for staging_dir, derived_train_file in zip(
                part_staging_dirs,
                part_derived_train_files,
                strict=True,
            )
        ]
    )

    merged_inputs: list[dict[str, Any]] = []
    merged_raw_candidates: list[dict[str, Any]] = []
    merged_candidate_assessments: list[dict[str, Any]] = []
    merged_accepted_grouped: list[dict[str, Any]] = []
    merged_derived_train_records: list[dict[str, Any]] = []
    rejections_by_reason: Counter[str] = Counter()
    part_summaries: list[dict[str, Any]] = []

    for bundle in bundles:
        partition = _partition_info(bundle)
        merged_inputs.extend(bundle.inputs)
        merged_raw_candidates.extend(bundle.raw_candidates)
        merged_candidate_assessments.extend(bundle.candidate_assessments)
        merged_accepted_grouped.extend(bundle.accepted_grouped)
        merged_derived_train_records.extend(bundle.derived_train_records)
        rejections_by_reason.update(bundle.summary.get("rejections_by_reason", {}))
        part_summaries.append(
            {
                "part_index": partition["part_index"],
                "num_parts": partition["num_parts"],
                "staging_dir": str(bundle.staging_dir),
                "derived_train_file": str(bundle.derived_train_file),
                "selected_descriptions": int(bundle.summary.get("selected_descriptions", len(bundle.inputs))),
                "descriptions_with_accepted_molecules": int(
                    bundle.summary.get(
                        "descriptions_with_accepted_molecules",
                        len(bundle.accepted_grouped),
                    )
                ),
                "raw_candidates": int(bundle.summary.get("raw_candidates", len(bundle.raw_candidates))),
                "derived_examples": int(
                    bundle.summary.get("derived_examples", len(bundle.derived_train_records))
                ),
                "partition": partition,
            }
        )

    _assert_unique_ids(merged_inputs, key="id", context="merged inputs")
    _assert_unique_ids(merged_raw_candidates, key="id", context="merged raw candidates")
    _assert_unique_ids(
        merged_candidate_assessments,
        key="id",
        context="merged candidate assessments",
    )
    _assert_unique_ids(merged_accepted_grouped, key="id", context="merged accepted grouped")
    _assert_unique_ids(merged_derived_train_records, key="id", context="merged derived train records")

    base_summary = bundles[0].summary
    base_config = deepcopy(bundles[0].resolved_config)
    output_path = ensure_dir(output_dir)
    merged_train_path = Path(merged_derived_train_file)

    base_config.setdefault("data", {})
    base_config["data"]["staging_dir"] = str(output_path)
    base_config["data"]["derived_train_file"] = str(merged_train_path)
    base_config.setdefault("runtime", {})
    base_config["runtime"]["num_parts"] = len(bundles)
    base_config["runtime"].pop("part_index", None)
    base_config["runtime"]["merged_part_indices"] = [
        item["part_index"] for item in part_summaries
    ]

    dump_yaml(output_path / "resolved_config.yaml", base_config)
    write_jsonl(output_path / "inputs.jsonl", merged_inputs)
    write_jsonl(output_path / "raw_candidates.jsonl", merged_raw_candidates)
    write_jsonl(output_path / "candidate_assessments.jsonl", merged_candidate_assessments)
    write_jsonl(output_path / "accepted_grouped.jsonl", merged_accepted_grouped)
    write_jsonl(merged_train_path, merged_derived_train_records)

    merged_summary = {
        "train_file": str(base_summary["train_file"]),
        "staging_dir": str(output_path),
        "derived_train_file": str(merged_train_path),
        "seed": int(base_summary["seed"]),
        "generation_strategy": str(base_summary["generation_strategy"]),
        "selected_descriptions": len(merged_inputs),
        "descriptions_with_accepted_molecules": len(merged_accepted_grouped),
        "raw_candidates": len(merged_raw_candidates),
        "accepted_candidates": sum(
            int(record["accepted_candidate_count"]) for record in merged_accepted_grouped
        ),
        "derived_examples": len(merged_derived_train_records),
        "target_molecules_per_description": int(base_summary["target_molecules_per_description"]),
        "max_molecules_per_example": int(base_summary["max_molecules_per_example"]),
        "rejections_by_reason": dict(sorted(rejections_by_reason.items())),
        "partition": {
            "num_parts": len(bundles),
            "merged_part_indices": [item["part_index"] for item in part_summaries],
            "total_selected_descriptions_before_partition": _partition_info(bundles[0])[
                "total_selected_descriptions_before_partition"
            ],
        },
        "parts": part_summaries,
        "files": {
            "inputs": str(output_path / "inputs.jsonl"),
            "raw_candidates": str(output_path / "raw_candidates.jsonl"),
            "candidate_assessments": str(output_path / "candidate_assessments.jsonl"),
            "accepted_grouped": str(output_path / "accepted_grouped.jsonl"),
            "derived_train_file": str(merged_train_path),
            "resolved_config": str(output_path / "resolved_config.yaml"),
            "summary": str(output_path / "summary.json"),
        },
    }
    write_json(output_path / "summary.json", merged_summary)
    return merged_summary


__all__ = ["merge_biot5_collection_parts"]
