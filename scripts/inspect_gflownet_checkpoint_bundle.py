#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path
from typing import Any
from zipfile import ZipFile


_MODEL_WEIGHT_FILENAMES = (
    "model.safetensors",
    "model.safetensors.index.json",
    "pytorch_model.bin",
    "pytorch_model.bin.index.json",
)
_TOKENIZER_FILENAMES = ("spiece.model", "tokenizer.json")
_ADAPTER_WEIGHT_FILENAMES = ("adapter_model.safetensors", "adapter_model.bin")


def _load_json_if_exists(path: Path) -> Any | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def _extract_iteration(payload: dict[str, Any] | None) -> int | None:
    if not isinstance(payload, dict) or "iteration" not in payload:
        return None
    try:
        return int(float(payload["iteration"]))
    except (TypeError, ValueError):
        return None


def _extract_learning_rate(payload: Any) -> float | None:
    candidates: list[Any] = []
    if isinstance(payload, dict):
        candidates.append(payload.get("learning_rate"))
        for section_name in ("gflownet", "training", "optimizer"):
            section = payload.get(section_name)
            if isinstance(section, dict):
                candidates.append(section.get("learning_rate"))
    for candidate in candidates:
        if candidate is None:
            continue
        try:
            return float(candidate)
        except (TypeError, ValueError):
            continue
    return None


def _last_iteration_metrics_jsonl_record(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    last_record: dict[str, Any] | None = None
    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict) and _extract_iteration(payload) is not None:
                last_record = payload
    return last_record


def _classify_root(root: Path) -> dict[str, Any]:
    config_exists = (root / "config.json").exists()
    tokenizer_config_exists = (root / "tokenizer_config.json").exists()
    model_weight_files = sorted(name for name in _MODEL_WEIGHT_FILENAMES if (root / name).exists())
    tokenizer_files = sorted(name for name in _TOKENIZER_FILENAMES if (root / name).exists())
    adapter_weight_files = sorted(
        name for name in _ADAPTER_WEIGHT_FILENAMES if (root / name).exists()
    )
    adapter_config = _load_json_if_exists(root / "adapter_config.json")
    training_config = _load_json_if_exists(root / "training_config.json")
    iteration_metrics_json = _load_json_if_exists(root / "iteration_metrics.json")
    metrics_json = _load_json_if_exists(root / "metrics.json")
    iteration_metrics_jsonl_record = _last_iteration_metrics_jsonl_record(
        root / "iteration_metrics.jsonl"
    )

    is_full_checkpoint = (
        config_exists
        and tokenizer_config_exists
        and bool(model_weight_files)
        and bool(tokenizer_files)
    )
    is_adapter_bundle = bool(adapter_config) and bool(adapter_weight_files)

    classification = "unknown"
    if is_full_checkpoint:
        classification = "full_checkpoint"
    elif is_adapter_bundle:
        classification = "adapter_bundle"

    missing_for_full_checkpoint: list[str] = []
    if not config_exists:
        missing_for_full_checkpoint.append("config.json")
    if not tokenizer_config_exists:
        missing_for_full_checkpoint.append("tokenizer_config.json")
    if not model_weight_files:
        missing_for_full_checkpoint.append("one of model.safetensors / pytorch_model.bin")
    if not tokenizer_files:
        missing_for_full_checkpoint.append("spiece.model or tokenizer.json")

    restart_iteration = (
        _extract_iteration(iteration_metrics_jsonl_record)
        if iteration_metrics_jsonl_record is not None
        else _extract_iteration(iteration_metrics_json)
    )
    learning_rate = None
    learning_rate_source = None
    for source_name, payload in (
        ("iteration_metrics.jsonl", iteration_metrics_jsonl_record),
        ("iteration_metrics.json", iteration_metrics_json),
        ("metrics.json", metrics_json),
        ("training_config.json", training_config),
    ):
        value = _extract_learning_rate(payload)
        if value is not None:
            learning_rate = value
            learning_rate_source = source_name
            break

    notes: list[str] = []
    if classification == "adapter_bundle":
        notes.append(
            "Archive is a LoRA adapter bundle, not a full merged checkpoint bundle."
        )
        notes.append(
            "prepare_gflownet_restart_checkpoint.py will reject it in extract_checkpoint_archive "
            "because config.json and full model weights are absent."
        )
        if not (root / "iteration_metrics.jsonl").exists() and (root / "iteration_metrics.json").exists():
            notes.append(
                "iteration_metrics.jsonl is missing, but iteration_metrics.json exists; "
                "the 06_v2 fallback path can synthesize iteration_metrics.jsonl after merge."
            )
    if classification == "full_checkpoint" and not (root / "iteration_metrics.jsonl").exists():
        notes.append(
            "Checkpoint looks loadable, but restart metadata will still fail unless iteration_metrics.jsonl "
            "is present or synthesized."
        )
    if isinstance(adapter_config, dict):
        base_model = str(adapter_config.get("base_model_name_or_path") or "").strip()
        if base_model:
            notes.append(f"adapter_config.base_model_name_or_path={base_model}")
            root_parent = root.parent.resolve(strict=False)
            resolved_base = Path(base_model).expanduser().resolve(strict=False)
            if resolved_base == root.resolve(strict=False):
                notes.append(
                    "Adapter base model path points to the extracted adapter/checkpoint directory itself; "
                    "a fallback base model must be chosen before merge."
                )
            elif resolved_base == root_parent / root.name:
                notes.append(
                    "Adapter base model path resolves to the restart target directory; "
                    "06_v2 should ignore it and merge onto the upstream/base checkpoint."
                )

    return {
        "root": str(root),
        "classification": classification,
        "is_full_checkpoint": is_full_checkpoint,
        "is_adapter_bundle": is_adapter_bundle,
        "missing_for_full_checkpoint": missing_for_full_checkpoint,
        "model_weight_files": model_weight_files,
        "tokenizer_files": tokenizer_files,
        "adapter_weight_files": adapter_weight_files,
        "adapter_base_model_name_or_path": (
            adapter_config.get("base_model_name_or_path")
            if isinstance(adapter_config, dict)
            else None
        ),
        "restart_iteration": restart_iteration,
        "learning_rate": learning_rate,
        "learning_rate_source": learning_rate_source,
        "notes": notes,
    }


def inspect_checkpoint_path(path_value: str | Path) -> dict[str, Any]:
    path = Path(path_value).expanduser().resolve(strict=False)
    if not path.exists():
        raise FileNotFoundError(f"Path does not exist: {path}")

    if path.is_dir():
        return {
            "input_path": str(path),
            "input_kind": "directory",
            "inspection": _classify_root(path),
        }

    if path.suffix.lower() != ".zip":
        raise ValueError(f"Unsupported input path (expected directory or .zip): {path}")

    with ZipFile(path, "r") as archive:
        archive_entries = archive.namelist()

    with tempfile.TemporaryDirectory(prefix="inspect_gflownet_bundle_") as temp_dir:
        extracted_root = Path(temp_dir) / "extracted"
        with ZipFile(path, "r") as archive:
            archive.extractall(extracted_root)

        first_level_dirs = sorted(
            child for child in extracted_root.iterdir() if child.is_dir()
        )
        candidate_roots = [extracted_root, *first_level_dirs]
        inspections = [_classify_root(root) for root in candidate_roots]
        selected = next(
            (payload for payload in inspections if payload["classification"] != "unknown"),
            inspections[0],
        )

        return {
            "input_path": str(path),
            "input_kind": "zip",
            "archive_entry_count": len(archive_entries),
            "archive_entries_preview": archive_entries[:50],
            "candidate_roots": [str(root) for root in candidate_roots],
            "selected_inspection": selected,
            "all_inspections": inspections,
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Inspect a GFlowNet checkpoint zip or extracted directory and explain whether it is "
            "a full checkpoint bundle or an adapter-only bundle."
        )
    )
    parser.add_argument("path", help="Path to a .zip bundle or extracted checkpoint directory.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        payload = inspect_checkpoint_path(args.path)
    except Exception as exc:
        print(
            json.dumps(
                {
                    "status": "failed",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "path": args.path,
                },
                indent=2,
            )
        )
        raise SystemExit(1) from exc

    print(json.dumps({"status": "ok", **payload}, indent=2))


if __name__ == "__main__":
    main()
