#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any
from zipfile import ZipFile

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from post_training.shared.config import looks_like_remote_model_identifier
from scripts.prepare_gflownet_restart_checkpoint import (
    prepare_restart_checkpoint,
    read_restart_checkpoint_metadata,
)
from src.checkpoint_bootstrap import checkpoint_artifact_is_ready, resolve_google_drive_file_id
from src.runtime_bootstrap import json_dumps
from src.train_dataset_bootstrap import download_google_drive_zip

RESTART_SOURCE_METADATA_FILENAME = "restart_source_metadata.json"
FLOW_HEAD_FILENAME = "flow_head.pt"


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _restart_source_metadata_path(checkpoint_dir: Path) -> Path:
    return checkpoint_dir / RESTART_SOURCE_METADATA_FILENAME


def _load_restart_source_metadata(checkpoint_dir: Path) -> dict[str, Any] | None:
    return _load_json(_restart_source_metadata_path(checkpoint_dir))


def _build_source_metadata(
    *,
    download_source: str,
    zip_path: Path,
    artifact_kind: str,
    resolved_merge_base: str | None,
    resolved_merge_base_source: str | None,
) -> dict[str, Any]:
    return {
        "download_source": download_source,
        "zip_path": str(zip_path),
        "artifact_kind": artifact_kind,
        "resolved_merge_base": resolved_merge_base,
        "resolved_merge_base_source": resolved_merge_base_source,
    }


def _save_restart_source_metadata(checkpoint_dir: Path, payload: dict[str, Any]) -> None:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    _write_json(_restart_source_metadata_path(checkpoint_dir), payload)


def _maybe_reuse_existing_checkpoint(
    checkpoint_dir: Path,
    *,
    download_source: str,
    fallback_learning_rate: float | None,
) -> dict[str, Any] | None:
    if not checkpoint_artifact_is_ready(checkpoint_dir):
        return None

    metadata = _load_restart_source_metadata(checkpoint_dir)
    if download_source:
        if metadata is None or str(metadata.get("download_source") or "").strip() != download_source:
            return None

    payload = read_restart_checkpoint_metadata(
        checkpoint_dir,
        fallback_learning_rate=fallback_learning_rate,
    ).to_dict()
    payload.update(
        {
            "status": "ready",
            "downloaded": False,
            "download_source": download_source,
            "zip_path": str(Path(metadata.get("zip_path")).expanduser()) if metadata and metadata.get("zip_path") else None,
            "extract_dir": str(checkpoint_dir),
            "artifact_kind": (
                str(metadata.get("artifact_kind"))
                if metadata and metadata.get("artifact_kind")
                else "prepared_checkpoint"
            ),
            "resolved_merge_base": metadata.get("resolved_merge_base") if metadata else None,
            "resolved_merge_base_source": metadata.get("resolved_merge_base_source") if metadata else None,
            "reused_existing": True,
            "restart_source_metadata_path": str(_restart_source_metadata_path(checkpoint_dir)),
        }
    )
    return payload


def _download_checkpoint_zip(
    download_source: str,
    zip_path: Path,
) -> bool:
    if not download_source:
        return False
    file_id = resolve_google_drive_file_id(download_source)
    if file_id is None:
        raise ValueError("Missing Google Drive checkpoint download source.")
    download_google_drive_zip(file_id, zip_path)
    return True


def _extract_restart_error(payload_text: str) -> str:
    try:
        payload = json.loads(payload_text)
    except json.JSONDecodeError:
        return payload_text
    if isinstance(payload, dict):
        return str(payload.get("error") or payload_text)
    return payload_text


def _archive_contains_adapter_bundle(zip_path: Path) -> bool:
    with ZipFile(zip_path, "r") as archive:
        return any(Path(name).name == "adapter_config.json" for name in archive.namelist())


def _extract_adapter_bundle_root(zip_path: Path, destination_root: Path) -> Path:
    staging_root = destination_root.parent / f".{destination_root.name}.extracting"
    if staging_root.exists():
        shutil.rmtree(staging_root)
    if destination_root.exists():
        shutil.rmtree(destination_root)
    staging_root.mkdir(parents=True, exist_ok=True)

    try:
        with ZipFile(zip_path, "r") as archive:
            archive.extractall(staging_root)
        adapter_config_candidates = sorted(staging_root.rglob("adapter_config.json"))
        if not adapter_config_candidates:
            raise RuntimeError("Adapter bundle is missing adapter_config.json.")
        adapter_root = adapter_config_candidates[0].parent
        shutil.move(str(adapter_root), str(destination_root))
    finally:
        if staging_root.exists():
            shutil.rmtree(staging_root, ignore_errors=True)

    return destination_root


def _adapter_dir_has_complete_tokenizer(adapter_dir: Path) -> bool:
    return (adapter_dir / "tokenizer_config.json").exists() and any(
        (adapter_dir / name).exists() for name in ("spiece.model", "tokenizer.json")
    )


def _path_is_within(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def _resolve_adapter_merge_base(
    adapter_config: dict[str, Any],
    *,
    checkpoint_dir: Path,
    upstream_checkpoint: str | None,
    default_base_model: str | None,
) -> tuple[str, bool, str]:
    adapter_base_text = str(adapter_config.get("base_model_name_or_path") or "").strip()
    invalid_base_reasons: list[str] = []

    if adapter_base_text:
        if looks_like_remote_model_identifier(adapter_base_text):
            return adapter_base_text, False, "adapter_config.base_model_name_or_path"

        resolved_adapter_base_path = Path(adapter_base_text).expanduser().resolve(strict=False)
        if resolved_adapter_base_path == checkpoint_dir.resolve(strict=False):
            invalid_base_reasons.append(
                "adapter_config.base_model_name_or_path points to checkpoint_dir"
            )
        elif (
            _path_is_within(resolved_adapter_base_path, checkpoint_dir.parent.resolve(strict=False))
            and not resolved_adapter_base_path.exists()
        ):
            invalid_base_reasons.append(
                "adapter_config.base_model_name_or_path points inside checkpoint_dir.parent but does not exist"
            )
        elif not resolved_adapter_base_path.exists():
            invalid_base_reasons.append(
                "adapter_config.base_model_name_or_path is a local path that does not exist"
            )
        elif not resolved_adapter_base_path.is_dir():
            invalid_base_reasons.append(
                "adapter_config.base_model_name_or_path is not a directory"
            )
        else:
            return (
                str(resolved_adapter_base_path),
                True,
                "adapter_config.base_model_name_or_path",
            )

    upstream_text = str(upstream_checkpoint or "").strip()
    if upstream_text:
        if looks_like_remote_model_identifier(upstream_text):
            return upstream_text, False, "upstream_checkpoint"
        resolved_upstream_path = Path(upstream_text).expanduser().resolve(strict=False)
        if resolved_upstream_path.exists() and resolved_upstream_path.is_dir():
            return str(resolved_upstream_path), True, "upstream_checkpoint"
        invalid_base_reasons.append("upstream_checkpoint is not a usable local directory")

    default_base_text = str(default_base_model or "").strip()
    if default_base_text:
        if looks_like_remote_model_identifier(default_base_text):
            return default_base_text, False, "default_base_model"
        resolved_default_path = Path(default_base_text).expanduser().resolve(strict=False)
        if resolved_default_path.exists() and resolved_default_path.is_dir():
            return str(resolved_default_path), True, "default_base_model"
        invalid_base_reasons.append("default_base_model is not a usable local directory or remote id")

    invalid_reason_text = "; ".join(invalid_base_reasons) if invalid_base_reasons else (
        "adapter_config.base_model_name_or_path is missing"
    )
    raise RuntimeError(
        "Adapter merge base resolution failed. Neither adapter_config.base_model_name_or_path, "
        "upstream_checkpoint, nor default_base_model was usable. "
        f"{invalid_reason_text}"
    )


def _copy_restart_artifacts(adapter_dir: Path, checkpoint_dir: Path) -> None:
    if not (adapter_dir / FLOW_HEAD_FILENAME).exists():
        raise RuntimeError(f"Adapter bundle is missing required {FLOW_HEAD_FILENAME}.")

    for artifact_name in (
        FLOW_HEAD_FILENAME,
        "iteration_metrics.json",
        "iteration_metrics.jsonl",
        "metrics.json",
        "training_config.json",
        "sampled_trajectories.jsonl",
    ):
        source_path = adapter_dir / artifact_name
        if source_path.exists():
            shutil.copy2(source_path, checkpoint_dir / artifact_name)

    if (
        not (checkpoint_dir / "iteration_metrics.jsonl").exists()
        and (checkpoint_dir / "iteration_metrics.json").exists()
    ):
        iteration_payload = json.loads(
            (checkpoint_dir / "iteration_metrics.json").read_text(encoding="utf-8")
        )
        (checkpoint_dir / "iteration_metrics.jsonl").write_text(
            json.dumps(iteration_payload) + "\n",
            encoding="utf-8",
        )


def _load_merge_dependencies():
    from peft import PeftModel
    from transformers import AutoTokenizer, T5ForConditionalGeneration

    return AutoTokenizer, PeftModel, T5ForConditionalGeneration


def _prepare_adapter_bundle_checkpoint(
    *,
    zip_path: Path,
    checkpoint_dir: Path,
    upstream_checkpoint: str | None,
    default_base_model: str | None,
    fallback_learning_rate: float | None,
    download_source: str,
) -> dict[str, Any]:
    AutoTokenizer, PeftModel, T5ForConditionalGeneration = _load_merge_dependencies()

    adapter_dir = checkpoint_dir.parent / f"{checkpoint_dir.name}_adapter"
    _extract_adapter_bundle_root(zip_path, adapter_dir)

    adapter_config_path = adapter_dir / "adapter_config.json"
    adapter_config = json.loads(adapter_config_path.read_text(encoding="utf-8"))
    (
        resolved_merge_base,
        resolved_merge_base_is_local,
        resolved_merge_base_source,
    ) = _resolve_adapter_merge_base(
        adapter_config,
        checkpoint_dir=checkpoint_dir,
        upstream_checkpoint=upstream_checkpoint,
        default_base_model=default_base_model,
    )

    base_model = T5ForConditionalGeneration.from_pretrained(
        resolved_merge_base,
        local_files_only=resolved_merge_base_is_local,
    )
    merged_model = PeftModel.from_pretrained(base_model, str(adapter_dir)).merge_and_unload()

    checkpoint_dir.parent.mkdir(parents=True, exist_ok=True)
    if checkpoint_dir.exists():
        shutil.rmtree(checkpoint_dir)
    merged_model.save_pretrained(checkpoint_dir)

    tokenizer_source = (
        adapter_dir if _adapter_dir_has_complete_tokenizer(adapter_dir) else resolved_merge_base
    )
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_source,
        use_fast=True,
        local_files_only=(tokenizer_source == adapter_dir or resolved_merge_base_is_local),
    )
    tokenizer.save_pretrained(checkpoint_dir)

    _copy_restart_artifacts(adapter_dir, checkpoint_dir)
    metadata = read_restart_checkpoint_metadata(
        checkpoint_dir,
        fallback_learning_rate=fallback_learning_rate,
    ).to_dict()
    metadata.update(
        {
            "status": "ready",
            "downloaded": False,
            "download_source": download_source,
            "zip_path": str(zip_path),
            "extract_dir": str(checkpoint_dir),
            "artifact_kind": "adapter_bundle",
            "resolved_merge_base": resolved_merge_base,
            "resolved_merge_base_source": resolved_merge_base_source,
            "reused_existing": False,
            "restart_source_metadata_path": str(_restart_source_metadata_path(checkpoint_dir)),
        }
    )
    _save_restart_source_metadata(
        checkpoint_dir,
        _build_source_metadata(
            download_source=download_source,
            zip_path=zip_path,
            artifact_kind="adapter_bundle",
            resolved_merge_base=resolved_merge_base,
            resolved_merge_base_source=resolved_merge_base_source,
        ),
    )
    return metadata


def prepare_uploaded_checkpoint(
    *,
    download_source: str | None,
    zip_path: str | Path,
    extract_dir: str | Path,
    upstream_checkpoint: str | None = None,
    default_base_model: str | None = None,
    skip_existing: bool = False,
    fallback_learning_rate: float | None = None,
) -> dict[str, Any]:
    source = str(download_source or "").strip()
    zip_destination = Path(zip_path).expanduser().resolve()
    checkpoint_dir = Path(extract_dir).expanduser().resolve()

    if skip_existing:
        reused_payload = _maybe_reuse_existing_checkpoint(
            checkpoint_dir,
            download_source=source,
            fallback_learning_rate=fallback_learning_rate,
        )
        if reused_payload is not None:
            return reused_payload

    downloaded = False
    if source:
        downloaded = _download_checkpoint_zip(source, zip_destination)

    try:
        payload = prepare_restart_checkpoint(
            download_source=None,
            zip_path=zip_destination,
            extract_dir=checkpoint_dir,
            skip_existing=False,
            fallback_learning_rate=fallback_learning_rate,
        )
    except Exception as exc:
        if not zip_destination.exists():
            raise
        if not _archive_contains_adapter_bundle(zip_destination):
            raise RuntimeError(_extract_restart_error(str(exc))) from exc
        payload = _prepare_adapter_bundle_checkpoint(
            zip_path=zip_destination,
            checkpoint_dir=checkpoint_dir,
            upstream_checkpoint=upstream_checkpoint,
            default_base_model=default_base_model,
            fallback_learning_rate=fallback_learning_rate,
            download_source=source,
        )
        payload["downloaded"] = downloaded
        return payload

    payload.update(
        {
            "artifact_kind": "full_checkpoint",
            "resolved_merge_base": None,
            "resolved_merge_base_source": None,
            "reused_existing": False,
            "restart_source_metadata_path": str(_restart_source_metadata_path(checkpoint_dir)),
        }
    )
    _save_restart_source_metadata(
        checkpoint_dir,
        _build_source_metadata(
            download_source=source,
            zip_path=zip_destination,
            artifact_kind="full_checkpoint",
            resolved_merge_base=None,
            resolved_merge_base_source=None,
        ),
    )
    payload["downloaded"] = downloaded
    payload["download_source"] = source
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare a notebook-ready GFlowNet restart checkpoint from either a full checkpoint "
            "bundle or an adapter-only bundle."
        )
    )
    parser.add_argument(
        "--download-source",
        "--source",
        dest="download_source",
        default=None,
        help="Google Drive file id or share URL for the zipped checkpoint or adapter bundle.",
    )
    parser.add_argument("--zip-path", required=True, help="Path for the downloaded or local zip archive.")
    parser.add_argument("--extract-dir", required=True, help="Prepared checkpoint output directory.")
    parser.add_argument(
        "--upstream-checkpoint",
        default=None,
        help="Preferred local directory or remote id for the base checkpoint when merging adapters.",
    )
    parser.add_argument(
        "--default-base-model",
        default=None,
        help="Fallback local directory or remote id for adapter merge when upstream-checkpoint is unusable.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Reuse an already-prepared checkpoint dir only when it matches the provided download source.",
    )
    parser.add_argument(
        "--fallback-learning-rate",
        type=float,
        default=None,
        help="Learning rate to use only when checkpoint metadata has no learning_rate.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        payload = prepare_uploaded_checkpoint(
            download_source=args.download_source,
            zip_path=args.zip_path,
            extract_dir=args.extract_dir,
            upstream_checkpoint=args.upstream_checkpoint,
            default_base_model=args.default_base_model,
            skip_existing=args.skip_existing,
            fallback_learning_rate=args.fallback_learning_rate,
        )
    except Exception as exc:
        print(
            json_dumps(
                {
                    "status": "failed",
                    "error": str(exc),
                    "zip_path": str(Path(args.zip_path).expanduser()),
                    "extract_dir": str(Path(args.extract_dir).expanduser()),
                }
            )
        )
        raise SystemExit(1) from exc

    print(json_dumps(payload))


if __name__ == "__main__":
    main()
