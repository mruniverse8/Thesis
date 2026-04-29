#!/usr/bin/env python3

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.checkpoint_bootstrap import (
    checkpoint_artifact_is_ready,
    extract_checkpoint_archive,
    resolve_google_drive_file_id,
)
from src.runtime_bootstrap import json_dumps
from src.train_dataset_bootstrap import download_google_drive_zip


@dataclass(frozen=True)
class RestartCheckpointMetadata:
    checkpoint_dir: str
    restart_iteration: int
    next_iteration: int
    learning_rate: float
    iteration_source_path: str
    learning_rate_source_path: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _extract_iteration(payload: dict[str, Any]) -> int | None:
    if "iteration" not in payload:
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
    last_record: dict[str, Any] | None = None
    if not path.exists():
        return None

    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(record, dict) and _extract_iteration(record) is not None:
                last_record = record
    return last_record


def read_restart_checkpoint_metadata(
    checkpoint_dir: str | Path,
    *,
    fallback_learning_rate: float | None = None,
) -> RestartCheckpointMetadata:
    resolved_checkpoint_dir = Path(checkpoint_dir).expanduser().resolve()
    if not checkpoint_artifact_is_ready(resolved_checkpoint_dir):
        raise RuntimeError(f"Checkpoint bundle is incomplete: {resolved_checkpoint_dir}")

    iteration_metrics_jsonl = resolved_checkpoint_dir / "iteration_metrics.jsonl"
    last_jsonl_record = _last_iteration_metrics_jsonl_record(iteration_metrics_jsonl)
    if last_jsonl_record is None:
        raise ValueError(
            "No restart iteration found. Expected at least one valid record with "
            f"'iteration' in {iteration_metrics_jsonl}."
        )

    restart_iteration = _extract_iteration(last_jsonl_record)
    if restart_iteration is None:
        raise ValueError(f"No restart iteration found in {iteration_metrics_jsonl}.")

    learning_rate = _extract_learning_rate(last_jsonl_record)
    learning_rate_source_path: Path | None = (
        iteration_metrics_jsonl if learning_rate is not None else None
    )
    for relative_path in (
        Path("iteration_metrics.json"),
        Path("metrics.json"),
        Path("training_config.json"),
    ):
        if learning_rate is not None:
            break
        path = resolved_checkpoint_dir / relative_path
        if not path.exists():
            continue
        try:
            payload = _load_json(path)
        except json.JSONDecodeError:
            continue
        learning_rate = _extract_learning_rate(payload)
        if learning_rate is not None:
            learning_rate_source_path = path

    if learning_rate is None and fallback_learning_rate is not None:
        learning_rate = float(fallback_learning_rate)
        learning_rate_source_path = resolved_checkpoint_dir / "<fallback-learning-rate>"

    if learning_rate is None or learning_rate_source_path is None:
        raise ValueError(
            "No learning_rate found in checkpoint metrics. Checked "
            "iteration_metrics.jsonl, iteration_metrics.json, metrics.json, "
            "and training_config.json."
        )

    return RestartCheckpointMetadata(
        checkpoint_dir=str(resolved_checkpoint_dir),
        restart_iteration=restart_iteration,
        next_iteration=restart_iteration + 1,
        learning_rate=float(learning_rate),
        iteration_source_path=str(iteration_metrics_jsonl),
        learning_rate_source_path=str(learning_rate_source_path),
    )


def prepare_restart_checkpoint(
    *,
    download_source: str | None,
    zip_path: str | Path,
    extract_dir: str | Path,
    skip_existing: bool = False,
    fallback_learning_rate: float | None = None,
) -> dict[str, Any]:
    zip_destination = Path(zip_path).expanduser().resolve()
    checkpoint_dir = Path(extract_dir).expanduser().resolve()
    downloaded = False

    if not (skip_existing and checkpoint_artifact_is_ready(checkpoint_dir)):
        source = str(download_source or "").strip()
        if not zip_destination.exists():
            file_id = resolve_google_drive_file_id(source)
            if file_id is None:
                raise ValueError("Missing Google Drive checkpoint download source.")
            download_google_drive_zip(file_id, zip_destination)
            downloaded = True
        extract_checkpoint_archive(zip_destination, checkpoint_dir)

    metadata = read_restart_checkpoint_metadata(
        checkpoint_dir,
        fallback_learning_rate=fallback_learning_rate,
    ).to_dict()
    metadata.update(
        {
            "status": "ready",
            "downloaded": downloaded,
            "download_source": str(download_source or "").strip(),
            "zip_path": str(zip_destination),
            "extract_dir": str(checkpoint_dir),
        }
    )
    return metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Download/extract a trained GFlowNet checkpoint and report restart "
            "iteration and learning-rate metadata."
        )
    )
    parser.add_argument(
        "--download-source",
        "--source",
        dest="download_source",
        default=None,
        help="Google Drive file id or share URL for the zipped GFlowNet checkpoint bundle.",
    )
    parser.add_argument(
        "--zip-path",
        required=True,
        help="Destination path for the downloaded checkpoint zip.",
    )
    parser.add_argument(
        "--extract-dir",
        required=True,
        help="Directory where the checkpoint should be extracted.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Reuse extract-dir when it already contains a complete checkpoint.",
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
        payload = prepare_restart_checkpoint(
            download_source=args.download_source,
            zip_path=args.zip_path,
            extract_dir=args.extract_dir,
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
