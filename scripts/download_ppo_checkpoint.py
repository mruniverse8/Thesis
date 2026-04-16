#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.checkpoint_bootstrap import (
    DEFAULT_PPO_CHECKPOINT_EXTRACT_DIR,
    DEFAULT_PPO_CHECKPOINT_ZIP_PATH,
    checkpoint_artifact_is_ready,
    extract_checkpoint_archive,
    resolve_google_drive_file_id,
)
from src.runtime_bootstrap import json_dumps
from src.train_dataset_bootstrap import download_google_drive_zip, reset_download_targets


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download and extract an optional PPO checkpoint bundle for Colab/Kaggle bootstrap."
    )
    parser.add_argument(
        "--download-source",
        "--source",
        dest="download_source",
        default=None,
        help="Google Drive file id or share URL for the zipped checkpoint bundle.",
    )
    parser.add_argument(
        "--zip-path",
        default=str(PROJECT_ROOT / DEFAULT_PPO_CHECKPOINT_ZIP_PATH),
        help="Destination path for the downloaded checkpoint zip.",
    )
    parser.add_argument(
        "--extract-dir",
        default=str(PROJECT_ROOT / DEFAULT_PPO_CHECKPOINT_EXTRACT_DIR),
        help="Directory where the checkpoint should be extracted.",
    )
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--force-download",
        action="store_true",
        help="Delete any existing zip/extracted checkpoint and fetch it again.",
    )
    mode_group.add_argument(
        "--skip-existing",
        action="store_true",
        help="Exit successfully without downloading when the checkpoint is already ready.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    download_source = str(args.download_source or "").strip()
    zip_path = Path(args.zip_path).expanduser().resolve()
    extract_dir = Path(args.extract_dir).expanduser().resolve()

    if not download_source:
        print(
            json_dumps(
                {
                    "status": "skipped_no_download_source",
                    "zip_path": str(zip_path),
                    "extract_dir": str(extract_dir),
                }
            )
        )
        return

    if args.skip_existing and checkpoint_artifact_is_ready(extract_dir):
        print(
            json_dumps(
                {
                    "status": "skipped_existing",
                    "zip_path": str(zip_path),
                    "extract_dir": str(extract_dir),
                }
            )
        )
        return

    if args.force_download:
        reset_download_targets(zip_path, extract_dir)

    downloaded = False
    try:
        file_id = resolve_google_drive_file_id(download_source)
        if file_id is None:
            raise ValueError("Missing Google Drive checkpoint download source.")
        if not zip_path.exists():
            download_google_drive_zip(file_id, zip_path)
            downloaded = True
        extract_checkpoint_archive(zip_path, extract_dir)
        if not checkpoint_artifact_is_ready(extract_dir):
            raise RuntimeError("Extracted checkpoint is incomplete.")
        print(
            json_dumps(
                {
                    "status": "ready",
                    "downloaded": downloaded,
                    "file_id": file_id,
                    "zip_path": str(zip_path),
                    "extract_dir": str(extract_dir),
                }
            )
        )
    except Exception as exc:
        print(
            json_dumps(
                {
                    "status": "failed",
                    "error": str(exc),
                    "zip_path": str(zip_path),
                    "extract_dir": str(extract_dir),
                }
            )
        )


if __name__ == "__main__":
    main()
