#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.runtime_bootstrap import json_dumps
from src.train_dataset_bootstrap import (
    DEFAULT_TRAIN_DATASET_EXTRACT_DIR,
    DEFAULT_TRAIN_DATASET_FILE_ID,
    DEFAULT_TRAIN_DATASET_ZIP_PATH,
    dataset_is_ready,
    download_google_drive_zip,
    extract_dataset_archive,
    missing_dataset_paths,
    reset_download_targets,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download and extract the mini post-training dataset used by the Colab/Kaggle bootstrap flow."
    )
    parser.add_argument(
        "--file-id",
        default=DEFAULT_TRAIN_DATASET_FILE_ID,
        help="Google Drive file id for the zipped dataset bundle.",
    )
    parser.add_argument(
        "--zip-path",
        default=str(PROJECT_ROOT / DEFAULT_TRAIN_DATASET_ZIP_PATH),
        help="Destination path for the downloaded zip file.",
    )
    parser.add_argument(
        "--extract-dir",
        default=str(PROJECT_ROOT / DEFAULT_TRAIN_DATASET_EXTRACT_DIR),
        help="Directory where the dataset should be extracted.",
    )
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--force-download",
        action="store_true",
        help="Delete any existing zip/extracted dataset and fetch the archive again.",
    )
    mode_group.add_argument(
        "--skip-existing",
        action="store_true",
        help="Exit successfully without downloading when the extracted dataset is already complete.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    zip_path = Path(args.zip_path).expanduser().resolve()
    extract_dir = Path(args.extract_dir).expanduser().resolve()

    if args.skip_existing and dataset_is_ready(extract_dir):
        print(
            json_dumps(
                {
                    "status": "skipped_existing",
                    "file_id": args.file_id,
                    "zip_path": str(zip_path),
                    "extract_dir": str(extract_dir),
                }
            )
        )
        return

    if args.force_download:
        reset_download_targets(zip_path, extract_dir)

    downloaded = False
    if not zip_path.exists():
        download_google_drive_zip(args.file_id, zip_path)
        downloaded = True

    extract_dataset_archive(zip_path, extract_dir)

    missing_paths = [str(path) for path in missing_dataset_paths(extract_dir)]
    if missing_paths:
        raise RuntimeError(
            json_dumps(
                {
                    "status": "invalid_dataset_layout",
                    "extract_dir": str(extract_dir),
                    "missing_paths": missing_paths,
                }
            )
        )

    print(
        json_dumps(
            {
                "status": "ready",
                "file_id": args.file_id,
                "downloaded": downloaded,
                "zip_path": str(zip_path),
                "extract_dir": str(extract_dir),
                "required_files": [str(path) for path in Path(extract_dir).rglob("*.jsonl")],
            }
        )
    )


if __name__ == "__main__":
    main()
