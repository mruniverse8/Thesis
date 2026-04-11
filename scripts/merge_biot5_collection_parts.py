#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data_collection import merge_biot5_collection_parts
from src.io_utils import resolve_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge partitioned BioT5 collection outputs into one merged bundle."
    )
    parser.add_argument(
        "--part-staging-dir",
        action="append",
        required=True,
        help="Path to one part collection staging directory containing summary.json and JSONL outputs.",
    )
    parser.add_argument(
        "--part-derived-train-file",
        action="append",
        required=True,
        help="Path to one part derived train_multimol.jsonl file.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Path to the merged collection output directory.",
    )
    parser.add_argument(
        "--merged-derived-train-file",
        required=True,
        help="Path to the merged train_multimol.jsonl file.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = merge_biot5_collection_parts(
        part_staging_dirs=[
            resolve_path(path_value, PROJECT_ROOT)
            for path_value in args.part_staging_dir
        ],
        part_derived_train_files=[
            resolve_path(path_value, PROJECT_ROOT)
            for path_value in args.part_derived_train_file
        ],
        output_dir=resolve_path(args.output_dir, PROJECT_ROOT),
        merged_derived_train_file=resolve_path(args.merged_derived_train_file, PROJECT_ROOT),
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
