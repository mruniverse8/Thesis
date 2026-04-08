#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data_collection import download_and_preprocess_lpm24


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download and preprocess LPM-24 into grouped multi-molecule JSONL files."
    )
    parser.add_argument(
        "--train-dataset-name",
        default="language-plus-molecules/LPM-24_train",
        help="Hugging Face dataset identifier for the LPM-24 training split.",
    )
    parser.add_argument(
        "--eval-dataset-name",
        default="language-plus-molecules/LPM-24_eval-molgen",
        help="Hugging Face dataset identifier for the LPM-24 molecule-generation evaluation set.",
    )
    parser.add_argument(
        "--test-dataset-name",
        dest="eval_dataset_name",
        help="Deprecated alias for --eval-dataset-name.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(PROJECT_ROOT / "data" / "lpm24"),
        help="Directory where raw metadata and grouped JSONL files will be written.",
    )
    parser.add_argument(
        "--test-eval-description-limit",
        type=int,
        default=1000,
        help="Number of grouped test descriptions to keep in the paper-style evaluation subset.",
    )
    parser.add_argument(
        "--max-train-descriptions",
        type=int,
        default=None,
        help="Optional debugging limit applied after grouping the training descriptions.",
    )
    parser.add_argument(
        "--max-test-descriptions",
        type=int,
        default=None,
        help="Optional debugging limit applied after grouping the test descriptions.",
    )
    parser.add_argument(
        "--max-molecules-per-example",
        type=int,
        default=256,
        help="Maximum number of unique molecules to retain per grouped description.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = download_and_preprocess_lpm24(
        train_dataset_name=args.train_dataset_name,
        eval_dataset_name=args.eval_dataset_name,
        output_dir=args.output_dir,
        test_eval_description_limit=args.test_eval_description_limit,
        max_train_descriptions=args.max_train_descriptions,
        max_test_descriptions=args.max_test_descriptions,
        max_molecules_per_example=args.max_molecules_per_example,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
