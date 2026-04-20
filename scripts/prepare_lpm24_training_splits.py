#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data_collection import export_lpm24_training_splits


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create training-ready train/validation/test grouped splits for LPM24."
    )
    parser.add_argument(
        "--input-dir",
        default=str(PROJECT_ROOT / "data" / "lpm24"),
        help="Directory containing the existing data/lpm24/processed JSONL files.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Optional output directory. Defaults to <input-dir>/grouped_splits.",
    )
    parser.add_argument(
        "--validation-fraction",
        type=float,
        default=0.05,
        help="Fraction of filtered train records to reserve for validation.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used when shuffling train records before the validation split.",
    )
    parser.add_argument(
        "--max-target-symbols",
        type=int,
        default=1024,
        help="Maximum staged-target SELFIES symbol count allowed per grouped record; 0 disables the filter.",
    )
    parser.add_argument(
        "--max-stage-symbols",
        type=int,
        default=192,
        help="Maximum SELFIES symbol count allowed for any single target molecule; 0 disables the filter.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = export_lpm24_training_splits(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        validation_fraction=args.validation_fraction,
        seed=args.seed,
        max_target_symbols=args.max_target_symbols,
        max_stage_symbols=args.max_stage_symbols,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
