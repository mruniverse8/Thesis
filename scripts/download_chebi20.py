#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from datasets import load_dataset
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.datasets import build_processed_record
from src.io_utils import ensure_dir, write_json, write_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download and preprocess ChEBI-20-MM for BioT5+ SFT.")
    parser.add_argument(
        "--dataset-name",
        default="liupf/ChEBI-20-MM",
        help="Hugging Face dataset identifier.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(PROJECT_ROOT / "data" / "chebi20"),
        help="Directory where raw metadata and processed JSONL files will be written.",
    )
    parser.add_argument(
        "--max-samples-per-split",
        type=int,
        default=None,
        help="Optional debugging limit applied after preprocessing each split.",
    )
    parser.add_argument(
        "--disable-smiles-fallback",
        action="store_true",
        help="Do not convert SMILES to SELFIES when the SELFIES column is missing.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    output_dir = Path(args.output_dir).expanduser().resolve()
    raw_dir = ensure_dir(output_dir / "raw")
    processed_dir = ensure_dir(output_dir / "processed")

    dataset = load_dataset(args.dataset_name)
    metadata: dict[str, object] = {
        "dataset_name": args.dataset_name,
        "output_dir": str(output_dir),
        "splits": {},
    }

    for split_name in ("train", "validation", "test"):
        if split_name not in dataset:
            continue

        processed_records = []
        skipped_examples = 0
        split_dataset = dataset[split_name]

        for row_index, raw_record in enumerate(
            tqdm(split_dataset, desc=f"Processing {split_name}", leave=False)
        ):
            try:
                processed_record = build_processed_record(
                    raw_record=raw_record,
                    split=split_name,
                    index=row_index,
                    convert_missing_selfies_from_smiles=not args.disable_smiles_fallback,
                )
            except Exception:
                skipped_examples += 1
                continue

            processed_records.append(processed_record)
            if (
                args.max_samples_per_split is not None
                and len(processed_records) >= args.max_samples_per_split
            ):
                break

        write_jsonl(processed_dir / f"{split_name}.jsonl", processed_records)
        metadata["splits"][split_name] = {
            "written_examples": len(processed_records),
            "skipped_examples": skipped_examples,
        }

    write_json(raw_dir / "download_metadata.json", metadata)
    print(f"Wrote processed splits to {processed_dir}")


if __name__ == "__main__":
    main()
