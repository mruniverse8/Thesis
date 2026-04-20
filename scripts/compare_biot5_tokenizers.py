#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from transformers import AutoTokenizer, T5Tokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from post_training.shared.sequence import serialize_staged_molecule, serialize_staged_target
from src.constants import BOM_TOKEN, EOM_TOKEN
from src.tokenizer_utils import build_tokenizer_comparison_report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare fast AutoTokenizer and T5Tokenizer encodings for staged SELFIES text."
        )
    )
    parser.add_argument(
        "--model",
        default="QizhiPei/biot5-plus-base-chebi20",
        help="Checkpoint or model id to compare.",
    )
    parser.add_argument(
        "--text",
        action="append",
        default=[],
        help="Optional extra text to include in the report. May be passed multiple times.",
    )
    return parser.parse_args()


def build_default_texts() -> list[str]:
    return [
        BOM_TOKEN,
        EOM_TOKEN,
        serialize_staged_molecule("[C][C][O]"),
        serialize_staged_molecule("[C@@H1][Branch1][C][O][C]"),
        serialize_staged_target(["[C][C][O]", "[C][N]"]),
    ]


def main() -> None:
    args = parse_args()
    model_name_or_path = str(args.model)
    texts = build_default_texts()
    texts.extend(str(value) for value in args.text)

    auto_tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)
    t5_tokenizer = T5Tokenizer.from_pretrained(model_name_or_path)

    report = build_tokenizer_comparison_report(
        tokenizers={
            "auto_fast": auto_tokenizer,
            "t5_slow": t5_tokenizer,
        },
        texts=texts,
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
