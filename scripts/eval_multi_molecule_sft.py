#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from post_training.config_utils import resolve_multi_molecule_sft_config_paths
from post_training.sft_evaluation import evaluate_multi_molecule_checkpoint
from src.io_utils import load_yaml, resolve_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run generation evaluation for a multi-molecule BioT5+ SFT checkpoint."
    )
    parser.add_argument(
        "--config",
        default="configs/multi_molecule_sft.yaml",
        help="Path to the YAML config relative to the project root.",
    )
    parser.add_argument(
        "--checkpoint",
        default="outputs/multi_molecule_sft/checkpoints/best",
        help="Checkpoint directory produced by train_multi_molecule_sft.py.",
    )
    parser.add_argument(
        "--split",
        default="validation",
        choices=("train", "validation", "test"),
        help="Dataset split to evaluate.",
    )
    parser.add_argument(
        "--prediction-file",
        default=None,
        help="Optional output JSONL path for prediction dumps.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = resolve_path(args.config, PROJECT_ROOT)
    config = resolve_multi_molecule_sft_config_paths(load_yaml(config_path), project_root=PROJECT_ROOT)

    metrics, prediction_path = evaluate_multi_molecule_checkpoint(
        config=config,
        checkpoint_path=resolve_path(args.checkpoint, PROJECT_ROOT),
        split=args.split,
        prediction_path=resolve_path(args.prediction_file, PROJECT_ROOT) if args.prediction_file else None,
    )
    payload = {
        "split": args.split,
        "prediction_file": str(prediction_path),
        "metrics": metrics,
    }
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
