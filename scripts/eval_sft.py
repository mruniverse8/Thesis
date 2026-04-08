#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation import MoleculeMetricConfig
from src.config_utils import resolve_runtime_config_paths
from src.evaluation import evaluate_checkpoint
from src.io_utils import load_yaml, resolve_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run generation evaluation for a trained BioT5+ SFT checkpoint.")
    parser.add_argument(
        "--config",
        default="configs/sft_chebi20.yaml",
        help="Path to the YAML config relative to the project root.",
    )
    parser.add_argument(
        "--checkpoint",
        default="outputs/chebi20_sft/checkpoints/best",
        help="Checkpoint directory produced by train_sft.py.",
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
    parser.add_argument(
        "--molecule-metrics-file",
        default=None,
        help="Optional output JSON path for molecule-level evaluation metrics.",
    )
    parser.add_argument(
        "--group-metrics-file",
        default=None,
        help="Optional output JSONL path for per-description molecule metric reports.",
    )
    parser.add_argument(
        "--acceptance-dice-threshold",
        type=float,
        default=0.7,
        help="Dice similarity threshold used to decide whether a molecule is accepted.",
    )
    parser.add_argument(
        "--ncircles-tanimoto-threshold",
        type=float,
        default=0.6,
        help="Tanimoto threshold used by the NCircles compatibility rule.",
    )
    parser.add_argument(
        "--skip-molecule-metrics",
        action="store_true",
        help="Skip Accepted & Unique, NCircles, and IntDiv computation.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = resolve_path(args.config, PROJECT_ROOT)
    config = resolve_runtime_config_paths(load_yaml(config_path), project_root=PROJECT_ROOT)
    metric_config = MoleculeMetricConfig(
        acceptance_dice_threshold=args.acceptance_dice_threshold,
        ncircles_tanimoto_threshold=args.ncircles_tanimoto_threshold,
    )

    result = evaluate_checkpoint(
        config=config,
        checkpoint_path=resolve_path(args.checkpoint, PROJECT_ROOT),
        split=args.split,
        prediction_path=resolve_path(args.prediction_file, PROJECT_ROOT) if args.prediction_file else None,
        molecule_metrics_path=(
            resolve_path(args.molecule_metrics_file, PROJECT_ROOT)
            if args.molecule_metrics_file
            else None
        ),
        group_metrics_path=(
            resolve_path(args.group_metrics_file, PROJECT_ROOT)
            if args.group_metrics_file
            else None
        ),
        metric_config=metric_config,
        compute_molecule_metrics=not args.skip_molecule_metrics,
    )
    payload = {
        "split": args.split,
        "prediction_file": str(result.prediction_path),
        "generation_metrics": result.generation_metrics,
    }
    if result.molecule_metrics_path is not None:
        payload["molecule_metrics_file"] = str(result.molecule_metrics_path)
    if result.group_metrics_path is not None:
        payload["group_metrics_file"] = str(result.group_metrics_path)
    if result.molecule_metrics is not None:
        payload["molecule_metrics"] = result.molecule_metrics
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
