#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from post_training.gflownet.trainer import run_multi_molecule_gflownet
from post_training.shared.config import resolve_gflownet_config_paths
from src.io_utils import load_yaml, resolve_path, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the multi-molecule GFlowNet post-training stage."
    )
    parser.add_argument(
        "--config",
        default="configs/multi_molecule_gflownet.yaml",
        help="Path to the YAML GFlowNet config relative to the project root.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = resolve_path(args.config, PROJECT_ROOT)
    config = resolve_gflownet_config_paths(load_yaml(config_path), project_root=PROJECT_ROOT)
    set_seed(int(config.get("seed", 42)))
    summary = run_multi_molecule_gflownet(config)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
