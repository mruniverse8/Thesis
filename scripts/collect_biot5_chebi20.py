#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data_collection import collect_biot5_training_data, resolve_biot5_collection_config_paths
from src.io_utils import load_yaml, resolve_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect grouped ChEBI-20 training molecules with base BioT5+ and contrastive search."
    )
    parser.add_argument(
        "--config",
        default="configs/collect_biot5_chebi20.yaml",
        help="Path to the BioT5 collection YAML config relative to the project root.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = resolve_path(args.config, PROJECT_ROOT)
    config = resolve_biot5_collection_config_paths(load_yaml(config_path), project_root=PROJECT_ROOT)
    summary = collect_biot5_training_data(config)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
