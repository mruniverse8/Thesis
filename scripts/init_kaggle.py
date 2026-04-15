#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from kaggle.thesis_kaggle_support import KAGGLE_REPO_DIR
from kaggle.thesis_kaggle_support import get_bootstrap_environment
from src.runtime_bootstrap import (
    DEFAULT_REPO_BRANCH,
    DEFAULT_REPO_URL,
    DEFAULT_TRAIN_DATASET_FILE_ID,
    build_dataset_prep_command,
    build_training_command,
    clone_or_update_repo,
    install_repo_requirements,
    json_dumps,
    print_json_status,
    resolve_stage_config_path,
    resolve_stage_spec,
    run_command,
    summarize_environment,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Initialize a Kaggle runtime and launch one thesis training stage.")
    parser.add_argument("--stage", required=True, choices=("sft", "multi_sft", "ppo"), help="Training stage to run.")
    parser.add_argument("--repo-url", default=DEFAULT_REPO_URL, help="HTTPS Git URL for the thesis repo.")
    parser.add_argument("--repo-branch", default=DEFAULT_REPO_BRANCH, help="Git branch to fetch and run.")
    parser.add_argument(
        "--repo-dir",
        default=str(KAGGLE_REPO_DIR),
        help="Kaggle checkout path where the repo should exist.",
    )
    parser.add_argument(
        "--dataset-mode",
        default="auto",
        choices=("auto", "always", "never"),
        help="Whether to prepare the managed dataset automatically before training.",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Optional config override. Defaults to the stage-specific Kaggle-friendly config.",
    )
    parser.add_argument(
        "--train-dataset-file-id",
        default=DEFAULT_TRAIN_DATASET_FILE_ID,
        help="Google Drive file id used by the mini post-training dataset downloader.",
    )
    parser.add_argument(
        "--extra-script-args",
        nargs=argparse.REMAINDER,
        default=[],
        help="Additional arguments forwarded to the underlying training script. Prefix with --extra-script-args -- ...",
    )
    return parser.parse_args()


def _normalize_extra_args(values: list[str]) -> list[str]:
    if values and values[0] == "--":
        return values[1:]
    return values


def main() -> None:
    args = parse_args()
    stage_spec = resolve_stage_spec(args.stage)
    environment = get_bootstrap_environment(args.repo_dir)
    print(json_dumps(summarize_environment(environment)))

    repo_dir = clone_or_update_repo(
        repo_url=args.repo_url,
        repo_branch=args.repo_branch,
        repo_dir=args.repo_dir,
        current_repo_root=PROJECT_ROOT,
    )
    install_repo_requirements(repo_dir)

    config_path = resolve_stage_config_path(repo_dir, stage_spec, args.config)
    dataset_kind, dataset_command = build_dataset_prep_command(
        repo_dir=repo_dir,
        stage_spec=stage_spec,
        config_path=config_path,
        dataset_mode=args.dataset_mode,
        train_dataset_file_id=args.train_dataset_file_id,
    )
    print_json_status(
        "dataset_plan",
        stage=args.stage,
        dataset_mode=args.dataset_mode,
        dataset_kind=dataset_kind,
        config_path=str(config_path),
    )
    if dataset_command is not None:
        run_command(dataset_command, cwd=repo_dir)

    training_command = build_training_command(
        repo_dir=repo_dir,
        stage_spec=stage_spec,
        config_path=config_path,
        extra_script_args=_normalize_extra_args(args.extra_script_args),
    )
    run_command(training_command, cwd=repo_dir)


if __name__ == "__main__":
    main()
