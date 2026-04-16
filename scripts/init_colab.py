#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from colab.thesis_colab_support import DEFAULT_REPO_DIR as DEFAULT_COLAB_REPO_DIR
from colab.thesis_colab_support import get_bootstrap_environment
from src.checkpoint_bootstrap import build_ppo_checkpoint_prep_command
from src.runtime_bootstrap import (
    DEFAULT_REPO_BRANCH,
    DEFAULT_REPO_URL,
    DEFAULT_TRAIN_DATASET_FILE_ID,
    build_dataset_prep_command,
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
    parser = argparse.ArgumentParser(description="Initialize a Colab runtime and prepare one thesis training stage.")
    parser.add_argument(
        "--stage",
        required=True,
        choices=("sft", "multi_sft", "ppo"),
        help="Training stage to prepare.",
    )
    parser.add_argument("--repo-url", default=DEFAULT_REPO_URL, help="HTTPS Git URL for the thesis repo.")
    parser.add_argument("--repo-branch", default=DEFAULT_REPO_BRANCH, help="Git branch to fetch and run.")
    parser.add_argument(
        "--repo-dir",
        default=str(DEFAULT_COLAB_REPO_DIR),
        help="Colab checkout path where the repo should exist.",
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
        help="Optional config override. Defaults to the stage-specific Colab/Kaggle-friendly config.",
    )
    parser.add_argument(
        "--train-dataset-file-id",
        default=DEFAULT_TRAIN_DATASET_FILE_ID,
        help="Google Drive file id used by the mini post-training dataset downloader.",
    )
    parser.add_argument(
        "--ppo-checkpoint-download-source",
        "--ppo-checkpoint-source",
        dest="ppo_checkpoint_download_source",
        default=None,
        help="Optional Google Drive file id or share URL for a zipped PPO checkpoint bundle.",
    )
    return parser.parse_args()


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

    checkpoint_kind, checkpoint_command, checkpoint_target = ("not_applicable", None, None)
    if stage_spec.stage == "ppo":
        checkpoint_kind, checkpoint_command, checkpoint_target = build_ppo_checkpoint_prep_command(
            repo_dir=repo_dir,
            config_path=config_path,
            checkpoint_download_source=args.ppo_checkpoint_download_source,
        )
    print_json_status(
        "checkpoint_plan",
        stage=args.stage,
        checkpoint_kind=checkpoint_kind,
        checkpoint_target=checkpoint_target,
        checkpoint_download_source_provided=bool(str(args.ppo_checkpoint_download_source or "").strip()),
    )
    if checkpoint_command is not None:
        try:
            run_command(checkpoint_command, cwd=repo_dir)
        except Exception as exc:
            print_json_status(
                "checkpoint_prepare_failed",
                stage=args.stage,
                checkpoint_target=checkpoint_target,
                error=str(exc),
            )

    print_json_status(
        "bootstrap_complete",
        stage=args.stage,
        repo_dir=str(repo_dir),
        config_path=str(config_path),
        training_script=str(repo_dir / stage_spec.training_script),
    )


if __name__ == "__main__":
    main()
