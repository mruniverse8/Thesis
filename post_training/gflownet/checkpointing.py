from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

from src.io_utils import write_json, write_jsonl

from post_training.shared.io import initialize_run_output, write_history_json


def prepare_gflownet_output_dir(
    output_dir: str | Path,
    *,
    config: dict[str, Any],
    reward_config: dict[str, Any],
) -> Path:
    resolved_output_dir = initialize_run_output(output_dir, config=config)
    write_json(resolved_output_dir / "reward_config.json", reward_config)
    return resolved_output_dir


def write_gflownet_history(output_dir: str | Path, history: list[dict[str, Any]]) -> None:
    write_history_json(output_dir, history)


def save_gflownet_iteration_artifacts(
    *,
    output_dir: str | Path,
    iteration_index: int,
    model,
    tokenizer,
    config: dict[str, Any],
    metrics: dict[str, Any],
    trajectories: Sequence,
    create_archive: bool = False,
) -> Path:
    checkpoint_dir = Path(output_dir) / "checkpoints" / f"iteration-{iteration_index:04d}"
    model.save_checkpoint(
        checkpoint_dir,
        tokenizer=tokenizer,
        config=config,
        metrics=metrics,
        create_archive=create_archive,
    )
    write_jsonl(
        checkpoint_dir / "sampled_trajectories.jsonl",
        [trajectory.to_dict() for trajectory in trajectories],
    )
    write_json(checkpoint_dir / "iteration_metrics.json", metrics)
    return checkpoint_dir
