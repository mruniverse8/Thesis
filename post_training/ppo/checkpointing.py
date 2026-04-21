from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

from src.io_utils import ensure_dir, write_json, write_jsonl

from post_training.shared.io import initialize_run_output, write_history_json


def prepare_ppo_output_dir(
    output_dir: str | Path,
    *,
    config: dict[str, Any],
    reward_config: dict[str, Any],
) -> Path:
    resolved_output_dir = initialize_run_output(output_dir, config=config)
    ensure_dir(resolved_output_dir / "diagnostics")
    write_json(resolved_output_dir / "reward_config.json", reward_config)
    return resolved_output_dir


def write_ppo_history(output_dir: str | Path, history: list[dict[str, Any]]) -> None:
    write_history_json(output_dir, history)


def _append_jsonl(path_value: str | Path, records: Sequence[dict[str, Any]]) -> None:
    if not records:
        return

    path = Path(path_value)
    ensure_dir(path.parent)
    with path.open("a", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False))
            handle.write("\n")


def append_optimizer_step_metrics(
    output_dir: str | Path,
    metrics: Sequence[dict[str, Any]],
) -> Path:
    diagnostics_path = Path(output_dir) / "diagnostics" / "optimizer_step_metrics.jsonl"
    _append_jsonl(diagnostics_path, metrics)
    return diagnostics_path


def append_iteration_diagnostics(
    output_dir: str | Path,
    metrics: Sequence[dict[str, Any]],
) -> Path:
    diagnostics_path = Path(output_dir) / "diagnostics" / "iteration_diagnostics.jsonl"
    _append_jsonl(diagnostics_path, metrics)
    return diagnostics_path


def append_trajectory_previews(
    output_dir: str | Path,
    previews: Sequence[dict[str, Any]],
) -> Path:
    diagnostics_path = Path(output_dir) / "diagnostics" / "trajectory_previews.jsonl"
    _append_jsonl(diagnostics_path, previews)
    return diagnostics_path


def save_iteration_artifacts(
    *,
    output_dir: str | Path,
    iteration_index: int,
    policy_model,
    tokenizer,
    config: dict[str, Any],
    metrics: dict[str, Any],
    trajectories: Sequence,
    create_archive: bool = False,
) -> Path:
    checkpoint_dir = Path(output_dir) / "checkpoints" / f"iteration-{iteration_index:04d}"
    policy_model.save_checkpoint(
        checkpoint_dir,
        tokenizer=tokenizer,
        config=config,
        metrics=metrics,
        create_archive=create_archive,
    )
    write_jsonl(
        checkpoint_dir / "rollout_summary.jsonl",
        [trajectory.to_dict() for trajectory in trajectories],
    )
    write_json(checkpoint_dir / "iteration_metrics.json", metrics)
    return checkpoint_dir
