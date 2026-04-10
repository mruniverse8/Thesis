from __future__ import annotations

from pathlib import Path
from typing import Any

from src.io_utils import dump_yaml, ensure_dir, write_json


def initialize_run_output(
    output_dir: str | Path,
    *,
    config: dict[str, Any] | None = None,
) -> Path:
    resolved_output_dir = ensure_dir(output_dir)
    ensure_dir(resolved_output_dir / "checkpoints")
    if config is not None:
        dump_yaml(resolved_output_dir / "resolved_config.yaml", config)
    return resolved_output_dir


def write_history_json(output_dir: str | Path, history: list[dict[str, Any]]) -> None:
    write_json(Path(output_dir) / "history.json", history)
