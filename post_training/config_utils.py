from __future__ import annotations

from pathlib import Path
from typing import Any

from src.config_utils import resolve_config_path_fields


def resolve_multi_molecule_sft_config_paths(
    config: dict[str, Any],
    project_root: Path | None = None,
) -> dict[str, Any]:
    return resolve_config_path_fields(
        config,
        fields_by_section={
            "model": ("selfies_vocab_path",),
            "data": ("train_file", "validation_file", "test_file"),
            "training": ("output_dir",),
        },
        project_root=project_root,
    )


def resolve_ppo_config_paths(
    config: dict[str, Any],
    project_root: Path | None = None,
) -> dict[str, Any]:
    return resolve_config_path_fields(
        config,
        fields_by_section={
            "model": ("checkpoint",),
            "data": ("train_file", "validation_file", "test_file"),
            "training": ("output_dir",),
        },
        project_root=project_root,
    )
