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


def detect_dataset_family(*candidates: object) -> str:
    for candidate in candidates:
        text = str(candidate or "").lower()
        if "chebi" in text:
            return "chebi20"
        if "data/post_training/processed" in text or "/post_training/processed/" in text:
            return "chebi20"
        if "lpm24" in text or "language-plus-molecules" in text:
            return "lpm24"
    return "generic"
