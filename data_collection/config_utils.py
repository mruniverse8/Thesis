from __future__ import annotations

from pathlib import Path
from typing import Any

from src.config_utils import resolve_config_path_fields


def resolve_biot5_collection_config_paths(
    config: dict[str, Any],
    project_root: Path | None = None,
) -> dict[str, Any]:
    return resolve_config_path_fields(
        config,
        fields_by_section={
            "model": ("selfies_vocab_path",),
            "data": ("train_file", "staging_dir", "derived_train_file"),
        },
        project_root=project_root,
    )
