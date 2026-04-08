from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Iterable

from .io_utils import PROJECT_ROOT, resolve_path


def resolve_config_path_fields(
    config: dict[str, Any],
    fields_by_section: dict[str, Iterable[str]],
    project_root: Path | None = None,
) -> dict[str, Any]:
    resolved = deepcopy(config)
    root = project_root or PROJECT_ROOT

    for section, fields in fields_by_section.items():
        if section not in resolved:
            continue
        for field in fields:
            if field not in resolved[section]:
                continue
            resolved[section][field] = str(resolve_path(resolved[section][field], root))
    return resolved


def resolve_runtime_config_paths(
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
