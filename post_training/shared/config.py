from __future__ import annotations

from pathlib import Path
from typing import Any

from src.config_utils import resolve_config_path_fields
from src.io_utils import PROJECT_ROOT, resolve_path


DEFAULT_PPO_FALLBACK_CHECKPOINT = "QizhiPei/biot5-plus-base-chebi20"
_LOCAL_CHECKPOINT_ROOT_NAMES = frozenset(
    {
        "checkpoints",
        "colab",
        "configs",
        "data",
        "kaggle",
        "legacy",
        "molecules",
        "notebooks",
        "outputs",
        "post_training",
        "reward_utils",
        "scripts",
        "src",
        "tests",
    }
)


def resolve_multi_molecule_sft_config_paths(
    config: dict[str, Any],
    project_root: Path | None = None,
) -> dict[str, Any]:
    return resolve_config_path_fields(
        config,
        fields_by_section={
            "data": ("train_file", "validation_file", "test_file"),
            "training": ("output_dir",),
        },
        project_root=project_root,
    )


def resolve_ppo_config_paths(
    config: dict[str, Any],
    project_root: Path | None = None,
) -> dict[str, Any]:
    resolved = resolve_config_path_fields(
        config,
        fields_by_section={
            "data": ("train_file", "validation_file", "test_file"),
            "training": ("output_dir",),
        },
        project_root=project_root,
    )
    if "model" in resolved and "checkpoint" in resolved["model"]:
        resolved["model"]["checkpoint"] = resolve_ppo_checkpoint_source(
            resolved["model"]["checkpoint"],
            project_root=project_root,
        )
    return resolved


def looks_like_remote_model_identifier(checkpoint_value: str | Path) -> bool:
    checkpoint_text = str(checkpoint_value).strip()
    if not checkpoint_text:
        return False
    if checkpoint_text.startswith(("hf://", "http://", "https://")):
        return True
    if checkpoint_text.startswith(("/", "./", "../", "~")):
        return False

    candidate = Path(checkpoint_text)
    if candidate.suffix:
        return False
    if any(part in {".", ".."} for part in candidate.parts):
        return False
    if len(candidate.parts) != 2:
        return False
    return candidate.parts[0] not in _LOCAL_CHECKPOINT_ROOT_NAMES


def _looks_like_remote_model_identifier(checkpoint_value: str | Path) -> bool:
    return looks_like_remote_model_identifier(checkpoint_value)


def resolve_ppo_checkpoint_source(
    checkpoint_value: str | Path,
    *,
    project_root: Path | None = None,
    fallback_checkpoint: str = DEFAULT_PPO_FALLBACK_CHECKPOINT,
) -> str:
    resolved_local_path = resolve_path(checkpoint_value, project_root or PROJECT_ROOT)
    if resolved_local_path.exists():
        return str(resolved_local_path)
    if _looks_like_remote_model_identifier(checkpoint_value):
        return str(checkpoint_value)
    return fallback_checkpoint


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
