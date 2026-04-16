from __future__ import annotations

import json
import shlex
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from .io_utils import PROJECT_ROOT, load_yaml, resolve_path
from .train_dataset_bootstrap import (
    DEFAULT_TRAIN_DATASET_EXTRACT_DIR,
    DEFAULT_TRAIN_DATASET_FILE_ID,
    DEFAULT_TRAIN_DATASET_ZIP_PATH,
    dataset_is_ready as mini_dataset_is_ready,
)

DEFAULT_REPO_URL = "https://github.com/mruniverse8/Thesis.git"
DEFAULT_REPO_BRANCH = "gflownet"
DATASET_MODES = ("auto", "always", "never")
CHEBI_REQUIRED_RELATIVE_PATHS = (
    Path("data") / "chebi20" / "processed" / "train.jsonl",
    Path("data") / "chebi20" / "processed" / "validation.jsonl",
    Path("data") / "chebi20" / "processed" / "test.jsonl",
)

@dataclass(frozen=True)
class RuntimeEnvironment:
    name: str
    workspace_root: Path
    default_repo_dir: Path


@dataclass(frozen=True)
class StageSpec:
    stage: str
    training_script: Path
    default_config: Path


STAGE_SPECS: dict[str, StageSpec] = {
    "sft": StageSpec(
        stage="sft",
        training_script=Path("scripts") / "train_sft.py",
        default_config=Path("configs") / "sft_chebi20.yaml",
    ),
    "multi_sft": StageSpec(
        stage="multi_sft",
        training_script=Path("scripts") / "train_multi_molecule_sft.py",
        default_config=Path("configs") / "multi_molecule_sft_mini.yaml",
    ),
    "ppo": StageSpec(
        stage="ppo",
        training_script=Path("scripts") / "train_molecule_wise_ppo.py",
        default_config=Path("configs") / "molecule_wise_ppo_mini.yaml",
    ),
}


def json_dumps(payload: object) -> str:
    return json.dumps(payload, indent=2, sort_keys=True)


def print_json_status(event: str, **payload: object) -> None:
    message = {"event": event, **payload}
    print(json_dumps(message))


def resolve_stage_spec(stage: str) -> StageSpec:
    try:
        return STAGE_SPECS[stage]
    except KeyError as exc:
        supported = ", ".join(sorted(STAGE_SPECS))
        raise ValueError(f"Unsupported stage {stage!r}. Expected one of: {supported}") from exc


def resolve_repo_dir(repo_dir: str | Path) -> Path:
    return Path(repo_dir).expanduser().resolve()


def summarize_environment(environment: RuntimeEnvironment) -> dict[str, object]:
    disk_usage = shutil.disk_usage(environment.workspace_root)
    return {
        "environment": environment.name,
        "python": sys.version.split()[0],
        "workspace_root": str(environment.workspace_root),
        "default_repo_dir": str(environment.default_repo_dir),
        "workspace_free_gb": round(disk_usage.free / (1024**3), 2),
    }


def format_command(command: Sequence[str | Path]) -> str:
    return shlex.join(str(part) for part in command)


def run_command(
    command: Sequence[str | Path],
    *,
    cwd: str | Path | None = None,
    env: dict[str, str] | None = None,
) -> None:
    command_text = format_command(command)
    print_json_status(
        "run_command",
        command=command_text,
        cwd=str(Path(cwd).resolve()) if cwd is not None else None,
    )
    subprocess.run(
        [str(part) for part in command],
        cwd=str(cwd) if cwd is not None else None,
        env=env,
        check=True,
    )


def clone_or_update_repo(
    *,
    repo_url: str,
    repo_branch: str,
    repo_dir: str | Path,
    current_repo_root: str | Path | None = None,
) -> Path:
    destination = resolve_repo_dir(repo_dir)
    active_repo_root = resolve_repo_dir(current_repo_root or PROJECT_ROOT)

    if destination == active_repo_root and (destination / ".git").exists():
        print_json_status("reuse_current_repo", repo_dir=str(destination))
        return destination

    if destination.exists() and not (destination / ".git").exists():
        raise RuntimeError(
            f"Refusing to initialize repo at {destination}: the directory exists but is not a git checkout."
        )

    if not destination.exists():
        run_command(["git", "clone", "--depth", "1", repo_url, destination])

    run_command(["git", "-C", destination, "fetch", "--depth", "1", repo_url, repo_branch])
    run_command(["git", "-C", destination, "checkout", "-B", repo_branch, "FETCH_HEAD"])
    return destination


def install_repo_requirements(repo_dir: str | Path) -> None:
    repo_root = resolve_repo_dir(repo_dir)
    requirements_path = repo_root / "requirements.txt"
    if not requirements_path.exists():
        raise FileNotFoundError(f"Missing requirements file: {requirements_path}")

    run_command([sys.executable, "-m", "pip", "install", "--upgrade", "pip", "setuptools", "wheel"])
    run_command([sys.executable, "-m", "pip", "install", "-r", requirements_path])


def resolve_stage_config_path(
    repo_dir: str | Path,
    stage_spec: StageSpec,
    config_override: str | Path | None = None,
) -> Path:
    repo_root = resolve_repo_dir(repo_dir)
    target = config_override or stage_spec.default_config
    return resolve_path(target, repo_root)


def _load_config_data_paths(config_path: Path, repo_dir: Path) -> dict[str, Path]:
    config = load_yaml(config_path)
    data = config.get("data", {})
    resolved: dict[str, Path] = {}
    for field in ("train_file", "validation_file", "test_file"):
        value = data.get(field)
        if value:
            resolved[field] = resolve_path(value, repo_dir)
    return resolved


def infer_managed_dataset(
    *,
    stage_spec: StageSpec,
    config_path: str | Path,
    repo_dir: str | Path,
) -> str:
    resolved_config_path = Path(config_path).resolve()
    resolved_repo_dir = resolve_repo_dir(repo_dir)

    try:
        data_paths = _load_config_data_paths(resolved_config_path, resolved_repo_dir)
    except Exception:
        data_paths = {}

    searchable_values = [str(resolved_config_path)] + [str(path) for path in data_paths.values()]
    searchable_text = " ".join(searchable_values).lower()

    if stage_spec.stage == "sft":
        default_paths = {
            "train_file": resolved_repo_dir / CHEBI_REQUIRED_RELATIVE_PATHS[0],
            "validation_file": resolved_repo_dir / CHEBI_REQUIRED_RELATIVE_PATHS[1],
            "test_file": resolved_repo_dir / CHEBI_REQUIRED_RELATIVE_PATHS[2],
        }
        if not data_paths or all(data_paths.get(name, default_paths[name]) == default_paths[name] for name in default_paths):
            return "chebi20"
        if "chebi20" in searchable_text:
            return "chebi20"
        return "manual"

    if "mini_post_training" in searchable_text or "_mini.yaml" in searchable_text or resolved_config_path.stem.endswith("_mini"):
        return "mini_post_training"
    return "manual"


def chebi_dataset_is_ready(repo_dir: str | Path) -> bool:
    repo_root = resolve_repo_dir(repo_dir)
    return all((repo_root / relative_path).exists() for relative_path in CHEBI_REQUIRED_RELATIVE_PATHS)


def build_dataset_prep_command(
    *,
    repo_dir: str | Path,
    stage_spec: StageSpec,
    config_path: str | Path,
    dataset_mode: str,
    train_dataset_file_id: str = DEFAULT_TRAIN_DATASET_FILE_ID,
) -> tuple[str, list[str] | None]:
    if dataset_mode not in DATASET_MODES:
        supported = ", ".join(DATASET_MODES)
        raise ValueError(f"Unsupported dataset mode {dataset_mode!r}. Expected one of: {supported}")

    repo_root = resolve_repo_dir(repo_dir)
    dataset_kind = infer_managed_dataset(stage_spec=stage_spec, config_path=config_path, repo_dir=repo_root)

    if dataset_mode == "never" or dataset_kind == "manual":
        return dataset_kind, None

    if dataset_kind == "chebi20":
        if dataset_mode == "auto" and chebi_dataset_is_ready(repo_root):
            return dataset_kind, None
        return dataset_kind, [
            sys.executable,
            str(repo_root / "scripts" / "download_chebi20.py"),
            "--output-dir",
            str(repo_root / "data" / "chebi20"),
        ]

    extract_dir = repo_root / DEFAULT_TRAIN_DATASET_EXTRACT_DIR
    if dataset_mode == "auto" and mini_dataset_is_ready(extract_dir):
        return dataset_kind, None

    command = [
        sys.executable,
        str(repo_root / "scripts" / "download_train_dataset.py"),
        "--file-id",
        train_dataset_file_id,
        "--zip-path",
        str(repo_root / DEFAULT_TRAIN_DATASET_ZIP_PATH),
        "--extract-dir",
        str(extract_dir),
    ]
    if dataset_mode == "always":
        command.append("--force-download")
    else:
        command.append("--skip-existing")
    return dataset_kind, command
