from __future__ import annotations

import re
import shutil
import sys
from pathlib import Path
from urllib.parse import parse_qs, urlparse
from zipfile import ZIP_DEFLATED, ZipFile

from post_training.shared.config import looks_like_remote_model_identifier

from .io_utils import ensure_dir, load_yaml, resolve_path

DEFAULT_PPO_CHECKPOINT_EXTRACT_DIR = (
    Path("outputs") / "multi_molecule_sft_mini" / "checkpoints" / "best"
)
DEFAULT_PPO_CHECKPOINT_ZIP_PATH = (
    Path("outputs") / "multi_molecule_sft_mini" / "checkpoints" / "best.zip"
)
_MODEL_WEIGHT_FILENAMES = (
    "model.safetensors",
    "model.safetensors.index.json",
    "pytorch_model.bin",
    "pytorch_model.bin.index.json",
)
_TOKENIZER_FILENAMES = ("spiece.model", "tokenizer.json")
_GOOGLE_DRIVE_FILE_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]{10,}$")


def default_checkpoint_archive_path(checkpoint_dir: str | Path) -> Path:
    checkpoint_path = Path(checkpoint_dir)
    return checkpoint_path.parent / f"{checkpoint_path.name}.zip"


def archive_checkpoint_directory(
    checkpoint_dir: str | Path,
    archive_path: str | Path | None = None,
) -> Path:
    source_dir = Path(checkpoint_dir)
    if not source_dir.exists():
        raise FileNotFoundError(f"Missing checkpoint directory: {source_dir}")
    if not source_dir.is_dir():
        raise ValueError(f"Expected a checkpoint directory at {source_dir}")

    destination = (
        Path(archive_path)
        if archive_path is not None
        else default_checkpoint_archive_path(source_dir)
    )
    ensure_dir(destination.parent)

    with ZipFile(destination, "w", compression=ZIP_DEFLATED) as archive:
        for file_path in sorted(source_dir.rglob("*")):
            if file_path.is_file():
                archive.write(
                    file_path,
                    arcname=str(Path(source_dir.name) / file_path.relative_to(source_dir)),
                )
    return destination


def checkpoint_artifact_is_ready(checkpoint_dir: str | Path) -> bool:
    root = Path(checkpoint_dir)
    return (
        (root / "config.json").exists()
        and (root / "tokenizer_config.json").exists()
        and any((root / name).exists() for name in _MODEL_WEIGHT_FILENAMES)
        and any((root / name).exists() for name in _TOKENIZER_FILENAMES)
    )


def resolve_google_drive_file_id(download_source: str | None) -> str | None:
    text = str(download_source or "").strip()
    if not text:
        return None
    if _GOOGLE_DRIVE_FILE_ID_PATTERN.fullmatch(text):
        return text

    values = parse_qs(urlparse(text).query).get("id", [])
    if values and _GOOGLE_DRIVE_FILE_ID_PATTERN.fullmatch(values[0].strip()):
        return values[0].strip()

    match = re.search(r"/d/([A-Za-z0-9_-]{10,})", urlparse(text).path)
    if match is not None:
        return match.group(1)

    raise ValueError("Expected a Google Drive file id or share URL.")


def _find_checkpoint_root(extracted_root: Path) -> Path | None:
    if checkpoint_artifact_is_ready(extracted_root):
        return extracted_root
    for child in sorted(extracted_root.iterdir()):
        if child.is_dir() and checkpoint_artifact_is_ready(child):
            return child
    return None


def extract_checkpoint_archive(zip_path: str | Path, extract_dir: str | Path) -> Path:
    archive_path = Path(zip_path)
    destination_root = Path(extract_dir)
    staging_root = destination_root.parent / f".{destination_root.name}.extracting"

    if not archive_path.exists():
        raise FileNotFoundError(f"Missing checkpoint archive: {archive_path}")

    if staging_root.exists():
        shutil.rmtree(staging_root)
    ensure_dir(staging_root)

    try:
        with ZipFile(archive_path, "r") as archive:
            archive.extractall(staging_root)

        source_root = _find_checkpoint_root(staging_root)
        if source_root is None:
            raise RuntimeError("The extracted archive does not contain a valid checkpoint bundle.")

        if destination_root.exists():
            shutil.rmtree(destination_root)

        shutil.move(str(source_root), str(destination_root))
    finally:
        if staging_root.exists():
            shutil.rmtree(staging_root, ignore_errors=True)

    return destination_root


def build_ppo_checkpoint_prep_command(
    *,
    repo_dir: str | Path,
    config_path: str | Path,
    checkpoint_download_source: str | None = None,
) -> tuple[str, list[str] | None, str | None]:
    repo_root = Path(repo_dir).expanduser().resolve()
    resolved_config_path = resolve_path(config_path, repo_root)
    config = load_yaml(resolved_config_path)
    checkpoint_value = config.get("model", {}).get("checkpoint")
    if not checkpoint_value:
        return "missing_config_checkpoint", None, None
    if looks_like_remote_model_identifier(checkpoint_value):
        return "remote_model", None, str(checkpoint_value)

    extract_dir = resolve_path(checkpoint_value, repo_root)
    if checkpoint_artifact_is_ready(extract_dir):
        return "existing_local_checkpoint", None, str(extract_dir)
    if not str(checkpoint_download_source or "").strip():
        return "missing_download_source", None, str(extract_dir)

    return (
        "managed_download",
        [
            sys.executable,
            str(repo_root / "scripts" / "download_ppo_checkpoint.py"),
            "--download-source",
            str(checkpoint_download_source).strip(),
            "--zip-path",
            str(default_checkpoint_archive_path(extract_dir)),
            "--extract-dir",
            str(extract_dir),
            "--skip-existing",
        ],
        str(extract_dir),
    )
