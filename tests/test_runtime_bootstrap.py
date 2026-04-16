from __future__ import annotations

import importlib.util
from io import BytesIO
from pathlib import Path
from zipfile import ZipFile

from colab.thesis_colab_support import get_bootstrap_environment as get_colab_environment
from kaggle.thesis_kaggle_support import get_bootstrap_environment as get_kaggle_environment
from src.checkpoint_bootstrap import (
    archive_checkpoint_directory,
    build_ppo_checkpoint_prep_command,
    checkpoint_artifact_is_ready,
    extract_checkpoint_archive,
    resolve_google_drive_file_id,
)
from src.runtime_bootstrap import (
    DEFAULT_TRAIN_DATASET_FILE_ID,
    STAGE_SPECS,
    build_dataset_prep_command,
    infer_managed_dataset,
    resolve_stage_config_path,
)


def _write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _make_repo(tmp_path: Path) -> Path:
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir(parents=True)
    _write_text(
        repo_dir / "configs" / "sft_chebi20.yaml",
        "\n".join(
            [
                "data:",
                "  train_file: data/chebi20/processed/train.jsonl",
                "  validation_file: data/chebi20/processed/validation.jsonl",
                "  test_file: data/chebi20/processed/test.jsonl",
            ]
        )
        + "\n",
    )
    _write_text(
        repo_dir / "configs" / "multi_molecule_sft_mini.yaml",
        "\n".join(
            [
                "data:",
                "  train_file: data/mini_post_training/post_training_processed/train_multimol.jsonl",
                "  validation_file: data/mini_post_training/grouped_splits/validation_multimol.jsonl",
                "  test_file: data/mini_post_training/grouped_splits/test_multimol.jsonl",
            ]
        )
        + "\n",
    )
    _write_text(
        repo_dir / "configs" / "molecule_wise_ppo_mini.yaml",
        "\n".join(
            [
                "model:",
                "  checkpoint: outputs/multi_molecule_sft_mini/checkpoints/best",
                "data:",
                "  train_file: data/mini_post_training/post_training_processed/train_multimol.jsonl",
                "  validation_file: data/mini_post_training/grouped_splits/validation_multimol.jsonl",
                "  test_file: data/mini_post_training/grouped_splits/test_multimol.jsonl",
            ]
        )
        + "\n",
    )
    return repo_dir


def test_environment_helpers_report_expected_names() -> None:
    assert get_colab_environment().name == "colab"
    assert get_kaggle_environment().name == "kaggle"


def test_new_bootstrap_scripts_import_cleanly() -> None:
    project_root = Path(__file__).resolve().parents[1]
    for relative_path in (
        Path("scripts") / "download_train_dataset.py",
        Path("scripts") / "download_ppo_checkpoint.py",
        Path("scripts") / "init_colab.py",
        Path("scripts") / "init_kaggle.py",
    ):
        script_path = project_root / relative_path
        spec = importlib.util.spec_from_file_location(relative_path.stem, script_path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)


def test_resolve_stage_config_path_uses_stage_default(tmp_path: Path) -> None:
    repo_dir = _make_repo(tmp_path)

    resolved = resolve_stage_config_path(repo_dir, STAGE_SPECS["multi_sft"])

    assert resolved == (repo_dir / "configs" / "multi_molecule_sft_mini.yaml").resolve()


def test_stage_specs_use_expected_training_scripts() -> None:
    assert STAGE_SPECS["sft"].training_script == Path("scripts") / "train_sft.py"
    assert STAGE_SPECS["multi_sft"].training_script == Path("scripts") / "train_multi_molecule_sft.py"
    assert STAGE_SPECS["ppo"].training_script == Path("scripts") / "train_molecule_wise_ppo.py"


def test_infer_managed_dataset_detects_chebi_sft(tmp_path: Path) -> None:
    repo_dir = _make_repo(tmp_path)
    config_path = repo_dir / "configs" / "sft_chebi20.yaml"

    dataset_kind = infer_managed_dataset(
        stage_spec=STAGE_SPECS["sft"],
        config_path=config_path,
        repo_dir=repo_dir,
    )

    assert dataset_kind == "chebi20"


def test_infer_managed_dataset_detects_mini_post_training(tmp_path: Path) -> None:
    repo_dir = _make_repo(tmp_path)
    config_path = repo_dir / "configs" / "multi_molecule_sft_mini.yaml"

    dataset_kind = infer_managed_dataset(
        stage_spec=STAGE_SPECS["multi_sft"],
        config_path=config_path,
        repo_dir=repo_dir,
    )

    assert dataset_kind == "mini_post_training"


def test_build_dataset_prep_command_skips_when_mini_dataset_ready(tmp_path: Path) -> None:
    repo_dir = _make_repo(tmp_path)
    _write_text(repo_dir / "data" / "mini_post_training" / "post_training_processed" / "train_multimol.jsonl", "{}\n")
    _write_text(repo_dir / "data" / "mini_post_training" / "grouped_splits" / "train_multimol.jsonl", "{}\n")
    _write_text(repo_dir / "data" / "mini_post_training" / "grouped_splits" / "validation_multimol.jsonl", "{}\n")
    _write_text(repo_dir / "data" / "mini_post_training" / "grouped_splits" / "test_multimol.jsonl", "{}\n")

    dataset_kind, command = build_dataset_prep_command(
        repo_dir=repo_dir,
        stage_spec=STAGE_SPECS["multi_sft"],
        config_path=repo_dir / "configs" / "multi_molecule_sft_mini.yaml",
        dataset_mode="auto",
        train_dataset_file_id=DEFAULT_TRAIN_DATASET_FILE_ID,
    )

    assert dataset_kind == "mini_post_training"
    assert command is None


def test_build_dataset_prep_command_builds_google_drive_download_when_needed(tmp_path: Path) -> None:
    repo_dir = _make_repo(tmp_path)

    dataset_kind, command = build_dataset_prep_command(
        repo_dir=repo_dir,
        stage_spec=STAGE_SPECS["ppo"],
        config_path=repo_dir / "configs" / "molecule_wise_ppo_mini.yaml",
        dataset_mode="auto",
        train_dataset_file_id="custom-id",
    )

    assert dataset_kind == "mini_post_training"
    assert command is not None
    assert command[1].endswith("scripts/download_train_dataset.py")
    assert "--file-id" in command
    assert "custom-id" in command
    assert "--skip-existing" in command


def test_build_dataset_prep_command_builds_chebi_download_for_sft(tmp_path: Path) -> None:
    repo_dir = _make_repo(tmp_path)

    dataset_kind, command = build_dataset_prep_command(
        repo_dir=repo_dir,
        stage_spec=STAGE_SPECS["sft"],
        config_path=repo_dir / "configs" / "sft_chebi20.yaml",
        dataset_mode="auto",
    )

    assert dataset_kind == "chebi20"
    assert command is not None
    assert command[1].endswith("scripts/download_chebi20.py")
    assert command[-1].endswith("data/chebi20")


def _build_checkpoint_zip_bytes() -> bytes:
    buffer = BytesIO()
    with ZipFile(buffer, "w") as archive:
        archive.writestr("best/config.json", "{}\n")
        archive.writestr("best/tokenizer_config.json", "{}\n")
        archive.writestr("best/pytorch_model.bin", "weights")
        archive.writestr("best/spiece.model", "tokenizer")
    return buffer.getvalue()


def test_build_ppo_checkpoint_prep_command_skips_when_source_missing(tmp_path: Path) -> None:
    repo_dir = _make_repo(tmp_path)

    checkpoint_kind, command, checkpoint_target = build_ppo_checkpoint_prep_command(
        repo_dir=repo_dir,
        config_path=repo_dir / "configs" / "molecule_wise_ppo_mini.yaml",
        checkpoint_download_source=None,
    )

    assert checkpoint_kind == "missing_download_source"
    assert command is None
    assert checkpoint_target is not None and checkpoint_target.endswith("outputs/multi_molecule_sft_mini/checkpoints/best")


def test_build_ppo_checkpoint_prep_command_builds_download_when_needed(tmp_path: Path) -> None:
    repo_dir = _make_repo(tmp_path)

    checkpoint_kind, command, checkpoint_target = build_ppo_checkpoint_prep_command(
        repo_dir=repo_dir,
        config_path=repo_dir / "configs" / "molecule_wise_ppo_mini.yaml",
        checkpoint_download_source="drive-id-12345",
    )

    assert checkpoint_kind == "managed_download"
    assert checkpoint_target is not None and checkpoint_target.endswith("outputs/multi_molecule_sft_mini/checkpoints/best")
    assert command is not None
    assert command[1].endswith("scripts/download_ppo_checkpoint.py")
    assert "--download-source" in command
    assert "drive-id-12345" in command
    assert "--skip-existing" in command


def test_resolve_google_drive_file_id_accepts_share_url() -> None:
    resolved = resolve_google_drive_file_id(
        "https://drive.google.com/file/d/1AbCdEfGhIJkLmNoP/view?usp=sharing"
    )

    assert resolved == "1AbCdEfGhIJkLmNoP"


def test_extract_ppo_checkpoint_archive_normalizes_nested_zip_root(tmp_path: Path) -> None:
    archive_path = tmp_path / "checkpoint.zip"
    archive_path.write_bytes(_build_checkpoint_zip_bytes())

    destination = extract_checkpoint_archive(archive_path, tmp_path / "best")

    assert destination == tmp_path / "best"
    assert checkpoint_artifact_is_ready(destination) is True


def test_archive_checkpoint_directory_writes_zip_bundle(tmp_path: Path) -> None:
    checkpoint_dir = tmp_path / "best"
    checkpoint_dir.mkdir(parents=True)
    (checkpoint_dir / "config.json").write_text("{}\n", encoding="utf-8")
    (checkpoint_dir / "tokenizer.json").write_text("ok\n", encoding="utf-8")

    archive_path = archive_checkpoint_directory(checkpoint_dir)

    assert archive_path.exists()
    with ZipFile(archive_path, "r") as archive:
        assert sorted(archive.namelist()) == [
            "best/config.json",
            "best/tokenizer.json",
        ]

