from __future__ import annotations

import importlib.util
from pathlib import Path

from colab.thesis_colab_support import get_bootstrap_environment as get_colab_environment
from kaggle.thesis_kaggle_support import get_bootstrap_environment as get_kaggle_environment
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
