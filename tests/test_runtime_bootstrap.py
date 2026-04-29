from __future__ import annotations

import importlib.util
import sys
from io import BytesIO
from pathlib import Path
from zipfile import ZipFile

import pytest

from colab.thesis_colab_support import get_bootstrap_environment as get_colab_environment
from kaggle.thesis_kaggle_support import get_bootstrap_environment as get_kaggle_environment
from src.checkpoint_bootstrap import (
    archive_directory_to_zip,
    archive_checkpoint_directory,
    build_gflownet_checkpoint_prep_command,
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


def _load_script_module(relative_path: Path):
    project_root = Path(__file__).resolve().parents[1]
    script_path = project_root / relative_path
    spec = importlib.util.spec_from_file_location(relative_path.stem, script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


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
    _write_text(
        repo_dir / "configs" / "multi_molecule_gflownet_mini.yaml",
        "\n".join(
            [
                "model:",
                "  checkpoint: outputs/multi_molecule_sft_mini/checkpoints/best",
                "data:",
                "  train_file: data/mini_post_training/grouped_splits/train_multimol.jsonl",
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
        Path("scripts") / "prepare_gflownet_restart_checkpoint.py",
        Path("scripts") / "init_colab.py",
        Path("scripts") / "init_kaggle.py",
        Path("scripts") / "train_multi_molecule_gflownet.py",
    ):
        script_path = project_root / relative_path
        if relative_path.name == "train_multi_molecule_gflownet.py":
            pytest.importorskip("transformers")
        spec = importlib.util.spec_from_file_location(relative_path.stem, script_path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)


def test_train_multi_molecule_sft_script_applies_output_dir_override_before_path_resolution(
    monkeypatch,
    tmp_path: Path,
    capsys,
) -> None:
    pytest.importorskip("transformers")
    module = _load_script_module(Path("scripts") / "train_multi_molecule_sft.py")
    resolve_inputs: list[dict[str, object]] = []
    run_inputs: list[dict[str, object]] = []

    monkeypatch.setattr(
        module,
        "load_yaml",
        lambda _path: {"seed": 42, "training": {"output_dir": "outputs/default_sft"}},
    )

    def _fake_resolve(config, project_root=None):
        resolve_inputs.append(config)
        return config

    def _fake_run(config):
        run_inputs.append(config)
        return {"output_dir": config["training"]["output_dir"]}

    monkeypatch.setattr(module, "resolve_multi_molecule_sft_config_paths", _fake_resolve)
    monkeypatch.setattr(module, "run_multi_molecule_sft", _fake_run)
    monkeypatch.setattr(module, "set_seed", lambda _seed: None)
    monkeypatch.setattr(
        module.sys,
        "argv",
        [
            "train_multi_molecule_sft.py",
            "--config",
            str(tmp_path / "config.yaml"),
            "--output-dir",
            "/content/drive/MyDrive/my_results",
        ],
    )

    module.main()

    assert resolve_inputs == [
        {"seed": 42, "training": {"output_dir": "/content/drive/MyDrive/my_results"}}
    ]
    assert run_inputs == resolve_inputs
    assert "/content/drive/MyDrive/my_results" in capsys.readouterr().out


def test_train_multi_molecule_gflownet_script_applies_relative_output_dir_override_before_path_resolution(
    monkeypatch,
    tmp_path: Path,
    capsys,
) -> None:
    pytest.importorskip("transformers")
    module = _load_script_module(Path("scripts") / "train_multi_molecule_gflownet.py")
    resolve_inputs: list[dict[str, object]] = []
    run_inputs: list[dict[str, object]] = []

    monkeypatch.setattr(
        module,
        "load_yaml",
        lambda _path: {
            "seed": 123,
            "training": {"output_dir": "outputs/default_gflownet"},
            "model": {"checkpoint": "outputs/multi_molecule_sft_lpm24/checkpoints/best"},
        },
    )

    def _fake_resolve(config, project_root=None):
        resolve_inputs.append(config)
        return config

    def _fake_run(config):
        run_inputs.append(config)
        return {"output_dir": config["training"]["output_dir"]}

    monkeypatch.setattr(module, "resolve_gflownet_config_paths", _fake_resolve)
    monkeypatch.setattr(module, "run_multi_molecule_gflownet", _fake_run)
    monkeypatch.setattr(module, "set_seed", lambda _seed: None)
    monkeypatch.setattr(
        module.sys,
        "argv",
        [
            "train_multi_molecule_gflownet.py",
            "--config",
            str(tmp_path / "config.yaml"),
            "--output-dir",
            "outputs/custom_gflownet_run",
        ],
    )

    module.main()

    assert resolve_inputs == [
        {
            "seed": 123,
            "training": {"output_dir": "outputs/custom_gflownet_run"},
            "model": {"checkpoint": "outputs/multi_molecule_sft_lpm24/checkpoints/best"},
        }
    ]
    assert run_inputs == resolve_inputs
    assert "outputs/custom_gflownet_run" in capsys.readouterr().out


def test_resolve_stage_config_path_uses_stage_default(tmp_path: Path) -> None:
    repo_dir = _make_repo(tmp_path)

    resolved = resolve_stage_config_path(repo_dir, STAGE_SPECS["multi_sft"])

    assert resolved == (repo_dir / "configs" / "multi_molecule_sft_mini.yaml").resolve()


def test_resolve_stage_config_path_uses_gflownet_stage_default(tmp_path: Path) -> None:
    repo_dir = _make_repo(tmp_path)

    resolved = resolve_stage_config_path(repo_dir, STAGE_SPECS["gflownet"])

    assert resolved == (repo_dir / "configs" / "multi_molecule_gflownet_mini.yaml").resolve()


def test_stage_specs_use_expected_training_scripts() -> None:
    assert STAGE_SPECS["sft"].training_script == Path("scripts") / "train_sft.py"
    assert STAGE_SPECS["multi_sft"].training_script == Path("scripts") / "train_multi_molecule_sft.py"
    assert STAGE_SPECS["ppo"].training_script == Path("scripts") / "train_molecule_wise_ppo.py"
    assert STAGE_SPECS["gflownet"].training_script == Path("scripts") / "train_multi_molecule_gflownet.py"
    assert STAGE_SPECS["gflownet"].default_config == Path("configs") / "multi_molecule_gflownet_mini.yaml"


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


def test_build_dataset_prep_command_builds_google_drive_download_for_gflownet_when_needed(
    tmp_path: Path,
) -> None:
    repo_dir = _make_repo(tmp_path)

    dataset_kind, command = build_dataset_prep_command(
        repo_dir=repo_dir,
        stage_spec=STAGE_SPECS["gflownet"],
        config_path=repo_dir / "configs" / "multi_molecule_gflownet_mini.yaml",
        dataset_mode="auto",
        train_dataset_file_id="gflownet-id",
    )

    assert dataset_kind == "mini_post_training"
    assert command is not None
    assert command[1].endswith("scripts/download_train_dataset.py")
    assert "--file-id" in command
    assert "gflownet-id" in command
    assert "--skip-existing" in command


def test_colab_bootstrap_scripts_prepare_gflownet_reports_missing_checkpoint_download_source(
    monkeypatch,
    tmp_path: Path,
) -> None:
    module = _load_script_module(Path("scripts") / "init_colab.py")
    repo_dir = _make_repo(tmp_path)
    config_path = repo_dir / "configs" / "multi_molecule_gflownet_mini.yaml"
    checkpoint_target = repo_dir / "outputs" / "multi_molecule_sft_mini" / "checkpoints" / "best"
    events: list[tuple[str, dict[str, object]]] = []
    commands: list[tuple[list[str], Path | None]] = []
    builder_calls: list[dict[str, object]] = []

    monkeypatch.setattr(module, "get_bootstrap_environment", lambda _repo_dir: object())
    monkeypatch.setattr(module, "summarize_environment", lambda _environment: {"environment": "test"})
    monkeypatch.setattr(module, "json_dumps", lambda payload: str(payload))
    monkeypatch.setattr(module, "clone_or_update_repo", lambda **_kwargs: repo_dir)
    monkeypatch.setattr(module, "install_repo_requirements", lambda _repo_dir: None)
    monkeypatch.setattr(module, "resolve_stage_config_path", lambda *_args, **_kwargs: config_path)
    monkeypatch.setattr(
        module,
        "build_dataset_prep_command",
        lambda **_kwargs: (
            "mini_post_training",
            [sys.executable, str(repo_dir / "scripts" / "download_train_dataset.py"), "--skip-existing"],
        ),
    )
    monkeypatch.setattr(module, "build_ppo_checkpoint_prep_command", lambda **_kwargs: ("not_applicable", None, None))

    def _fake_builder(**kwargs):
        builder_calls.append(kwargs)
        return ("missing_download_source", None, str(checkpoint_target))

    monkeypatch.setattr(module, "build_gflownet_checkpoint_prep_command", _fake_builder)
    monkeypatch.setattr(
        module,
        "run_command",
        lambda command, *, cwd=None, env=None: commands.append((list(command), Path(cwd) if cwd else None)),
    )
    monkeypatch.setattr(
        module,
        "print_json_status",
        lambda event, **payload: events.append((event, payload)),
    )
    monkeypatch.setattr(module.sys, "argv", ["init_colab.py", "--stage", "gflownet", "--repo-dir", str(repo_dir)])

    module.main()

    assert builder_calls == [
        {
            "repo_dir": repo_dir,
            "config_path": config_path,
            "checkpoint_download_source": None,
        }
    ]
    assert commands == [
        (
            [
                sys.executable,
                str(repo_dir / "scripts" / "download_train_dataset.py"),
                "--skip-existing",
            ],
            repo_dir,
        )
    ]
    assert any(
        event == "checkpoint_plan"
        and payload["stage"] == "gflownet"
        and payload["checkpoint_kind"] == "missing_download_source"
        and payload["checkpoint_target"] == str(checkpoint_target)
        and payload["checkpoint_download_source_provided"] is False
        for event, payload in events
    )


def test_colab_bootstrap_scripts_prepare_gflownet_builds_managed_checkpoint_download(
    monkeypatch,
    tmp_path: Path,
) -> None:
    module = _load_script_module(Path("scripts") / "init_colab.py")
    repo_dir = _make_repo(tmp_path)
    config_path = repo_dir / "configs" / "multi_molecule_gflownet_mini.yaml"
    checkpoint_target = repo_dir / "outputs" / "multi_molecule_sft_mini" / "checkpoints" / "best"
    checkpoint_command = [
        sys.executable,
        str(repo_dir / "scripts" / "download_ppo_checkpoint.py"),
        "--download-source",
        "drive-id-12345",
        "--zip-path",
        str(checkpoint_target.with_suffix(".zip")),
        "--extract-dir",
        str(checkpoint_target),
        "--skip-existing",
    ]
    events: list[tuple[str, dict[str, object]]] = []
    commands: list[tuple[list[str], Path | None]] = []
    builder_calls: list[dict[str, object]] = []

    monkeypatch.setattr(module, "get_bootstrap_environment", lambda _repo_dir: object())
    monkeypatch.setattr(module, "summarize_environment", lambda _environment: {"environment": "test"})
    monkeypatch.setattr(module, "json_dumps", lambda payload: str(payload))
    monkeypatch.setattr(module, "clone_or_update_repo", lambda **_kwargs: repo_dir)
    monkeypatch.setattr(module, "install_repo_requirements", lambda _repo_dir: None)
    monkeypatch.setattr(module, "resolve_stage_config_path", lambda *_args, **_kwargs: config_path)
    monkeypatch.setattr(
        module,
        "build_dataset_prep_command",
        lambda **_kwargs: (
            "mini_post_training",
            [sys.executable, str(repo_dir / "scripts" / "download_train_dataset.py"), "--skip-existing"],
        ),
    )
    monkeypatch.setattr(module, "build_ppo_checkpoint_prep_command", lambda **_kwargs: ("not_applicable", None, None))

    def _fake_builder(**kwargs):
        builder_calls.append(kwargs)
        return ("managed_download", checkpoint_command, str(checkpoint_target))

    monkeypatch.setattr(module, "build_gflownet_checkpoint_prep_command", _fake_builder)
    monkeypatch.setattr(
        module,
        "run_command",
        lambda command, *, cwd=None, env=None: commands.append((list(command), Path(cwd) if cwd else None)),
    )
    monkeypatch.setattr(
        module,
        "print_json_status",
        lambda event, **payload: events.append((event, payload)),
    )
    monkeypatch.setattr(
        module.sys,
        "argv",
        [
            "init_colab.py",
            "--stage",
            "gflownet",
            "--repo-dir",
            str(repo_dir),
            "--gflownet-checkpoint-download-source",
            "drive-id-12345",
        ],
    )

    module.main()

    assert builder_calls == [
        {
            "repo_dir": repo_dir,
            "config_path": config_path,
            "checkpoint_download_source": "drive-id-12345",
        }
    ]
    assert commands == [
        (
            [
                sys.executable,
                str(repo_dir / "scripts" / "download_train_dataset.py"),
                "--skip-existing",
            ],
            repo_dir,
        ),
        (checkpoint_command, repo_dir),
    ]
    assert any(
        event == "checkpoint_plan"
        and payload["stage"] == "gflownet"
        and payload["checkpoint_kind"] == "managed_download"
        and payload["checkpoint_target"] == str(checkpoint_target)
        and payload["checkpoint_download_source_provided"] is True
        for event, payload in events
    )


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


def test_build_gflownet_checkpoint_prep_command_skips_when_source_missing(tmp_path: Path) -> None:
    repo_dir = _make_repo(tmp_path)

    checkpoint_kind, command, checkpoint_target = build_gflownet_checkpoint_prep_command(
        repo_dir=repo_dir,
        config_path=repo_dir / "configs" / "multi_molecule_gflownet_mini.yaml",
        checkpoint_download_source=None,
    )

    assert checkpoint_kind == "missing_download_source"
    assert command is None
    assert checkpoint_target is not None and checkpoint_target.endswith("outputs/multi_molecule_sft_mini/checkpoints/best")


def test_build_gflownet_checkpoint_prep_command_builds_download_when_needed(tmp_path: Path) -> None:
    repo_dir = _make_repo(tmp_path)

    checkpoint_kind, command, checkpoint_target = build_gflownet_checkpoint_prep_command(
        repo_dir=repo_dir,
        config_path=repo_dir / "configs" / "multi_molecule_gflownet_mini.yaml",
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


def test_archive_directory_to_zip_can_write_zip_inside_source_dir_without_self_inclusion(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "my_results"
    output_dir.mkdir(parents=True)
    (output_dir / "run_summary.json").write_text("{}\n", encoding="utf-8")
    archive_path = output_dir / "my_results.zip"
    archive_path.write_text("stale-archive", encoding="utf-8")

    first_archive = archive_directory_to_zip(output_dir, archive_path)

    assert first_archive == archive_path
    assert first_archive.exists()
    with ZipFile(first_archive, "r") as archive:
        assert sorted(archive.namelist()) == ["my_results/run_summary.json"]

    (output_dir / "checkpoints").mkdir()
    (output_dir / "checkpoints" / "best.zip").write_text("checkpoint", encoding="utf-8")

    second_archive = archive_directory_to_zip(output_dir, archive_path)

    assert second_archive == archive_path
    with ZipFile(second_archive, "r") as archive:
        assert sorted(archive.namelist()) == [
            "my_results/checkpoints/best.zip",
            "my_results/run_summary.json",
        ]
