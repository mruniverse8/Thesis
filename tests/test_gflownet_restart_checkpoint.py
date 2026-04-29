from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from zipfile import ZipFile

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _load_restart_module():
    script_path = PROJECT_ROOT / "scripts" / "prepare_gflownet_restart_checkpoint.py"
    spec = importlib.util.spec_from_file_location(script_path.stem, script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _build_checkpoint_zip(
    archive_path: Path,
    *,
    iteration_metrics_jsonl: str,
    extra_json_files: dict[str, object] | None = None,
) -> None:
    with ZipFile(archive_path, "w") as archive:
        archive.writestr("best/config.json", "{}\n")
        archive.writestr("best/tokenizer_config.json", "{}\n")
        archive.writestr("best/pytorch_model.bin", "weights")
        archive.writestr("best/spiece.model", "tokenizer")
        archive.writestr("best/iteration_metrics.jsonl", iteration_metrics_jsonl)
        for relative_path, payload in (extra_json_files or {}).items():
            archive.writestr(f"best/{relative_path}", json.dumps(payload))


def test_prepare_gflownet_restart_checkpoint_extracts_nested_zip_and_reads_jsonl_metadata(
    tmp_path: Path,
) -> None:
    module = _load_restart_module()
    archive_path = tmp_path / "best.zip"
    extract_dir = tmp_path / "extracted" / "best"
    _build_checkpoint_zip(
        archive_path,
        iteration_metrics_jsonl="\n".join(
            [
                json.dumps({"iteration": 1499, "learning_rate": 2.0e-6}),
                "not json",
                json.dumps({"iteration": 1500, "learning_rate": 1.0e-6}),
            ]
        ),
    )

    payload = module.prepare_restart_checkpoint(
        download_source=None,
        zip_path=archive_path,
        extract_dir=extract_dir,
        skip_existing=False,
    )

    assert payload["status"] == "ready"
    assert payload["downloaded"] is False
    assert payload["checkpoint_dir"] == str(extract_dir.resolve())
    assert payload["restart_iteration"] == 1500
    assert payload["next_iteration"] == 1501
    assert payload["learning_rate"] == pytest.approx(1.0e-6)
    assert payload["iteration_source_path"].endswith("iteration_metrics.jsonl")
    assert payload["learning_rate_source_path"].endswith("iteration_metrics.jsonl")


def test_prepare_gflownet_restart_checkpoint_falls_back_to_json_learning_rate(
    tmp_path: Path,
) -> None:
    module = _load_restart_module()
    archive_path = tmp_path / "best.zip"
    extract_dir = tmp_path / "best"
    _build_checkpoint_zip(
        archive_path,
        iteration_metrics_jsonl=json.dumps({"iteration": 1500}) + "\n",
        extra_json_files={"iteration_metrics.json": {"learning_rate": 1.0e-6}},
    )

    payload = module.prepare_restart_checkpoint(
        download_source=None,
        zip_path=archive_path,
        extract_dir=extract_dir,
        skip_existing=False,
    )

    assert payload["restart_iteration"] == 1500
    assert payload["next_iteration"] == 1501
    assert payload["learning_rate"] == pytest.approx(1.0e-6)
    assert payload["learning_rate_source_path"].endswith("iteration_metrics.json")


def test_prepare_gflownet_restart_checkpoint_fails_when_no_iteration_is_found(
    tmp_path: Path,
) -> None:
    module = _load_restart_module()
    archive_path = tmp_path / "best.zip"
    _build_checkpoint_zip(
        archive_path,
        iteration_metrics_jsonl=json.dumps({"learning_rate": 1.0e-6}) + "\n",
    )

    with pytest.raises(ValueError, match="No restart iteration found"):
        module.prepare_restart_checkpoint(
            download_source=None,
            zip_path=archive_path,
            extract_dir=tmp_path / "best",
            skip_existing=False,
        )
