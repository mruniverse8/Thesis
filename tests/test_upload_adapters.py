from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from zipfile import ZipFile

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _load_upload_module():
    script_path = PROJECT_ROOT / "scripts" / "upload_adapters.py"
    spec = importlib.util.spec_from_file_location(script_path.stem, script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _build_full_checkpoint_zip(
    archive_path: Path,
    *,
    iteration_metrics_jsonl: str,
    extra_json_files: dict[str, object] | None = None,
    root_prefix: str = "best",
) -> None:
    with ZipFile(archive_path, "w") as archive:
        archive.writestr(f"{root_prefix}/config.json", "{}\n")
        archive.writestr(f"{root_prefix}/tokenizer_config.json", "{}\n")
        archive.writestr(f"{root_prefix}/pytorch_model.bin", "weights")
        archive.writestr(f"{root_prefix}/spiece.model", "tokenizer")
        archive.writestr(f"{root_prefix}/iteration_metrics.jsonl", iteration_metrics_jsonl)
        for relative_path, payload in (extra_json_files or {}).items():
            archive.writestr(f"{root_prefix}/{relative_path}", json.dumps(payload))


def _build_adapter_bundle_zip(
    archive_path: Path,
    *,
    checkpoint_dir_text: str,
    iteration_metrics_json: dict[str, object],
    root_prefix: str = "best",
) -> None:
    with ZipFile(archive_path, "w") as archive:
        archive.writestr(
            f"{root_prefix}/adapter_config.json",
            json.dumps({"base_model_name_or_path": checkpoint_dir_text}),
        )
        archive.writestr(f"{root_prefix}/adapter_model.safetensors", "adapter-weights")
        archive.writestr(f"{root_prefix}/tokenizer_config.json", "{}\n")
        archive.writestr(f"{root_prefix}/tokenizer.json", "{}\n")
        archive.writestr(f"{root_prefix}/flow_head.pt", "flow-head")
        archive.writestr(f"{root_prefix}/iteration_metrics.json", json.dumps(iteration_metrics_json))
        archive.writestr(f"{root_prefix}/metrics.json", json.dumps(iteration_metrics_json))
        archive.writestr(
            f"{root_prefix}/training_config.json",
            json.dumps({"learning_rate": iteration_metrics_json["learning_rate"]}),
        )


class _DummyMergedModel:
    def save_pretrained(self, destination: str | Path) -> None:
        root = Path(destination)
        root.mkdir(parents=True, exist_ok=True)
        (root / "config.json").write_text("{}", encoding="utf-8")
        (root / "pytorch_model.bin").write_text("weights", encoding="utf-8")


class _DummyPeftWrapper:
    def merge_and_unload(self) -> _DummyMergedModel:
        return _DummyMergedModel()


class _DummyPeftModel:
    @staticmethod
    def from_pretrained(base_model, adapter_dir: str) -> _DummyPeftWrapper:
        assert Path(adapter_dir, "adapter_model.safetensors").exists()
        return _DummyPeftWrapper()


class _DummyBaseModel:
    def __init__(self, source: str) -> None:
        self.source = source


class _DummyT5ForConditionalGeneration:
    @staticmethod
    def from_pretrained(source: str, local_files_only: bool = False) -> _DummyBaseModel:
        return _DummyBaseModel(source)


class _DummyTokenizer:
    def save_pretrained(self, destination: str | Path) -> None:
        root = Path(destination)
        root.mkdir(parents=True, exist_ok=True)
        (root / "tokenizer_config.json").write_text("{}", encoding="utf-8")
        (root / "tokenizer.json").write_text("{}", encoding="utf-8")


class _DummyAutoTokenizer:
    @staticmethod
    def from_pretrained(source: str | Path, use_fast: bool = True, local_files_only: bool = False) -> _DummyTokenizer:
        return _DummyTokenizer()


def test_upload_adapters_prepares_full_checkpoint_bundle(tmp_path: Path) -> None:
    module = _load_upload_module()
    archive_path = tmp_path / "best.zip"
    extract_dir = tmp_path / "prepared" / "best"
    _build_full_checkpoint_zip(
        archive_path,
        iteration_metrics_jsonl=json.dumps({"iteration": 1500, "learning_rate": 1.0e-6}) + "\n",
    )

    payload = module.prepare_uploaded_checkpoint(
        download_source=None,
        zip_path=archive_path,
        extract_dir=extract_dir,
        skip_existing=False,
        fallback_learning_rate=1e-6,
    )

    assert payload["status"] == "ready"
    assert payload["artifact_kind"] == "full_checkpoint"
    assert payload["reused_existing"] is False
    assert payload["restart_iteration"] == 1500
    assert payload["next_iteration"] == 1501
    assert payload["learning_rate"] == pytest.approx(1.0e-6)
    assert (extract_dir / "restart_source_metadata.json").exists()


def test_upload_adapters_merges_adapter_bundle_and_preserves_restart_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_upload_module()
    archive_path = tmp_path / "best.zip"
    extract_dir = tmp_path / "prepared" / "best"
    upstream_checkpoint = tmp_path / "upstream" / "best"
    upstream_checkpoint.mkdir(parents=True)
    _build_adapter_bundle_zip(
        archive_path,
        checkpoint_dir_text=str(extract_dir),
        iteration_metrics_json={"iteration": 2765.0, "learning_rate": 1.0e-6},
    )

    monkeypatch.setattr(
        module,
        "_load_merge_dependencies",
        lambda: (_DummyAutoTokenizer, _DummyPeftModel, _DummyT5ForConditionalGeneration),
    )

    payload = module.prepare_uploaded_checkpoint(
        download_source=None,
        zip_path=archive_path,
        extract_dir=extract_dir,
        upstream_checkpoint=str(upstream_checkpoint),
        default_base_model="remote/base-model",
        skip_existing=False,
        fallback_learning_rate=1e-6,
    )

    assert payload["status"] == "ready"
    assert payload["artifact_kind"] == "adapter_bundle"
    assert payload["reused_existing"] is False
    assert payload["restart_iteration"] == 2765
    assert payload["next_iteration"] == 2766
    assert payload["resolved_merge_base"] == str(upstream_checkpoint.resolve())
    assert payload["resolved_merge_base_source"] == "upstream_checkpoint"
    assert (extract_dir / "config.json").exists()
    assert (extract_dir / "pytorch_model.bin").exists()
    assert (extract_dir / "flow_head.pt").exists()
    assert (extract_dir / "iteration_metrics.jsonl").exists()
    assert (extract_dir / "restart_source_metadata.json").exists()


def test_upload_adapters_reuses_existing_checkpoint_only_when_source_matches(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_upload_module()
    archive_path = tmp_path / "best.zip"
    extract_dir = tmp_path / "prepared" / "best"
    extract_dir.mkdir(parents=True)
    (extract_dir / "config.json").write_text("{}", encoding="utf-8")
    (extract_dir / "tokenizer_config.json").write_text("{}", encoding="utf-8")
    (extract_dir / "tokenizer.json").write_text("{}", encoding="utf-8")
    (extract_dir / "pytorch_model.bin").write_text("weights", encoding="utf-8")
    (extract_dir / "iteration_metrics.jsonl").write_text(
        json.dumps({"iteration": 1536.0, "learning_rate": 1.0e-6}) + "\n",
        encoding="utf-8",
    )
    (extract_dir / "restart_source_metadata.json").write_text(
        json.dumps(
            {
                "download_source": "old-id",
                "zip_path": str(archive_path),
                "artifact_kind": "full_checkpoint",
            }
        ),
        encoding="utf-8",
    )
    _build_full_checkpoint_zip(
        archive_path,
        iteration_metrics_jsonl=json.dumps({"iteration": 2765.0, "learning_rate": 1.0e-6}) + "\n",
    )

    monkeypatch.setattr(module, "_download_checkpoint_zip", lambda download_source, zip_path: False)

    payload = module.prepare_uploaded_checkpoint(
        download_source="new-id",
        zip_path=archive_path,
        extract_dir=extract_dir,
        skip_existing=True,
        fallback_learning_rate=1e-6,
    )

    assert payload["status"] == "ready"
    assert payload["reused_existing"] is False
    assert payload["restart_iteration"] == 2765
    assert payload["download_source"] == "new-id"
