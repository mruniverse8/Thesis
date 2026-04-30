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
    root_prefix: str = "best",
) -> None:
    """Build a synthetic checkpoint zip with the layout `extract_checkpoint_archive` expects."""
    with ZipFile(archive_path, "w") as archive:
        archive.writestr(f"{root_prefix}/config.json", "{}\n")
        archive.writestr(f"{root_prefix}/tokenizer_config.json", "{}\n")
        archive.writestr(f"{root_prefix}/pytorch_model.bin", "weights")
        archive.writestr(f"{root_prefix}/spiece.model", "tokenizer")
        archive.writestr(f"{root_prefix}/iteration_metrics.jsonl", iteration_metrics_jsonl)
        for relative_path, payload in (extra_json_files or {}).items():
            archive.writestr(f"{root_prefix}/{relative_path}", json.dumps(payload))


# -----------------------------------------------------------------------------
# Existing positive tests (kept as-is for back-compat).
# -----------------------------------------------------------------------------

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


# -----------------------------------------------------------------------------
# NEW: focused validation test that explicitly verifies all 5 plan items in one
# place, including `checkpoint_artifact_is_ready(...)` accepting the bundle.
# -----------------------------------------------------------------------------

def test_prepare_gflownet_restart_checkpoint_validates_bundle_and_metadata(
    tmp_path: Path,
) -> None:
    """Verify, in one place, the five things the notebook depends on:

    1. nested archive extraction works (zip -> extract_dir),
    2. checkpoint_artifact_is_ready(extract_dir) returns True after extraction,
    3. the last valid `iteration` from iteration_metrics.jsonl is read correctly,
    4. the restart `learning_rate` is recovered correctly,
    5. next_iteration == restart_iteration + 1.
    """
    from src.checkpoint_bootstrap import checkpoint_artifact_is_ready

    module = _load_restart_module()
    archive_path = tmp_path / "best.zip"
    extract_dir = tmp_path / "extracted" / "best"
    _build_checkpoint_zip(
        archive_path,
        iteration_metrics_jsonl="\n".join(
            [
                json.dumps({"iteration": 999, "learning_rate": 5.0e-6}),
                json.dumps({"iteration": 1500, "learning_rate": 1.0e-6}),
            ]
        ),
    )

    payload = module.prepare_restart_checkpoint(
        download_source=None,
        zip_path=archive_path,
        extract_dir=extract_dir,
        skip_existing=False,
        fallback_learning_rate=1e-6,  # exercise same flag the notebook passes
    )

    # 1. Nested extraction landed at the requested extract_dir.
    assert extract_dir.exists() and extract_dir.is_dir()
    assert (extract_dir / "config.json").exists()
    assert (extract_dir / "tokenizer_config.json").exists()

    # 2. The same predicate the helper uses internally also accepts the bundle.
    assert checkpoint_artifact_is_ready(extract_dir)

    # 3. Last valid iteration wins (1500, not 999).
    assert payload["restart_iteration"] == 1500

    # 4. Learning rate recovered from the same jsonl line as the iteration.
    assert payload["learning_rate"] == pytest.approx(1.0e-6)
    assert payload["learning_rate_source_path"].endswith("iteration_metrics.jsonl")

    # 5. next_iteration is exactly restart_iteration + 1.
    assert payload["next_iteration"] == payload["restart_iteration"] + 1
    assert payload["status"] == "ready"


# -----------------------------------------------------------------------------
# NEW: malformed-archive negative test. This is the exact failure mode the
# notebook hits in production when the Drive bundle does not contain a valid
# checkpoint layout. The helper must fail early with a clear error rather than
# leaving a half-extracted directory the trainer might silently use.
# -----------------------------------------------------------------------------

def test_prepare_gflownet_restart_checkpoint_rejects_malformed_archive(
    tmp_path: Path,
) -> None:
    from src.checkpoint_bootstrap import checkpoint_artifact_is_ready

    module = _load_restart_module()
    archive_path = tmp_path / "best.zip"
    extract_dir = tmp_path / "extracted" / "best"

    # A real zip, but the contents do not match the bundle predicate at any
    # depth that `_find_checkpoint_root` searches: no config.json, no weights,
    # no tokenizer files. This is what happens when the upload was zipped from
    # the wrong working directory (e.g. files end up too deeply nested or
    # under the wrong parent).
    with ZipFile(archive_path, "w") as archive:
        archive.writestr("best/README.md", "not a checkpoint")
        archive.writestr(
            "best/iteration_metrics.jsonl",
            json.dumps({"iteration": 100, "learning_rate": 1.0e-6}) + "\n",
        )
        archive.writestr("best/some_other_file.txt", "noise")

    with pytest.raises(RuntimeError, match="does not contain a valid checkpoint bundle"):
        module.prepare_restart_checkpoint(
            download_source=None,
            zip_path=archive_path,
            extract_dir=extract_dir,
            skip_existing=False,
            fallback_learning_rate=1e-6,
        )

    # Critical: nothing was installed at the destination, so a follow-up
    # training run cannot accidentally consume a half-baked checkpoint.
    assert not checkpoint_artifact_is_ready(extract_dir)
