from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from zipfile import ZipFile


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = PROJECT_ROOT / "scripts" / "inspect_gflownet_checkpoint_bundle.py"


def _run_script(path: Path) -> dict:
    completed = subprocess.run(
        [sys.executable, str(SCRIPT_PATH), str(path)],
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(completed.stdout)


def test_inspect_gflownet_checkpoint_bundle_classifies_adapter_zip(tmp_path: Path) -> None:
    bundle_root = tmp_path / "best"
    bundle_root.mkdir()
    (bundle_root / "adapter_config.json").write_text(
        json.dumps({"base_model_name_or_path": "/content/Thesis/outputs/gflownet_restart_checkpoint/best"}),
        encoding="utf-8",
    )
    (bundle_root / "adapter_model.safetensors").write_text("adapter", encoding="utf-8")
    (bundle_root / "tokenizer_config.json").write_text("{}", encoding="utf-8")
    (bundle_root / "tokenizer.json").write_text("{}", encoding="utf-8")
    (bundle_root / "iteration_metrics.json").write_text(
        json.dumps({"iteration": 2765.0, "learning_rate": 1e-6}),
        encoding="utf-8",
    )
    zip_path = tmp_path / "best.zip"
    with ZipFile(zip_path, "w") as archive:
        for file_path in sorted(bundle_root.iterdir()):
            archive.write(file_path, arcname=str(Path("best") / file_path.name))

    payload = _run_script(zip_path)
    inspection = payload["selected_inspection"]

    assert payload["status"] == "ok"
    assert inspection["classification"] == "adapter_bundle"
    assert inspection["is_full_checkpoint"] is False
    assert inspection["restart_iteration"] == 2765
    assert inspection["learning_rate"] == 1e-6
    assert "config.json" in inspection["missing_for_full_checkpoint"]


def test_inspect_gflownet_checkpoint_bundle_classifies_full_checkpoint_dir(tmp_path: Path) -> None:
    checkpoint_dir = tmp_path / "best"
    checkpoint_dir.mkdir()
    (checkpoint_dir / "config.json").write_text("{}", encoding="utf-8")
    (checkpoint_dir / "tokenizer_config.json").write_text("{}", encoding="utf-8")
    (checkpoint_dir / "tokenizer.json").write_text("{}", encoding="utf-8")
    (checkpoint_dir / "model.safetensors").write_text("weights", encoding="utf-8")
    (checkpoint_dir / "iteration_metrics.jsonl").write_text(
        json.dumps({"iteration": 1536.0, "learning_rate": 1e-6}) + "\n",
        encoding="utf-8",
    )

    payload = _run_script(checkpoint_dir)
    inspection = payload["inspection"]

    assert payload["status"] == "ok"
    assert inspection["classification"] == "full_checkpoint"
    assert inspection["is_full_checkpoint"] is True
    assert inspection["restart_iteration"] == 1536
    assert inspection["learning_rate"] == 1e-6
