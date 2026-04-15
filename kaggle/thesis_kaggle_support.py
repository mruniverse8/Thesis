from __future__ import annotations

import importlib
import json
import random
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from src.runtime_bootstrap import RuntimeEnvironment, summarize_environment


KAGGLE_INPUT_ROOT = Path("/kaggle/input")
KAGGLE_WORKING_ROOT = Path("/kaggle/working")
ARTIFACT_EXPORT_ROOT = KAGGLE_WORKING_ROOT / "thesis_artifacts"
REPO_VOCAB_RELATIVE_PATH = Path("molecules") / "dict" / "selfies_dict.txt"
KAGGLE_REPO_DIR = KAGGLE_WORKING_ROOT / "Thesis"


def _get_ipython():
    from IPython import get_ipython

    shell = get_ipython()
    if shell is None:
        raise RuntimeError("These helpers are intended to run inside a notebook kernel.")
    return shell


def json_dumps(payload: object) -> str:
    return json.dumps(payload, indent=2, sort_keys=True)


def import_or_none(module_name: str):
    try:
        return importlib.import_module(module_name)
    except Exception:
        return None


def parse_requirements(path: str | Path) -> list[str]:
    packages: list[str] = []
    for raw_line in Path(path).read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("torch"):
            continue
        packages.append(line)
    return packages


def pip_install(*packages: str) -> None:
    if not packages:
        return
    _get_ipython().run_line_magic("pip", "install --quiet " + " ".join(packages))


def ensure_runtime_dependencies(repo_dir: str | Path) -> None:
    pip_install("--upgrade", "pip", "setuptools", "wheel")
    requirements = parse_requirements(Path(repo_dir) / "requirements.txt")
    pip_install(*requirements)
    if import_or_none("rdkit") is None:
        pip_install("rdkit-pypi")


def report_runtime(*, require_gpu: bool = False) -> dict[str, object]:
    torch = import_or_none("torch")
    rdkit = import_or_none("rdkit")
    transformers = import_or_none("transformers")
    datasets = import_or_none("datasets")
    selfies = import_or_none("selfies")
    yaml = import_or_none("yaml")

    disk_usage = shutil.disk_usage(KAGGLE_WORKING_ROOT)
    report: dict[str, object] = {
        "python": str(sys.version.split()[0]),
        "working_dir": str(KAGGLE_WORKING_ROOT),
        "working_disk_free_gb": round(disk_usage.free / (1024**3), 2),
    }
    if torch is not None:
        report["torch"] = torch.__version__
        report["cuda_available"] = bool(torch.cuda.is_available())
        report["gpu_count"] = int(torch.cuda.device_count())
        if torch.cuda.is_available():
            report["gpu_name"] = torch.cuda.get_device_name(0)
    if rdkit is not None:
        report["rdkit"] = getattr(rdkit, "__version__", "unknown")
    if transformers is not None:
        report["transformers"] = getattr(transformers, "__version__", "unknown")
    if datasets is not None:
        report["datasets"] = getattr(datasets, "__version__", "unknown")
    if selfies is not None:
        report["selfies"] = getattr(selfies, "__version__", "unknown")
    if yaml is not None:
        report["pyyaml"] = getattr(yaml, "__version__", "unknown")

    if require_gpu and not report.get("cuda_available", False):
        raise RuntimeError("GPU not available. Enable a Kaggle GPU accelerator before running.")
    return report


def ensure_repo_selfies_vocab(repo_dir: str | Path) -> Path:
    vocab_path = Path(repo_dir) / REPO_VOCAB_RELATIVE_PATH
    if not vocab_path.exists():
        raise FileNotFoundError(
            "Expected the committed SELFIES vocabulary at "
            f"{vocab_path}. Make sure molecules/dict/selfies_dict.txt is present."
        )
    return vocab_path


def load_yaml(path: str | Path) -> dict[str, Any]:
    import yaml

    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def dump_yaml(payload: object, path: str | Path) -> Path:
    import yaml

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False)
    return destination


def read_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def write_jsonl(path: str | Path, records: list[dict[str, Any]]) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")
    return destination


def ensure_paths_exist(paths_by_name: Mapping[str, str | Path]) -> dict[str, str]:
    resolved: dict[str, str] = {}
    missing: dict[str, str] = {}
    for name, path_value in paths_by_name.items():
        path = Path(path_value)
        resolved[name] = str(path)
        if not path.exists():
            missing[name] = str(path)
    if missing:
        raise FileNotFoundError(json_dumps({"missing_paths": missing}))
    return resolved


def find_stage_artifact_root(
    stage_name: str,
    *,
    preferred_root: str | Path | None = None,
) -> Path | None:
    candidates: list[Path] = []
    if preferred_root is not None:
        preferred = Path(preferred_root)
        candidates.extend([preferred, preferred / stage_name, preferred / "thesis_artifacts"])
    if KAGGLE_INPUT_ROOT.exists():
        for dataset_root in sorted(KAGGLE_INPUT_ROOT.iterdir()):
            if dataset_root.is_dir():
                candidates.extend(
                    [
                        dataset_root,
                        dataset_root / stage_name,
                        dataset_root / "thesis_artifacts",
                    ]
                )

    seen: set[Path] = set()
    for candidate in candidates:
        candidate = candidate.resolve()
        if candidate in seen:
            continue
        seen.add(candidate)
        if candidate.name == stage_name and candidate.exists():
            return candidate
        nested = candidate / stage_name
        if nested.exists():
            return nested
    return None


def copy_path(source: str | Path, destination: str | Path) -> Path:
    source_path = Path(source)
    destination_path = Path(destination)
    if not source_path.exists():
        raise FileNotFoundError(f"Cannot copy missing path: {source_path}")
    if destination_path.exists():
        if destination_path.is_dir() and not destination_path.is_symlink():
            shutil.rmtree(destination_path)
        else:
            destination_path.unlink()
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    if source_path.is_dir():
        shutil.copytree(source_path, destination_path)
    else:
        shutil.copy2(source_path, destination_path)
    return destination_path


def copy_stage_artifact_to_local(
    *,
    stage_name: str,
    artifact_relpath: str | Path,
    local_path: str | Path,
    preferred_root: str | Path | None = None,
) -> Path | None:
    stage_root = find_stage_artifact_root(stage_name, preferred_root=preferred_root)
    if stage_root is None:
        return None
    source_path = stage_root / artifact_relpath
    if not source_path.exists():
        return None
    return copy_path(source_path, local_path)


def ensure_grouped_split_files(
    source_path: str | Path,
    output_dir: str | Path,
    *,
    seed: int = 42,
    validation_fraction: float = 0.05,
    test_fraction: float = 0.05,
) -> dict[str, Path]:
    output_root = Path(output_dir)
    paths = {
        "train": output_root / "train_multimol.jsonl",
        "validation": output_root / "validation_multimol.jsonl",
        "test": output_root / "test_multimol.jsonl",
    }
    if all(path.exists() for path in paths.values()):
        return paths

    records = read_jsonl(source_path)
    if len(records) < 3:
        raise RuntimeError("Need at least 3 grouped examples to derive train/validation/test splits.")

    shuffled = list(records)
    random.Random(seed).shuffle(shuffled)
    test_count = max(1, int(len(shuffled) * test_fraction))
    validation_count = max(1, int(len(shuffled) * validation_fraction))
    if test_count + validation_count >= len(shuffled):
        test_count = 1
        validation_count = 1

    train_end = len(shuffled) - validation_count - test_count
    validation_end = len(shuffled) - test_count

    write_jsonl(paths["train"], shuffled[:train_end])
    write_jsonl(paths["validation"], shuffled[train_end:validation_end])
    write_jsonl(paths["test"], shuffled[validation_end:])
    return paths


def create_zip_archive(source_path: str | Path, destination_path: str | Path) -> Path:
    source_root = Path(source_path)
    if not source_root.exists():
        raise FileNotFoundError(f"Cannot archive missing path: {source_root}")

    destination = Path(destination_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.suffix != ".zip":
        raise ValueError(f"Destination must end with .zip: {destination}")
    if destination.exists():
        destination.unlink()

    archive_base = destination.with_suffix("")
    archive_path = Path(
        shutil.make_archive(
            str(archive_base),
            "zip",
            root_dir=str(source_root.parent),
            base_dir=source_root.name,
        )
    )
    return archive_path


def export_stage_artifacts(
    *,
    stage_name: str,
    artifact_map: Mapping[str, str | Path],
    metadata: Mapping[str, object] | None = None,
) -> tuple[Path, dict[str, Any]]:
    stage_root = ARTIFACT_EXPORT_ROOT / stage_name
    if stage_root.exists():
        shutil.rmtree(stage_root)
    stage_root.mkdir(parents=True, exist_ok=True)

    exported: dict[str, str] = {}
    for relative_destination, source in artifact_map.items():
        destination_path = stage_root / relative_destination
        exported[str(relative_destination)] = str(copy_path(source, destination_path))

    manifest = {
        "stage_name": stage_name,
        "export_root": str(stage_root),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "artifacts": exported,
        "metadata": dict(metadata or {}),
    }
    (stage_root / "manifest.json").write_text(json_dumps(manifest), encoding="utf-8")
    return stage_root, manifest


def get_bootstrap_environment(repo_dir: str | Path | None = None) -> RuntimeEnvironment:
    resolved_repo_dir = Path(repo_dir).expanduser().resolve() if repo_dir is not None else KAGGLE_REPO_DIR
    return RuntimeEnvironment(
        name="kaggle",
        workspace_root=KAGGLE_WORKING_ROOT,
        default_repo_dir=resolved_repo_dir,
    )


def report_bootstrap_runtime(repo_dir: str | Path | None = None) -> dict[str, object]:
    environment = get_bootstrap_environment(repo_dir)
    report = summarize_environment(environment)
    report["kaggle_input_root"] = str(KAGGLE_INPUT_ROOT)
    report["kaggle_working_root"] = str(KAGGLE_WORKING_ROOT)
    return report
