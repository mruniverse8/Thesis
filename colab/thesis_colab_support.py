from __future__ import annotations

import sys
from pathlib import Path

from src.runtime_bootstrap import RuntimeEnvironment, summarize_environment

COLAB_ROOT = Path("/content")
DEFAULT_REPO_DIR = COLAB_ROOT / "Thesis"


def get_bootstrap_environment(repo_dir: str | Path | None = None) -> RuntimeEnvironment:
    resolved_repo_dir = Path(repo_dir).expanduser().resolve() if repo_dir is not None else DEFAULT_REPO_DIR
    return RuntimeEnvironment(
        name="colab",
        workspace_root=COLAB_ROOT,
        default_repo_dir=resolved_repo_dir,
    )


def report_bootstrap_runtime(repo_dir: str | Path | None = None) -> dict[str, object]:
    environment = get_bootstrap_environment(repo_dir)
    report = summarize_environment(environment)
    report["sys_path_entry_count"] = len(sys.path)
    return report
