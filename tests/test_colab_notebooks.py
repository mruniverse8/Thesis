from __future__ import annotations

import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _load_notebook(relative_path: str) -> dict:
    notebook_path = PROJECT_ROOT / relative_path
    with notebook_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _joined_source(cell: dict) -> str:
    return "".join(cell.get("source", []))


def test_colab_sft_notebook_is_thin_wrapper_over_init_script() -> None:
    notebook = _load_notebook("colab/01_train_sft.ipynb")

    assert notebook["nbformat"] == 4
    assert len(notebook["cells"]) == 5

    first_cell = _joined_source(notebook["cells"][0])
    run_cell = _joined_source(notebook["cells"][3])
    post_cell = _joined_source(notebook["cells"][4])

    assert "On Colab" in first_cell
    assert "scripts/init_colab.py" in first_cell
    assert "\"--stage\"" in run_cell and "\"sft\"" in run_cell
    assert "scripts/init_colab.py" in run_cell
    assert "BEST_CHECKPOINT" in post_cell
    assert "RUN_SUMMARY_PATH" in post_cell


def test_colab_multi_sft_notebook_uses_mini_dataset_flow() -> None:
    notebook = _load_notebook("colab/02_train_multi_molecule_sft.ipynb")

    assert notebook["nbformat"] == 4
    assert len(notebook["cells"]) == 5

    first_cell = _joined_source(notebook["cells"][0])
    parameter_cell = _joined_source(notebook["cells"][2])
    run_cell = _joined_source(notebook["cells"][3])
    post_cell = _joined_source(notebook["cells"][4])

    assert "On Colab" in first_cell
    assert "mini post-training dataset" in first_cell
    assert "TRAIN_DATASET_FILE_ID" in parameter_cell
    assert "\"data\" / \"mini_post_training\"" in parameter_cell
    assert "\"--stage\"" in run_cell and "\"multi_sft\"" in run_cell
    assert "\"--train-dataset-file-id\"" in run_cell
    assert "mini_dataset_checks" in post_cell
