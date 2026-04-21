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


def test_colab_sft_notebook_bootstraps_then_runs_train_script() -> None:
    notebook = _load_notebook("colab/01_train_sft.ipynb")

    assert notebook["nbformat"] == 4
    assert len(notebook["cells"]) == 6

    first_cell = _joined_source(notebook["cells"][0])
    wandb_cell = _joined_source(notebook["cells"][3])
    run_cell = _joined_source(notebook["cells"][4])
    post_cell = _joined_source(notebook["cells"][5])

    assert "On Colab" in first_cell
    assert "scripts/train_sft.py" in first_cell
    assert "WANDB_API_KEY" in wandb_cell
    assert "\"--stage\"" in run_cell and "\"sft\"" in run_cell
    assert "scripts/init_colab.py" in run_cell
    assert "scripts/train_sft.py" in run_cell
    assert "EXTRA_SCRIPT_ARGS" not in run_cell
    assert "BEST_CHECKPOINT" in post_cell
    assert "RUN_SUMMARY_PATH" in post_cell


def test_colab_multi_sft_notebook_uses_mini_dataset_flow() -> None:
    notebook = _load_notebook("colab/02_train_multi_molecule_sft.ipynb")

    assert notebook["nbformat"] == 4
    assert len(notebook["cells"]) == 6

    first_cell = _joined_source(notebook["cells"][0])
    parameter_cell = _joined_source(notebook["cells"][2])
    wandb_cell = _joined_source(notebook["cells"][3])
    run_cell = _joined_source(notebook["cells"][4])
    post_cell = _joined_source(notebook["cells"][5])

    assert "On Colab" in first_cell
    assert "scripts/train_multi_molecule_sft.py" in first_cell
    assert "WANDB_API_KEY" in wandb_cell
    assert "TRAIN_DATASET_FILE_ID" in parameter_cell
    assert "\"data\" / \"mini_post_training\"" in parameter_cell
    assert "TRAIN_CONFIG" in parameter_cell
    assert "\"multi_molecule_sft_mini\"" in parameter_cell
    assert "\"--stage\"" in run_cell and "\"multi_sft\"" in run_cell
    assert "\"--train-dataset-file-id\"" in run_cell
    assert "scripts/train_multi_molecule_sft.py" in run_cell
    assert "EXTRA_SCRIPT_ARGS" not in run_cell
    assert "mini_dataset_checks" in post_cell


def test_colab_lpm24_multi_sft_v2_notebook_prepares_manual_dataset_and_zips_artifacts() -> None:
    notebook = _load_notebook("colab/03_v2_train_multi_molecule_sft_lpm24.ipynb")

    assert notebook["nbformat"] == 4
    assert len(notebook["cells"]) == 7

    first_cell = _joined_source(notebook["cells"][0])
    setup_cell = _joined_source(notebook["cells"][1])
    parameter_cell = _joined_source(notebook["cells"][2])
    drive_cell = _joined_source(notebook["cells"][3])
    wandb_cell = _joined_source(notebook["cells"][4])
    run_cell = _joined_source(notebook["cells"][5])
    post_cell = _joined_source(notebook["cells"][6])

    assert "On Colab" in first_cell
    assert "scripts/train_multi_molecule_sft.py" in first_cell
    assert "scripts/download_lpm24.py" in first_cell
    assert "scripts/prepare_lpm24_training_splits.py" in first_cell
    assert "exact Drive output directory" in first_cell
    assert "REPO_BRANCH = \"gflownet_v2\"" in setup_cell
    assert "\"data\" / \"lpm24\"" in parameter_cell
    assert "\"multi_molecule_sft_lpm24\"" in parameter_cell
    assert "from datetime import datetime" in parameter_cell
    assert "COLAB_OUTPUT_ROOT = Path(" in parameter_cell
    assert "RUN_STAMP = datetime.now().strftime(\"%y%m%d_%H%M%S\")" in parameter_cell
    assert "RUN_NAME = f\"{DEFAULT_CONFIG_STEM}_{RUN_STAMP}\"" in parameter_cell
    assert "COLAB_OUTPUT_DIR = COLAB_OUTPUT_ROOT / RUN_NAME" in parameter_cell
    assert "\"/content/drive/MyDrive/" in parameter_cell
    assert "DATASET_MODE = \"never\"" in parameter_cell
    assert "MAX_TARGET_SYMBOLS = 1024" in parameter_cell
    assert "MAX_STAGE_SYMBOLS = 192" in parameter_cell
    assert "OUTPUT_DIR_ZIP" in parameter_cell
    assert "BEST_CHECKPOINT_ZIP" in parameter_cell
    assert "OUTPUT_DIR = COLAB_OUTPUT_DIR.expanduser()" in parameter_cell
    assert "OUTPUT_DIR / f\"{OUTPUT_DIR.name}.zip\"" in parameter_cell
    assert "from google.colab import drive" in drive_cell
    assert "drive.mount('/content/drive')" in drive_cell
    assert "os.makedirs(OUTPUT_DIR, exist_ok=True)" in drive_cell
    assert "\"run_name\": RUN_NAME" in drive_cell
    assert "WANDB_API_KEY =" in wandb_cell
    assert "wandb.login(key=WANDB_API_KEY, relogin=True)" in wandb_cell
    assert "\"--stage\"" in run_cell and "\"multi_sft\"" in run_cell
    assert "\"--dataset-mode\"" in run_cell and "DATASET_MODE" in run_cell
    assert "scripts/init_colab.py" in run_cell
    assert "scripts/download_lpm24.py" in run_cell
    assert "scripts/prepare_lpm24_training_splits.py" in run_cell
    assert "scripts/train_multi_molecule_sft.py" in run_cell
    assert "\"--output-dir\"" in run_cell and "str(OUTPUT_DIR)" in run_cell
    assert "processed_dataset_ready" in run_cell
    assert "output_dir_zip" in post_cell
    assert "best_checkpoint_zip" in post_cell
    assert "archive_directory_to_zip" in post_cell
    assert "processed_dataset_checks" in post_cell
    assert "grouped_split_checks" in post_cell
    assert "best_validation_loss" in post_cell


def test_colab_lpm24_gflownet_notebook_prepares_manual_dataset_and_runs_train_script() -> None:
    notebook = _load_notebook("colab/06_train_multi_molecule_gflownet_lpm24.ipynb")

    assert notebook["nbformat"] == 4
    assert len(notebook["cells"]) == 7

    first_cell = _joined_source(notebook["cells"][0])
    setup_cell = _joined_source(notebook["cells"][1])
    parameter_cell = _joined_source(notebook["cells"][2])
    drive_cell = _joined_source(notebook["cells"][3])
    wandb_cell = _joined_source(notebook["cells"][4])
    run_cell = _joined_source(notebook["cells"][5])
    post_cell = _joined_source(notebook["cells"][6])

    assert "On Colab" in first_cell
    assert "scripts/train_multi_molecule_gflownet.py" in first_cell
    assert "scripts/download_lpm24.py" in first_cell
    assert "scripts/prepare_lpm24_training_splits.py" in first_cell
    assert "REPO_BRANCH = \"gflownet_v2\"" in setup_cell
    assert "\"data\" / \"lpm24\"" in parameter_cell
    assert "\"multi_molecule_gflownet_lpm24\"" in parameter_cell
    assert "from datetime import datetime" in parameter_cell
    assert "COLAB_OUTPUT_ROOT = Path(" in parameter_cell
    assert "RUN_STAMP = datetime.now().strftime(\"%y%m%d_%H%M%S\")" in parameter_cell
    assert "RUN_NAME = f\"{DEFAULT_CONFIG_STEM}_{GFLOWNET_OBJECTIVE}_{RUN_STAMP}\"" in parameter_cell
    assert "COLAB_OUTPUT_DIR = COLAB_OUTPUT_ROOT / RUN_NAME" in parameter_cell
    assert "\"/content/drive/MyDrive/" in parameter_cell
    assert "DATASET_MODE = \"never\"" in parameter_cell
    assert "GFLOWNET_CHECKPOINT_DOWNLOAD_SOURCE = \"\"" in parameter_cell
    assert "MAX_TARGET_SYMBOLS = 1024" in parameter_cell
    assert "MAX_STAGE_SYMBOLS = 192" in parameter_cell
    assert "OUTPUT_DIR = COLAB_OUTPUT_DIR.expanduser()" in parameter_cell
    assert "OUTPUT_DIR / f\"{OUTPUT_DIR.name}.zip\"" in parameter_cell
    assert "from google.colab import drive" in drive_cell
    assert "drive.mount('/content/drive')" in drive_cell
    assert "os.makedirs(OUTPUT_DIR, exist_ok=True)" in drive_cell
    assert "\"run_name\": RUN_NAME" in drive_cell
    assert "os.environ.get(\"WANDB_API_KEY\"" in wandb_cell
    assert "wandb_v1_" not in wandb_cell
    assert "\"--stage\"" in run_cell and "\"gflownet\"" in run_cell
    assert "\"--dataset-mode\"" in run_cell and "DATASET_MODE" in run_cell
    assert "\"--gflownet-checkpoint-download-source\"" in run_cell
    assert "GFLOWNET_CHECKPOINT_DOWNLOAD_SOURCE.strip()" in run_cell
    assert "scripts/init_colab.py" in run_cell
    assert "scripts/download_lpm24.py" in run_cell
    assert "scripts/prepare_lpm24_training_splits.py" in run_cell
    assert "scripts/train_multi_molecule_gflownet.py" in run_cell
    assert "yaml.safe_dump" in run_cell
    assert "constrained_decoding" in run_cell
    assert "stage_separator" in run_cell
    assert "\"--output-dir\"" in run_cell and "str(OUTPUT_DIR)" in run_cell
    assert "runtime_config.setdefault(\"training\", {})[\"output_dir\"]" not in run_cell
    assert "archive_directory_to_zip" in post_cell
    assert "processed_dataset_checks" in post_cell
    assert "grouped_split_checks" in post_cell
    assert "resolved_checkpoint_source" in post_cell
