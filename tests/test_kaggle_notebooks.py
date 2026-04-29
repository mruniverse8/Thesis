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


def test_kaggle_lpm24_gflownet_ablation_notebook_uses_kaggle_bootstrap_and_artifacts() -> None:
    notebook = _load_notebook("kaggle/07_v2_train_multi_molecule_gflownet_lpm24_ablation.ipynb")

    assert notebook["nbformat"] == 4
    assert len(notebook["cells"]) == 8

    all_source = "\n".join(_joined_source(cell) for cell in notebook["cells"])
    setup_cell = _joined_source(notebook["cells"][1])
    parameter_cell = _joined_source(notebook["cells"][2])
    wandb_cell = _joined_source(notebook["cells"][3])
    bootstrap_cell = _joined_source(notebook["cells"][4])
    train_cell = _joined_source(notebook["cells"][5])
    eval_cell = _joined_source(notebook["cells"][6])
    export_cell = _joined_source(notebook["cells"][7])

    assert 'REPO_DIR = Path("/kaggle/working/Thesis")' in setup_cell
    assert 'REPO_BRANCH = "gflownet_v2.4.1"' in setup_cell
    assert "/content/Thesis" not in all_source
    assert "google.colab" not in all_source

    assert "scripts/init_kaggle.py" in bootstrap_cell
    assert "scripts/init_colab.py" not in all_source
    assert '"--stage"' in bootstrap_cell and '"gflownet"' in bootstrap_cell
    assert '"--gflownet-checkpoint-download-source"' in bootstrap_cell
    assert "GFLOWNET_CHECKPOINT_DOWNLOAD_SOURCE.strip()" in bootstrap_cell

    assert 'WANDB_API_KEY = os.environ.get("WANDB_API_KEY", "")' in wandb_cell
    assert "wandb.login(key=WANDB_API_KEY, relogin=True)" in wandb_cell
    assert "wandb_v1_" not in all_source

    assert 'STAGE_NAME = "train_gflownet_lpm24_ablation"' in parameter_cell
    assert 'ABLATION_NAME = "baseline_db_beam_target_guidance"' in parameter_cell
    assert 'CONFIG_OVERRIDE = Path("configs/multi_molecule_gflownet_lpm24.yaml")' in parameter_cell
    assert 'GFLOWNET_OBJECTIVE = "db"' in parameter_cell
    assert "TARGET_GUIDANCE_ENABLED = True" in parameter_cell
    assert "REPLAY_ENABLED = False" in parameter_cell
    assert 'ROLLOUT_DECODING_STRATEGY = "beam"' in parameter_cell
    assert "ROLLOUT_NUM_BEAMS = 2" in parameter_cell
    assert 'REWARD_VARIANT = "reward_var2"' in parameter_cell
    assert "ENABLE_INVALID_SIMILARITY_NGRAM_FALLBACK = False" in parameter_cell
    assert "PARALLEL_TRAINING_ENABLED = True" in parameter_cell
    assert 'PARALLEL_TRAINING_DEVICES = ["cuda:0", "cuda:1"]' in parameter_cell
    assert 'GFLOWNET_CHECKPOINT_DOWNLOAD_SOURCE = "1jCIVYbzgTw7xQAWvv6SfwM8Y1vL47PDg"' in parameter_cell

    assert "scripts/download_lpm24.py" in bootstrap_cell
    assert "scripts/prepare_lpm24_training_splits.py" in bootstrap_cell
    assert '"--max-target-symbols"' in bootstrap_cell
    assert '"--max-stage-symbols"' in bootstrap_cell

    assert 'tracking_payload["tags"]' in train_cell
    assert 'rollout_payload["decoding_strategy"] = ROLLOUT_DECODING_STRATEGY' in train_cell
    assert 'rollout_payload["num_beams"] = int(ROLLOUT_NUM_BEAMS)' in train_cell
    assert 'target_guidance_payload["enabled"] = bool(TARGET_GUIDANCE_ENABLED)' in train_cell
    assert 'replay_payload["enabled"] = bool(REPLAY_ENABLED)' in train_cell
    assert 'reward_payload["reward_variant"] = REWARD_VARIANT' in train_cell
    assert 'gflownet_payload["parallel_training"]' in train_cell
    assert '"devices": list(PARALLEL_TRAINING_DEVICES)' in train_cell
    assert "scripts/train_multi_molecule_gflownet.py" in train_cell
    assert '"--output-dir"' in train_cell and "str(OUTPUT_DIR)" in train_cell

    assert "validation_evaluation_metrics.json" in parameter_cell
    assert "evaluate_generation_groups" in eval_cell
    assert "evaluation_diagnosis" in eval_cell
    assert "eval/novelty_fraction" in eval_cell
    assert "wandb.log(evaluation_diagnosis)" in eval_cell

    assert 'ARTIFACT_OUTPUT_ROOT = Path("/kaggle/working/thesis_artifacts")' in parameter_cell
    assert "export_stage_artifacts" in export_cell
    assert "create_zip_archive(stage_root, STAGE_ARTIFACT_ZIP)" in export_cell
    assert '"output": OUTPUT_DIR' in export_cell
    assert '"run_summary.json": RUN_SUMMARY_PATH' in export_cell
    assert '"validation_evaluation_metrics.json": EVAL_OUTPUT_PATH' in export_cell
