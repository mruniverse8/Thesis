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

    title_cell = _joined_source(notebook["cells"][0])
    all_source = "\n".join(_joined_source(cell) for cell in notebook["cells"])
    setup_cell = _joined_source(notebook["cells"][1])
    parameter_cell = _joined_source(notebook["cells"][2])
    wandb_cell = _joined_source(notebook["cells"][3])
    bootstrap_cell = _joined_source(notebook["cells"][4])
    train_cell = _joined_source(notebook["cells"][5])
    eval_cell = _joined_source(notebook["cells"][6])
    export_cell = _joined_source(notebook["cells"][7])

    assert "GFlowNet v2.4.1" in title_cell
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

    assert 'GFLOWNET_VERSION = "gflownet_v2.4.1"' in parameter_cell
    assert 'STAGE_NAME = "train_gflownet_lpm24_v241_ablation"' in parameter_cell
    assert 'ABLATION_NAME = "replay_tb_mixture_db_sample"' in parameter_cell
    assert 'CONFIG_OVERRIDE = Path("configs/multi_molecule_gflownet_lpm24.yaml")' in parameter_cell
    assert 'GFLOWNET_OBJECTIVE = "db"' in parameter_cell
    assert "TARGET_GUIDANCE_ENABLED = False" in parameter_cell
    assert "REPLAY_ENABLED = True" in parameter_cell
    assert 'REPLAY_BUFFER_TYPE = "experimental_tb_mixture"' in parameter_cell
    assert "REPLAY_RECENT_FRACTION = 0.30" in parameter_cell
    assert "REPLAY_REWARD_FRACTION = 0.30" in parameter_cell
    assert "REPLAY_UNIFORM_FRACTION = 0.20" in parameter_cell
    assert "REPLAY_TB_RESIDUAL_FRACTION = 0.20" in parameter_cell
    assert 'ROLLOUT_DECODING_STRATEGY = "sample"' in parameter_cell
    assert "ROLLOUT_NUM_BEAMS = 2" in parameter_cell
    assert 'REWARD_VARIANT = "reward_var2"' in parameter_cell
    assert "ENABLE_INVALID_SIMILARITY_NGRAM_FALLBACK = False" in parameter_cell
    assert "PARALLEL_TRAINING_ENABLED = True" in parameter_cell
    assert 'PARALLEL_TRAINING_DEVICES = ["cuda:0", "cuda:1"]' in parameter_cell
    assert 'GFLOWNET_CHECKPOINT_DOWNLOAD_SOURCE = ""' in parameter_cell
    assert 'GFLOWNET_RESTART_CHECKPOINT_SOURCE = "1eXqmu4HfPS1Gc2I3AjZR2w0pC2dfnO4E"' in parameter_cell
    assert 'RESTART_CHECKPOINT_ROOT = REPO_DIR / "outputs" / "kaggle" / "restart_checkpoints"' in parameter_cell
    assert 'RESTART_CHECKPOINT_ZIP = RESTART_CHECKPOINT_ROOT / "best.zip"' in parameter_cell
    assert 'RESTART_CHECKPOINT_DIR = RESTART_CHECKPOINT_ROOT / "best"' in parameter_cell

    assert "scripts/download_lpm24.py" in bootstrap_cell
    assert "scripts/prepare_lpm24_training_splits.py" in bootstrap_cell
    assert '"--max-target-symbols"' in bootstrap_cell
    assert '"--max-stage-symbols"' in bootstrap_cell

    assert 'tracking_payload["tags"]' in train_cell
    assert "GFLOWNET_VERSION" in train_cell
    assert '"parallel_training"' in train_cell
    assert "scripts/prepare_gflownet_restart_checkpoint.py" in train_cell
    assert "GFLOWNET_RESTART_CHECKPOINT_SOURCE" in train_cell
    assert 'runtime_config.setdefault("model", {})["checkpoint"] = restart_metadata["checkpoint_dir"]' in train_cell
    assert 'gflownet_payload["learning_rate"] = float(restart_metadata["learning_rate"])' in train_cell
    assert 'gflownet_payload["start_iteration"] = int(restart_metadata["restart_iteration"])' in train_cell
    assert '"gflownet_start_iteration": runtime_config["gflownet"].get("start_iteration")' in train_cell
    assert '"restart_checkpoint_dir": restart_checkpoint_metadata.get("checkpoint_dir")' in train_cell
    assert '"restart_learning_rate_source": restart_checkpoint_metadata.get("learning_rate_source_path")' in train_cell
    assert '"restart_iteration": restart_checkpoint_metadata.get("restart_iteration")' in train_cell
    assert '"restart_next_iteration": restart_checkpoint_metadata.get("next_iteration")' in train_cell
    assert 'rollout_payload["decoding_strategy"] = ROLLOUT_DECODING_STRATEGY' in train_cell
    assert 'rollout_payload["num_beams"] = int(ROLLOUT_NUM_BEAMS)' in train_cell
    assert 'target_guidance_payload["enabled"] = bool(TARGET_GUIDANCE_ENABLED)' in train_cell
    assert 'replay_payload["enabled"] = bool(REPLAY_ENABLED)' in train_cell
    assert 'replay_payload["buffer_type"] = REPLAY_BUFFER_TYPE' in train_cell
    assert 'replay_payload["recent_fraction"] = float(REPLAY_RECENT_FRACTION)' in train_cell
    assert 'replay_payload["reward_fraction"] = float(REPLAY_REWARD_FRACTION)' in train_cell
    assert 'replay_payload["uniform_fraction"] = float(REPLAY_UNIFORM_FRACTION)' in train_cell
    assert 'replay_payload["tb_residual_fraction"] = float(REPLAY_TB_RESIDUAL_FRACTION)' in train_cell
    assert '"replay_tb_residual_fraction": runtime_config["gflownet"]["replay"].get("tb_residual_fraction")' in train_cell
    assert 'reward_payload["reward_variant"] = REWARD_VARIANT' in train_cell
    assert 'gflownet_payload["parallel_training"]' in train_cell
    assert '"devices": list(PARALLEL_TRAINING_DEVICES)' in train_cell
    assert '"parallel_training_mode": runtime_config["gflownet"]["parallel_training"].get("mode")' in train_cell
    assert '"parallel_training_strict": runtime_config["gflownet"]["parallel_training"].get("strict")' in train_cell
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
    assert '"gflownet_version": GFLOWNET_VERSION' in export_cell
    assert '"repo_branch": REPO_BRANCH' in export_cell
    assert '"parallel_training": {' in export_cell
    assert '"restart_checkpoint": {' in export_cell
    assert '"learning_rate_source_path": restart_checkpoint_metadata.get("learning_rate_source_path")' in export_cell
    assert '"restart_iteration": restart_checkpoint_metadata.get("restart_iteration")' in export_cell
    assert '"next_iteration": restart_checkpoint_metadata.get("next_iteration")' in export_cell
    assert '"output": OUTPUT_DIR' in export_cell
    assert '"run_summary.json": RUN_SUMMARY_PATH' in export_cell
    assert '"validation_evaluation_metrics.json": EVAL_OUTPUT_PATH' in export_cell
