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


def test_colab_gflownet_notebook_uses_env_wandb_and_optional_checkpoint_source() -> None:
    notebook = _load_notebook("colab/05_train_multi_molecule_gflownet.ipynb")

    assert notebook["nbformat"] == 4
    assert len(notebook["cells"]) == 6

    first_cell = _joined_source(notebook["cells"][0])
    parameter_cell = _joined_source(notebook["cells"][2])
    wandb_cell = _joined_source(notebook["cells"][3])
    run_cell = _joined_source(notebook["cells"][4])
    post_cell = _joined_source(notebook["cells"][5])

    assert "On Colab" in first_cell
    assert "scripts/train_multi_molecule_gflownet.py" in first_cell
    assert "GFLOWNET_CHECKPOINT_DOWNLOAD_SOURCE = \"\"" in parameter_cell
    assert "WANDB_PROJECT_URL" in parameter_cell
    assert "gflownet_checkpoint_download_source_configured" in parameter_cell
    assert "os.environ.get(\"WANDB_API_KEY\"" in wandb_cell
    assert "wandb_v1_" not in wandb_cell
    assert "wandb_project_url" in wandb_cell
    assert "\"--stage\"" in run_cell and "\"gflownet\"" in run_cell
    assert "\"--train-dataset-file-id\"" in run_cell
    assert "\"--gflownet-checkpoint-download-source\"" in run_cell
    assert "GFLOWNET_CHECKPOINT_DOWNLOAD_SOURCE.strip()" in run_cell
    assert "scripts/init_colab.py" in run_cell
    assert "scripts/train_multi_molecule_gflownet.py" in run_cell
    assert "mini_dataset_checks" in post_cell


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
    assert 'GFLOWNET_CHECKPOINT_DOWNLOAD_SOURCE = "1jCIVYbzgTw7xQAWvv6SfwM8Y1vL47PDg"' in parameter_cell
    assert "MAX_TARGET_SYMBOLS = 1024" in parameter_cell
    assert "MAX_STAGE_SYMBOLS = 192" in parameter_cell
    assert "ROLLOUT_MAX_STAGE_NEW_TOKENS = 180" in parameter_cell
    assert "ROLLOUT_MAX_MOLECULES_PER_SEQUENCE = 16" in parameter_cell
    assert "ROLLOUT_MAX_SEQUENCE_LENGTH = 8192" in parameter_cell
    assert "Replay capacity counts stored stage trajectories" in parameter_cell
    assert "ROLLOUT_TEMPERATURE = 0.12" in parameter_cell
    assert "ROLLOUT_TOP_P = 0.40" in parameter_cell
    assert "REWARD_DIVERSITY_BETA = 0.6" in parameter_cell
    assert "REWARD_DIVERSITY_WEIGHT = 0.4" in parameter_cell
    assert "REWARD_MATCH_WEIGHT = 1.0" in parameter_cell
    assert 'REWARD_VARIANT = "reward_var2"' in parameter_cell
    assert '# reward_v1: REWARD_VARIANT = "reward_var1"' in parameter_cell
    assert "REWARD_PLUS_VALID = 0.8" in parameter_cell
    assert "filters samples in `scripts/prepare_lpm24_training_splits.py`" in parameter_cell
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
    assert 'replay_payload["max_total_action_tokens"]' not in run_cell
    assert 'rollout_payload["max_stage_new_tokens"] = int(ROLLOUT_MAX_STAGE_NEW_TOKENS)' in run_cell
    assert 'rollout_payload["max_molecules_per_sequence"] = int(ROLLOUT_MAX_MOLECULES_PER_SEQUENCE)' in run_cell
    assert 'rollout_payload["max_sequence_length"] = int(ROLLOUT_MAX_SEQUENCE_LENGTH)' in run_cell
    assert 'reward_payload["diversity_beta"] = float(REWARD_DIVERSITY_BETA)' in run_cell
    assert 'reward_payload["diversity_weight"] = float(REWARD_DIVERSITY_WEIGHT)' in run_cell
    assert 'reward_payload["match_weight"] = float(REWARD_MATCH_WEIGHT)' in run_cell
    assert 'reward_payload["reward_variant"] = REWARD_VARIANT' in run_cell
    assert 'reward_payload["plus_valid"] = float(REWARD_PLUS_VALID)' in run_cell
    assert "control_summary =" in run_cell
    assert "Tuning guide:" in run_cell
    assert "`termination_fraction_max_stage_new_tokens` is the external/ `max-stage` metric" in run_cell
    assert "mean_terminal_stop_logprob" in run_cell
    assert "termination_fraction_stop_token" in run_cell
    assert "constrained_decoding" in run_cell
    assert "stage_separator" in run_cell
    assert "\"--output-dir\"" in run_cell and "str(OUTPUT_DIR)" in run_cell
    assert "runtime_config.setdefault(\"training\", {})[\"output_dir\"]" not in run_cell
    assert "archive_directory_to_zip" in post_cell
    assert "processed_dataset_checks" in post_cell
    assert "grouped_split_checks" in post_cell
    assert "resolved_checkpoint_source" in post_cell


def test_colab_lpm24_gflownet_06_v2_notebook_uses_beam_rollout_controls() -> None:
    notebook = _load_notebook("colab/06_v2_train_multi_molecule_gflownet_lpm24.ipynb")

    assert notebook["nbformat"] == 4
    assert len(notebook["cells"]) == 7

    first_cell = _joined_source(notebook["cells"][0])
    parameter_cell = _joined_source(notebook["cells"][2])
    run_cell = _joined_source(notebook["cells"][5])

    assert "manual constrained beam-search rollout" in first_cell
    assert 'ROLLOUT_DECODING_STRATEGY = "beam"' in parameter_cell
    assert "ROLLOUT_NUM_BEAMS = 3" in parameter_cell
    assert "ROLLOUT_LENGTH_PENALTY = 1.0" in parameter_cell
    assert "ROLLOUT_EARLY_STOPPING = True" in parameter_cell
    assert 'REPLAY_BUFFER_TYPE = "experimental_mixture"' in parameter_cell
    assert "ENABLE_INVALID_SIMILARITY_NGRAM_FALLBACK = False" in parameter_cell
    assert "beam rollout is controlled by the beam settings above" in parameter_cell
    assert '"rollout_decoding_strategy": ROLLOUT_DECODING_STRATEGY' in parameter_cell
    assert '"rollout_num_beams": ROLLOUT_NUM_BEAMS' in parameter_cell
    assert '"rollout_length_penalty": ROLLOUT_LENGTH_PENALTY' in parameter_cell
    assert '"rollout_early_stopping": ROLLOUT_EARLY_STOPPING' in parameter_cell
    assert (
        '"enable_invalid_similarity_ngram_fallback": '
        "ENABLE_INVALID_SIMILARITY_NGRAM_FALLBACK"
    ) in parameter_cell

    assert 'rollout_payload["decoding_strategy"] = ROLLOUT_DECODING_STRATEGY' in run_cell
    assert 'rollout_payload["num_beams"] = int(ROLLOUT_NUM_BEAMS)' in run_cell
    assert 'rollout_payload["length_penalty"] = float(ROLLOUT_LENGTH_PENALTY)' in run_cell
    assert 'rollout_payload["early_stopping"] = bool(ROLLOUT_EARLY_STOPPING)' in run_cell
    assert (
        'reward_payload["enable_invalid_similarity_ngram_fallback"] = '
        "bool(ENABLE_INVALID_SIMILARITY_NGRAM_FALLBACK)"
    ) in run_cell
    assert '"rollout_decoding_strategy": rollout_payload.get("decoding_strategy")' in run_cell
    assert '"rollout_num_beams": rollout_payload.get("num_beams")' in run_cell
    assert '"rollout_length_penalty": rollout_payload.get("length_penalty")' in run_cell
    assert '"rollout_early_stopping": rollout_payload.get("early_stopping")' in run_cell
    assert (
        '"enable_invalid_similarity_ngram_fallback": '
        'reward_payload.get("enable_invalid_similarity_ngram_fallback")'
    ) in run_cell


def test_colab_lpm24_ppo_v2_notebook_reports_iteration_diagnostics_and_preview_summary() -> None:
    notebook = _load_notebook("colab/03_v2_train_molecule_wise_ppo_lpm24.ipynb")

    assert notebook["nbformat"] == 4
    assert len(notebook["cells"]) == 7

    first_cell = _joined_source(notebook["cells"][0])
    parameter_cell = _joined_source(notebook["cells"][2])
    wandb_cell = _joined_source(notebook["cells"][4])
    run_cell = _joined_source(notebook["cells"][5])
    post_cell = _joined_source(notebook["cells"][6])

    assert "On Colab" in first_cell
    assert "scripts/train_molecule_wise_ppo.py" in first_cell
    assert "OPTIMIZER_STEP_METRICS_PATH" in parameter_cell
    assert "ITERATION_DIAGNOSTICS_PATH" in parameter_cell
    assert "TRAJECTORY_PREVIEWS_PATH" in parameter_cell
    assert 'OPTIMIZER_STEP_DIAGNOSTICS = "every_optimizer_step"' in parameter_cell
    assert "ROLLOUT_TEMPERATURE = 0.12" in parameter_cell
    assert "ROLLOUT_TOP_P = 0.40" in parameter_cell
    assert "WANDB_API_KEY =" in wandb_cell
    assert "\"--stage\"" in run_cell and "\"ppo\"" in run_cell
    assert "scripts/init_colab.py" in run_cell
    assert "scripts/train_molecule_wise_ppo.py" in run_cell
    assert "trajectory_preview_every_iterations" in run_cell
    assert "num_trajectory_samples_to_log" in run_cell
    assert "trajectory_preview_max_chars" in run_cell
    assert '"optimizer_step_diagnostics": OPTIMIZER_STEP_DIAGNOSTICS' in run_cell
    assert '"optimizer_step_tracker_diagnostic_prefixes": ["ppo_optimizer"]' in run_cell
    assert 'rollout_payload["temperature"] = float(ROLLOUT_TEMPERATURE)' in run_cell
    assert 'rollout_payload["top_p"] = float(ROLLOUT_TOP_P)' in run_cell
    assert '"rollout_temperature": rollout_payload.get("temperature")' in run_cell
    assert '"rollout_top_p": rollout_payload.get("top_p")' in run_cell
    assert "archive_directory_to_zip" in post_cell
    assert "iteration_diagnostics_path" in post_cell
    assert '"optimizer_step_diagnostics": OPTIMIZER_STEP_DIAGNOSTICS' in post_cell
    assert '"optimizer_step_tracker_diagnostic_prefixes": ["ppo_optimizer"]' in post_cell
    assert "latest_rollout_diagnostics" in post_cell
    assert "latest_preview_iteration" in post_cell
    assert "trajectory_preview_examples" in post_cell


def test_colab_lpm24_ppo_03_03_notebook_aligns_shared_controls_with_gflownet_v2() -> None:
    notebook = _load_notebook("colab/03_03_train_molecule_wise_ppo_lpm24.ipynb")

    assert notebook["nbformat"] == 4
    assert len(notebook["cells"]) == 7

    first_cell = _joined_source(notebook["cells"][0])
    setup_cell = _joined_source(notebook["cells"][1])
    parameter_cell = _joined_source(notebook["cells"][2])
    run_cell = _joined_source(notebook["cells"][5])

    assert "On Colab" in first_cell
    assert "06_v2_train_multi_molecule_gflownet_lpm24" in first_cell
    assert 'REPO_BRANCH = "gflownet_v2.1"' in setup_cell
    assert 'RUN_NAME = f"{DEFAULT_CONFIG_STEM}_03_03_{RUN_STAMP}"' in parameter_cell
    assert "PPO_ITERATIONS = 64" in parameter_cell
    assert "MAX_TARGET_SYMBOLS = 1024" in parameter_cell
    assert "MAX_STAGE_SYMBOLS = 192" in parameter_cell
    assert "ROLLOUT_MAX_STAGE_NEW_TOKENS = 180" in parameter_cell
    assert "ROLLOUT_MAX_MOLECULES_PER_SEQUENCE = 16" in parameter_cell
    assert "ROLLOUT_MAX_SEQUENCE_LENGTH = 8192" in parameter_cell
    assert "ROLLOUT_APPEND_PROBABILITY = 0.8" in parameter_cell
    assert "REWARD_DIVERSITY_BETA = 0.6" in parameter_cell
    assert "REWARD_DIVERSITY_WEIGHT = 0.4" in parameter_cell
    assert "REWARD_MATCH_WEIGHT = 1.0" in parameter_cell
    assert 'REWARD_VARIANT = "reward_var2"' in parameter_cell
    assert "REWARD_PLUS_VALID = 0.8" in parameter_cell
    assert 'TEMP_CONFIG_PATH = REPO_DIR / "tmp" / "colab_runtime" / f"{BASE_TRAIN_CONFIG.stem}_03_03.yaml"' in parameter_cell
    assert 'rollout_payload["max_stage_new_tokens"] = int(ROLLOUT_MAX_STAGE_NEW_TOKENS)' in run_cell
    assert 'rollout_payload["max_molecules_per_sequence"] = int(ROLLOUT_MAX_MOLECULES_PER_SEQUENCE)' in run_cell
    assert 'rollout_payload["max_sequence_length"] = int(ROLLOUT_MAX_SEQUENCE_LENGTH)' in run_cell
    assert 'rollout_payload["append_probability"] = float(ROLLOUT_APPEND_PROBABILITY)' in run_cell
    assert 'reward_payload["diversity_beta"] = float(REWARD_DIVERSITY_BETA)' in run_cell
    assert 'reward_payload["diversity_weight"] = float(REWARD_DIVERSITY_WEIGHT)' in run_cell
    assert 'reward_payload["match_weight"] = float(REWARD_MATCH_WEIGHT)' in run_cell
    assert 'reward_payload["reward_variant"] = REWARD_VARIANT' in run_cell
    assert 'reward_payload["plus_valid"] = float(REWARD_PLUS_VALID)' in run_cell
    assert "Tuning guide:" in run_cell
