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


def test_kaggle_gflownet_eval_notebook_uses_two_gpu_process_sharding_and_exact_merge() -> None:
    notebook = _load_notebook("kaggle/07_evaluate_multi_molecule_gflownet_lpm24.ipynb")

    assert notebook["nbformat"] == 4
    assert len(notebook["cells"]) == 8

    all_source = "\n".join(_joined_source(cell) for cell in notebook["cells"])
    title_cell = _joined_source(notebook["cells"][0])
    setup_cell = _joined_source(notebook["cells"][1])
    parameter_cell = _joined_source(notebook["cells"][2])
    run_cell = _joined_source(notebook["cells"][6])

    assert "2-GPU Sharding" in title_cell
    assert 'REPO_DIR = Path("/kaggle/working/Thesis")' in setup_cell
    assert 'REPO_BRANCH = "gflownet_v2.4.e"' in setup_cell
    assert 'ENABLE_MULTI_GPU_EVAL = True' in parameter_cell
    assert "MAX_EVAL_WORKERS = 2" in parameter_cell
    assert "VALIDATION_NUM_EXAMPLES = 128" in parameter_cell
    assert "PROCESSED_TEST_FRACTION = 0.50" in parameter_cell
    assert "EVAL_BATCH_SIZE = 8" in parameter_cell
    assert 'VALIDATION_PROGRESS_SUMMARY_PATH = DIAGNOSTICS_DIR / "gflownet_validation_progress_summary.json"' in parameter_cell
    assert 'PROCESSED_TEST_PROGRESS_SUMMARY_PATH = DIAGNOSTICS_DIR / "gflownet_processed_test_progress_summary.json"' in parameter_cell
    assert "scripts/evaluate_multi_molecule_gflownet.py" in all_source
    assert "CUDA_VISIBLE_DEVICES" in all_source
    assert "--worker-shard-index" in all_source
    assert "--worker-num-shards" in all_source
    assert "--cuda-device" in all_source
    assert "torch.cuda.device_count()" in all_source
    assert "parallel_mode = parallel_workers > 1" in run_cell
    assert "parallel_workers = max(1, min(int(MAX_EVAL_WORKERS), detected_gpu_count))" in run_cell
    assert "IncrementalRolloutMetrics.from_state_dict" in all_source
    assert "sort_generation_rows_by_global_example_index(merged_rows)" in all_source
    assert "evaluate_generation_rows(ordered_rows, metric_config=metric_config)" in all_source
    assert "wandb.log(evaluation_diagnosis, step=result.num_groups)" in all_source


def test_multi_molecule_gflownet_eval_script_exposes_worker_sharding_outputs() -> None:
    script_source = (PROJECT_ROOT / "scripts" / "evaluate_multi_molecule_gflownet.py").read_text(
        encoding="utf-8"
    )

    assert "--worker-shard-index" in script_source
    assert "--worker-num-shards" in script_source
    assert "--cuda-device" in script_source
    assert "global_example_index" in script_source
    assert "selected_dataset_index" in script_source
    assert "parallel_mode = int(args.worker_num_shards) > 1" in script_source
    assert "write_json(output_paths[\"rollout_state_path\"], rollout_metrics.to_state_dict())" in script_source
