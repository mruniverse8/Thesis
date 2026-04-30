#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

import torch
from transformers import AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation_metrics import EvaluationMetricConfig
from notebooks.gflownet_eval_parallel import (
    batched,
    build_final_metrics_payload,
    build_generation_group_and_row,
    build_progress_payload,
    compact_metrics,
    select_examples,
    shard_selected_examples,
)
from notebooks.gflownet_eval_streaming import IncrementalGenerationMetrics, IncrementalRolloutMetrics
from post_training.gflownet import (
    GFlowNetModel,
    build_gflownet_config,
    build_reward_config,
    sample_stage_trajectories_for_examples,
)
from post_training.gflownet.model import assert_checkpoint_tokenizer_matches_model
from post_training.shared.config import resolve_gflownet_config_paths
from post_training.shared.decoding import build_stage_token_constraints
from post_training.sft_multi.dataset import MultiMoleculeDataset
from src.io_utils import load_yaml, resolve_path, set_seed, write_json, write_jsonl
from src.training import choose_device


def parse_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected a boolean value, got: {value!r}")


def append_jsonl(path: Path, row: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False))
        handle.write("\n")


def build_output_paths(args: argparse.Namespace) -> dict[str, Path]:
    worker_output_dir = Path(args.worker_output_dir).expanduser()
    return {
        "worker_output_dir": worker_output_dir,
        "metrics_path": Path(args.metrics_path).expanduser()
        if args.metrics_path
        else worker_output_dir / "metrics.json",
        "generations_path": Path(args.generations_path).expanduser()
        if args.generations_path
        else worker_output_dir / "generations.jsonl",
        "progress_path": Path(args.progress_path).expanduser()
        if args.progress_path
        else worker_output_dir / "progress_metrics.jsonl",
        "rollout_state_path": Path(args.rollout_state_path).expanduser()
        if args.rollout_state_path
        else worker_output_dir / "rollout_state.json",
        "summary_path": Path(args.summary_path).expanduser()
        if args.summary_path
        else worker_output_dir / "summary.json",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate one serial or sharded slice of the multi-molecule GFlowNet "
            "generation benchmark and write shard-local artifacts."
        )
    )
    parser.add_argument("--config-path", required=True, help="Path to the YAML GFlowNet config.")
    parser.add_argument("--checkpoint-path", required=True, help="Path to the checkpoint used for evaluation.")
    parser.add_argument("--dataset-path", required=True, help="Path to the grouped evaluation JSONL dataset.")
    parser.add_argument("--split-name", required=True, help="Split name used in output metadata.")
    parser.add_argument("--worker-output-dir", required=True, help="Directory used for shard-local outputs.")
    parser.add_argument("--metrics-path", default=None, help="Optional shard-local metrics JSON path.")
    parser.add_argument("--generations-path", default=None, help="Optional shard-local generations JSONL path.")
    parser.add_argument("--progress-path", default=None, help="Optional shard-local progress JSONL path.")
    parser.add_argument("--rollout-state-path", default=None, help="Optional shard-local rollout-state JSON path.")
    parser.add_argument("--summary-path", default=None, help="Optional shard-local summary JSON path.")
    parser.add_argument("--seed", type=int, default=42, help="Deterministic selection and rollout seed.")
    parser.add_argument(
        "--fraction",
        type=float,
        default=None,
        help="Optional dataset fraction used for selection before sharding.",
    )
    parser.add_argument(
        "--max-examples",
        type=int,
        default=None,
        help="Optional hard cap for selected examples before sharding.",
    )
    parser.add_argument(
        "--sample-with-replacement",
        type=parse_bool,
        default=False,
        help="Whether dataset selection should sample with replacement.",
    )
    parser.add_argument(
        "--report-every-examples",
        type=int,
        default=50,
        help="Emit shard-local progress rows every N evaluated examples; 0 disables progress rows.",
    )
    parser.add_argument("--eval-batch-size", type=int, default=8, help="Generation batch size.")
    parser.add_argument(
        "--acceptance-dice-threshold",
        type=float,
        default=0.7,
        help="Acceptance Dice threshold used for exact generation metrics.",
    )
    parser.add_argument(
        "--n-circles-tanimoto-threshold",
        type=float,
        default=0.6,
        help="NCircles Tanimoto threshold stored in the metric config.",
    )
    parser.add_argument("--worker-shard-index", type=int, default=0, help="0-based contiguous shard index.")
    parser.add_argument("--worker-num-shards", type=int, default=1, help="Total number of contiguous shards.")
    parser.add_argument(
        "--cuda-device",
        default="auto",
        help="Explicit device string. In multi-GPU mode the parent should also set CUDA_VISIBLE_DEVICES.",
    )
    parser.add_argument(
        "--rollout-decoding-strategy",
        default=None,
        help="Optional override for the rollout decoding strategy.",
    )
    parser.add_argument(
        "--rollout-num-beams",
        type=int,
        default=None,
        help="Optional override for rollout beam count.",
    )
    parser.add_argument(
        "--rollout-length-penalty",
        type=float,
        default=None,
        help="Optional override for rollout length penalty.",
    )
    parser.add_argument(
        "--rollout-early-stopping",
        type=parse_bool,
        default=None,
        help="Optional override for rollout early_stopping.",
    )
    parser.add_argument(
        "--rollout-temperature",
        type=float,
        default=None,
        help="Optional override for rollout temperature.",
    )
    parser.add_argument(
        "--rollout-top-p",
        type=float,
        default=None,
        help="Optional override for rollout top-p.",
    )
    return parser.parse_args()


def apply_rollout_overrides(
    config: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    resolved = dict(config)
    gflownet_payload = dict(resolved.get("gflownet", {}))
    rollout_payload = dict(gflownet_payload.get("rollout", {}))
    if args.rollout_decoding_strategy is not None:
        rollout_payload["decoding_strategy"] = str(args.rollout_decoding_strategy)
    if args.rollout_num_beams is not None:
        rollout_payload["num_beams"] = int(args.rollout_num_beams)
    if args.rollout_length_penalty is not None:
        rollout_payload["length_penalty"] = float(args.rollout_length_penalty)
    if args.rollout_early_stopping is not None:
        rollout_payload["early_stopping"] = bool(args.rollout_early_stopping)
    if args.rollout_temperature is not None:
        rollout_payload["temperature"] = float(args.rollout_temperature)
    if args.rollout_top_p is not None:
        rollout_payload["top_p"] = float(args.rollout_top_p)
    gflownet_payload["rollout"] = rollout_payload
    resolved["gflownet"] = gflownet_payload
    return resolved


def main() -> None:
    args = parse_args()
    output_paths = build_output_paths(args)
    output_paths["worker_output_dir"].mkdir(parents=True, exist_ok=True)
    output_paths["progress_path"].parent.mkdir(parents=True, exist_ok=True)
    output_paths["progress_path"].write_text("", encoding="utf-8")

    config_path = resolve_path(args.config_path, PROJECT_ROOT)
    checkpoint_path = resolve_path(args.checkpoint_path, PROJECT_ROOT)
    dataset_path = resolve_path(args.dataset_path, PROJECT_ROOT)

    config = resolve_gflownet_config_paths(load_yaml(config_path), project_root=PROJECT_ROOT)
    config = apply_rollout_overrides(config, args)
    config.setdefault("model", {})["checkpoint"] = str(checkpoint_path)

    set_seed(int(args.seed))
    dataset = MultiMoleculeDataset.from_jsonl(dataset_path)
    selected_examples, selected_indices = select_examples(
        dataset,
        max_examples=args.max_examples,
        fraction=args.fraction,
        seed=int(args.seed),
        with_replacement=bool(args.sample_with_replacement),
    )
    shard_examples, shard_selected_indices, shard = shard_selected_examples(
        selected_examples,
        selected_indices,
        shard_index=int(args.worker_shard_index),
        num_shards=int(args.worker_num_shards),
    )

    metric_config = EvaluationMetricConfig(
        acceptance_dice_threshold=float(args.acceptance_dice_threshold),
        compute_n_circles=False,
        n_circles_tanimoto_threshold=float(args.n_circles_tanimoto_threshold),
    )
    gflownet_config = build_gflownet_config(config)
    reward_config = build_reward_config(
        config.get("reward", {}),
        dataset_hint=str(dataset_path),
    )
    requested_device = args.cuda_device
    if str(requested_device).strip().lower() == "auto":
        requested_device = str(config.get("training", {}).get("device", "auto"))
    device = choose_device(str(requested_device))

    tokenizer = AutoTokenizer.from_pretrained(checkpoint_path, use_fast=True)
    tokenizer.model_max_length = int(1.0e9)
    model = GFlowNetModel.from_pretrained(
        checkpoint_path,
        use_lora=gflownet_config.use_lora,
        lora_rank=gflownet_config.lora_rank,
        lora_alpha=gflownet_config.lora_alpha,
        lora_dropout=gflownet_config.lora_dropout,
        target_modules=gflownet_config.target_modules,
        freeze_base_model_without_lora=gflownet_config.freeze_base_model_without_lora,
    )
    assert_checkpoint_tokenizer_matches_model(
        tokenizer,
        model.policy_model,
        context="GFlowNet evaluation checkpoint",
    )
    model.to(device)
    model.eval()

    if gflownet_config.rollout.constrained_decoding:
        validation_dataset_path = resolve_path(
            str(config["data"]["validation_file"]),
            PROJECT_ROOT,
        )
        validation_dataset = MultiMoleculeDataset.from_jsonl(validation_dataset_path)
        constraints = build_stage_token_constraints(
            tokenizer,
            validation_dataset,
            selfies_dict_path=gflownet_config.rollout.selfies_dict_path,
            separator_token=gflownet_config.rollout.stage_separator,
        )
        model.set_stage_token_constraints(constraints)

    parallel_mode = int(args.worker_num_shards) > 1
    generation_metrics = IncrementalGenerationMetrics(metric_config)
    rollout_metrics = IncrementalRolloutMetrics(
        max_molecules_per_sequence=gflownet_config.rollout.max_molecules_per_sequence,
        invalid_terminal_reward=gflownet_config.invalid_terminal_reward,
    )
    generation_rows: list[dict[str, object]] = []
    rng = random.Random(int(args.seed))
    report_every_examples = int(args.report_every_examples)
    should_report_progress = report_every_examples > 0

    for batch_start, example_batch in batched(shard_examples, int(args.eval_batch_size)):
        batch_global_indices = list(
            range(shard.start + batch_start, shard.start + batch_start + len(example_batch))
        )
        rollout_ids = [
            f"{args.split_name}-eval-{global_example_index:06d}-{example['id']}"
            for global_example_index, example in zip(batch_global_indices, example_batch)
        ]
        with torch.no_grad():
            trajectory_batches = sample_stage_trajectories_for_examples(
                model,
                tokenizer,
                example_batch,
                rollout_ids=rollout_ids,
                generation_config=gflownet_config.rollout,
                reward_config=reward_config,
                invalid_terminal_reward=gflownet_config.invalid_terminal_reward,
                device=device,
                rng=rng,
                return_last_valid_trajectory_only=False,
            )

        batch_selected_indices = shard_selected_indices[batch_start : batch_start + len(example_batch)]
        for example, trajectories, global_example_index, selected_dataset_index in zip(
            example_batch,
            trajectory_batches,
            batch_global_indices,
            batch_selected_indices,
        ):
            group, row = build_generation_group_and_row(
                example,
                trajectories,
                split_name=args.split_name,
                global_example_index=global_example_index,
                selected_dataset_index=selected_dataset_index,
                parallel_mode=parallel_mode,
                parallel_workers=int(args.worker_num_shards),
                worker_shard_index=int(args.worker_shard_index),
                worker_num_shards=int(args.worker_num_shards),
            )
            generation_metrics.update(group)
            rollout_metrics.update(trajectories)
            generation_rows.append(row)

            evaluated_examples = generation_metrics.num_groups
            if should_report_progress and evaluated_examples % report_every_examples == 0:
                partial_result = generation_metrics.to_result()
                partial_rollout_diagnostics = rollout_metrics.to_metrics()
                progress_payload = build_progress_payload(
                    partial_result,
                    split_name=args.split_name,
                    dataset_path=dataset_path,
                    dataset_size=len(dataset),
                    num_selected_examples=len(selected_examples),
                    evaluated_examples=evaluated_examples,
                    selected_fraction=args.fraction,
                    sample_with_replacement=bool(args.sample_with_replacement),
                    selected_indices=selected_indices,
                    mean_trajectory_length=partial_rollout_diagnostics["mean_trajectory_length"],
                    parallel_mode=parallel_mode,
                    parallel_workers=int(args.worker_num_shards),
                    worker_shard_index=int(args.worker_shard_index),
                    worker_num_shards=int(args.worker_num_shards),
                    shard_start_index=shard.start,
                    shard_end_index=shard.end,
                    selection_seed=int(args.seed),
                    eval_batch_size=int(args.eval_batch_size),
                )
                append_jsonl(output_paths["progress_path"], progress_payload)
                print(
                    compact_metrics(
                        partial_result,
                        split_name=args.split_name,
                        progress_examples=evaluated_examples,
                        mean_trajectory_length=partial_rollout_diagnostics["mean_trajectory_length"],
                    )
                )

    result = generation_metrics.to_result()
    rollout_diagnostics = rollout_metrics.to_metrics()
    metrics_payload = build_final_metrics_payload(
        result,
        split_name=args.split_name,
        dataset_path=dataset_path,
        dataset_size=len(dataset),
        num_selected_examples=len(shard_examples),
        selected_fraction=args.fraction,
        sample_with_replacement=bool(args.sample_with_replacement),
        selected_indices=selected_indices,
        report_every_examples=report_every_examples,
        eval_batch_size=int(args.eval_batch_size),
        checkpoint_for_eval=checkpoint_path,
        rollout_diagnostics=rollout_diagnostics,
        selection_seed=int(args.seed),
        parallel_mode=parallel_mode,
        parallel_workers=int(args.worker_num_shards),
        worker_shard_index=int(args.worker_shard_index),
        worker_num_shards=int(args.worker_num_shards),
        shard_start_index=shard.start,
        shard_end_index=shard.end,
        extra_metadata={
            "config_path": str(config_path),
            "global_num_selected_examples": len(selected_examples),
            "num_shard_examples": len(shard_examples),
            "resolved_device": str(device),
        },
    )
    summary_payload = {
        "split_name": args.split_name,
        "dataset_path": str(dataset_path),
        "dataset_size": len(dataset),
        "worker_shard_index": int(args.worker_shard_index),
        "worker_num_shards": int(args.worker_num_shards),
        "shard_start_index": shard.start,
        "shard_end_index": shard.end,
        "num_selected_examples": len(selected_examples),
        "num_shard_examples": len(shard_examples),
        "checkpoint_for_eval": str(checkpoint_path),
        "config_path": str(config_path),
        "resolved_device": str(device),
        "parallel_mode": parallel_mode,
        "parallel_workers": int(args.worker_num_shards),
        "metrics_path": str(output_paths["metrics_path"]),
        "generations_path": str(output_paths["generations_path"]),
        "progress_path": str(output_paths["progress_path"]),
        "rollout_state_path": str(output_paths["rollout_state_path"]),
        "mean_trajectory_length": float(rollout_diagnostics["mean_trajectory_length"]),
    }

    write_json(output_paths["metrics_path"], metrics_payload)
    write_jsonl(output_paths["generations_path"], generation_rows)
    write_json(output_paths["rollout_state_path"], rollout_metrics.to_state_dict())
    write_json(output_paths["summary_path"], summary_payload)

    print(
        json.dumps(
            {
                "split_name": args.split_name,
                "worker_shard_index": int(args.worker_shard_index),
                "worker_num_shards": int(args.worker_num_shards),
                "num_selected_examples": len(selected_examples),
                "num_shard_examples": len(shard_examples),
                "metrics_path": str(output_paths["metrics_path"]),
                "generations_path": str(output_paths["generations_path"]),
                "progress_path": str(output_paths["progress_path"]),
                "rollout_state_path": str(output_paths["rollout_state_path"]),
                "summary_path": str(output_paths["summary_path"]),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
