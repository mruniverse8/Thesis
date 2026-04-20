#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import Counter
import json
import random
import sys
from pathlib import Path
from typing import Any, Iterable

from transformers import AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from post_training.gflownet.config import build_gflownet_config
from post_training.gflownet.model import GFlowNetModel, assert_checkpoint_tokenizer_matches_model
from post_training.gflownet.trainer import MultiMoleculeGFlowNetTrainer
from post_training.gflownet.trajectory import SampledStageTrajectory
from post_training.gflownet.rewarding import build_reward_config
from post_training.sft_multi.dataset import MultiMoleculeDataset
from post_training.shared.config import resolve_gflownet_config_paths, resolve_ppo_checkpoint_source
from src.io_utils import ensure_dir, load_yaml, resolve_path, set_seed, write_json, write_jsonl
from src.training import choose_device


def print_section(title: str) -> None:
    print(f"\n=== {title} ===")


def print_json(title: str, payload: Any) -> None:
    print_section(title)
    print(json.dumps(payload, indent=2, ensure_ascii=False))


def print_records(
    title: str,
    records: Iterable[dict[str, Any]],
    *,
    limit: int | None = None,
    keys: list[str] | None = None,
) -> None:
    payload = list(records)
    total = len(payload)
    if keys is not None:
        payload = [{key: row.get(key) for key in keys} for row in payload]
    shown = payload if limit is None else payload[:limit]
    print_section(f"{title} (showing {len(shown)} of {total})")
    print(json.dumps(shown, indent=2, ensure_ascii=False))


def truncate_text(text: str | None, *, max_chars: int) -> str | None:
    if text is None:
        return None
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3] + "..."


def looks_like_selfies_text(text: str | None) -> bool:
    if text is None:
        return False
    compact = text.strip().replace("<bom>", "").replace("<eom>", "")
    if not compact:
        return False
    if " " in compact:
        return False
    if "[" not in compact or "]" not in compact:
        return False
    return compact.count("[") == compact.count("]")


def sample_review_examples(
    dataset: MultiMoleculeDataset,
    *,
    count: int,
    seed: int,
) -> list[dict[str, object]]:
    if count <= 0:
        raise ValueError("num_examples must be positive.")
    if len(dataset) == 0:
        return []

    effective_count = min(count, len(dataset))
    generator = random.Random(seed)
    indices = generator.sample(range(len(dataset)), k=effective_count)
    return [dataset[index] for index in indices]


def build_trajectory_record(
    trajectory: SampledStageTrajectory,
    *,
    invalid_terminal_reward: float,
) -> dict[str, Any]:
    record = trajectory.to_dict()
    record["num_actions"] = trajectory.num_actions
    record["parsed_selfies_available"] = trajectory.sampled_selfies is not None
    record["looks_like_selfies_stage_text"] = looks_like_selfies_text(trajectory.stage_text)
    record["stage_text_has_spaces"] = bool(trajectory.stage_text.strip() and " " in trajectory.stage_text)
    record["hit_reward_floor"] = abs(trajectory.terminal_reward - invalid_terminal_reward) <= 1.0e-12
    return record


def build_preview_record(
    record: dict[str, Any],
    *,
    max_stage_text_chars: int,
    description_chars: int = 220,
    action_head: int = 24,
    action_tail: int = 8,
) -> dict[str, Any]:
    action_token_ids = list(record.get("action_token_ids", []))
    return {
        "rollout_id": record.get("rollout_id"),
        "example_id": record.get("example_id"),
        "stage_index": record.get("stage_index"),
        "num_actions": record.get("num_actions"),
        "termination_reason": record.get("termination_reason"),
        "stop_token": record.get("stop_token"),
        "terminal_reward": record.get("terminal_reward"),
        "is_valid": record.get("is_valid"),
        "is_duplicate": record.get("is_duplicate"),
        "parsed_selfies_available": record.get("parsed_selfies_available"),
        "looks_like_selfies_stage_text": record.get("looks_like_selfies_stage_text"),
        "sampled_selfies": record.get("sampled_selfies"),
        "description_preview": truncate_text(
            str(record.get("description", "")),
            max_chars=description_chars,
        ),
        "stage_text_preview": truncate_text(
            str(record.get("stage_text", "")),
            max_chars=max_stage_text_chars,
        ),
        "action_token_ids_head": action_token_ids[:action_head],
        "action_token_ids_tail": action_token_ids[-action_tail:] if action_token_ids else [],
        "previous_valid_selfies": record.get("previous_valid_selfies", []),
        "target_selfies_preview": list(record.get("target_selfies_list", []))[:2],
    }


def summarize_trajectories(
    records: list[dict[str, Any]],
    *,
    num_examples_requested: int,
    num_examples_sampled: int,
    dataset_size: int,
    dataset_path: Path,
    resolved_checkpoint_source: str,
    config_path: Path,
    output_dir: Path,
    invalid_terminal_reward: float,
    dataset_split: str,
) -> dict[str, Any]:
    if not records:
        return {
            "config_path": str(config_path),
            "dataset_split": dataset_split,
            "dataset_path": str(dataset_path),
            "dataset_size": dataset_size,
            "num_examples_requested": num_examples_requested,
            "num_examples_sampled": num_examples_sampled,
            "num_trajectories": 0,
            "resolved_checkpoint_source": resolved_checkpoint_source,
            "output_dir": str(output_dir),
            "invalid_terminal_reward": invalid_terminal_reward,
        }

    num_trajectories = len(records)
    termination_counts = Counter(str(record["termination_reason"]) for record in records)
    num_valid = sum(int(bool(record["is_valid"])) for record in records)
    num_duplicates = sum(int(bool(record["is_duplicate"])) for record in records)
    num_parsed = sum(int(bool(record["parsed_selfies_available"])) for record in records)
    num_molecule_like = sum(int(bool(record["looks_like_selfies_stage_text"])) for record in records)
    num_spaces = sum(int(bool(record["stage_text_has_spaces"])) for record in records)
    num_reward_floor = sum(int(bool(record["hit_reward_floor"])) for record in records)
    action_counts = [int(record["num_actions"]) for record in records]
    rewards = [float(record["terminal_reward"]) for record in records]

    return {
        "config_path": str(config_path),
        "dataset_split": dataset_split,
        "dataset_path": str(dataset_path),
        "dataset_size": dataset_size,
        "num_examples_requested": num_examples_requested,
        "num_examples_sampled": num_examples_sampled,
        "num_trajectories": num_trajectories,
        "resolved_checkpoint_source": resolved_checkpoint_source,
        "output_dir": str(output_dir),
        "invalid_terminal_reward": invalid_terminal_reward,
        "valid_fraction": num_valid / num_trajectories,
        "duplicate_fraction": num_duplicates / num_trajectories,
        "parsed_selfies_fraction": num_parsed / num_trajectories,
        "looks_like_selfies_stage_text_fraction": num_molecule_like / num_trajectories,
        "stage_text_with_spaces_fraction": num_spaces / num_trajectories,
        "reward_floor_fraction": num_reward_floor / num_trajectories,
        "stop_token_fraction": termination_counts.get("stop_token", 0) / num_trajectories,
        "max_stage_new_tokens_fraction": termination_counts.get("max_stage_new_tokens", 0)
        / num_trajectories,
        "max_sequence_length_fraction": termination_counts.get("max_sequence_length", 0)
        / num_trajectories,
        "mean_num_actions": sum(action_counts) / num_trajectories,
        "max_num_actions": max(action_counts),
        "mean_terminal_reward": sum(rewards) / num_trajectories,
        "max_terminal_reward": max(rewards),
        "termination_reason_counts": dict(sorted(termination_counts.items())),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Collect and inspect on-policy multi-molecule GFlowNet trajectories without "
            "running optimizer updates."
        )
    )
    parser.add_argument(
        "--config",
        default="configs/multi_molecule_gflownet_mini.yaml",
        help="Path to the YAML GFlowNet config relative to the project root.",
    )
    parser.add_argument(
        "--dataset-split",
        default="train",
        choices=("train", "validation", "test"),
        help="Dataset split to sample examples from.",
    )
    parser.add_argument(
        "--num-examples",
        type=int,
        default=100,
        help="Number of examples to sample without replacement for trajectory review.",
    )
    parser.add_argument(
        "--preview-limit",
        type=int,
        default=20,
        help="Maximum number of rows to print in each preview section.",
    )
    parser.add_argument(
        "--max-stage-text-chars",
        type=int,
        default=480,
        help="Maximum number of stage-text characters shown in printed previews.",
    )
    parser.add_argument(
        "--iteration-index",
        type=int,
        default=1,
        help="Iteration index label used when creating rollout ids.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional override for the sampling seed. Defaults to the config seed.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = resolve_path(args.config, PROJECT_ROOT)
    config = resolve_gflownet_config_paths(load_yaml(config_path), project_root=PROJECT_ROOT)
    seed = int(args.seed if args.seed is not None else config.get("seed", 42))
    set_seed(seed)

    resolved_checkpoint_source = resolve_ppo_checkpoint_source(
        str(config["model"]["checkpoint"]),
        project_root=PROJECT_ROOT,
    )
    resolved_config = dict(config)
    resolved_model_config = dict(resolved_config.get("model", {}))
    resolved_model_config["checkpoint"] = resolved_checkpoint_source
    resolved_config["model"] = resolved_model_config
    config = resolved_config

    dataset_key = f"{args.dataset_split}_file"
    dataset_path = resolve_path(str(config["data"][dataset_key]), PROJECT_ROOT)
    dataset = MultiMoleculeDataset.from_jsonl(dataset_path)
    review_examples = sample_review_examples(
        dataset,
        count=args.num_examples,
        seed=seed,
    )

    gflownet_config = build_gflownet_config(config)
    reward_config = build_reward_config(
        config.get("reward", {}),
        dataset_hint=str(dataset_path),
    )
    device = choose_device(config["training"].get("device", "auto"))

    tokenizer = AutoTokenizer.from_pretrained(resolved_checkpoint_source, use_fast=True)
    tokenizer.model_max_length = int(1.0e9)
    model = GFlowNetModel.from_pretrained(
        resolved_checkpoint_source,
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
        context="GFlowNet verifier checkpoint",
    )
    model.to(device)

    trainer = MultiMoleculeGFlowNetTrainer(
        model=model,
        tokenizer=tokenizer,
        config=gflownet_config,
        reward_config=reward_config,
        device=device,
    )
    trajectories = trainer.collect_on_policy_trajectories(
        review_examples,
        iteration_index=args.iteration_index,
    )

    output_dir = ensure_dir(
        PROJECT_ROOT / "outputs" / "gflownet_trajectory_verifier" / Path(config_path).stem
    )
    run_stem = f"{args.dataset_split}_n{len(review_examples)}_seed{seed}"
    records_path = output_dir / f"{run_stem}_records.jsonl"
    summary_path = output_dir / f"{run_stem}_summary.json"

    records = [
        build_trajectory_record(
            trajectory,
            invalid_terminal_reward=gflownet_config.invalid_terminal_reward,
        )
        for trajectory in trajectories
    ]
    preview_records = [
        build_preview_record(
            record,
            max_stage_text_chars=max(64, args.max_stage_text_chars),
        )
        for record in records
    ]
    summary = summarize_trajectories(
        records,
        num_examples_requested=args.num_examples,
        num_examples_sampled=len(review_examples),
        dataset_size=len(dataset),
        dataset_path=dataset_path,
        resolved_checkpoint_source=resolved_checkpoint_source,
        config_path=config_path,
        output_dir=output_dir,
        invalid_terminal_reward=gflownet_config.invalid_terminal_reward,
        dataset_split=args.dataset_split,
    )

    suspicious_preview = [
        record
        for record in preview_records
        if (
            not bool(record["parsed_selfies_available"])
            or not bool(record["looks_like_selfies_stage_text"])
            or str(record["termination_reason"]) != "stop_token"
            or not bool(record["is_valid"])
        )
    ]
    valid_preview = [
        record
        for record in preview_records
        if bool(record["parsed_selfies_available"]) or bool(record["is_valid"])
    ]
    highest_reward_preview = sorted(
        preview_records,
        key=lambda record: (
            float(record["terminal_reward"] or 0.0),
            int(bool(record["parsed_selfies_available"])),
            int(bool(record["looks_like_selfies_stage_text"])),
        ),
        reverse=True,
    )

    write_jsonl(records_path, records)
    write_json(summary_path, summary)

    selected_examples_preview = [
        {
            "example_id": str(example.get("id")),
            "description_preview": truncate_text(str(example.get("description", "")), max_chars=220),
            "target_count": len(example.get("target_selfies_list", [])),
        }
        for example in review_examples[: args.preview_limit]
    ]

    print_json(
        "Verifier Inputs",
        {
            "config_path": str(config_path),
            "dataset_split": args.dataset_split,
            "dataset_path": str(dataset_path),
            "dataset_size": len(dataset),
            "num_examples_requested": args.num_examples,
            "num_examples_sampled": len(review_examples),
            "preview_limit": args.preview_limit,
            "max_stage_text_chars": args.max_stage_text_chars,
            "resolved_checkpoint_source": resolved_checkpoint_source,
            "device": str(device),
        },
    )
    print_records("Selected examples", selected_examples_preview, limit=args.preview_limit)
    print_json("Trajectory Summary", summary)
    print_records(
        "Suspicious trajectory preview",
        suspicious_preview,
        limit=args.preview_limit,
    )
    print_records(
        "Valid trajectory preview",
        valid_preview,
        limit=args.preview_limit,
    )
    print_records(
        "Highest reward trajectory preview",
        highest_reward_preview,
        limit=args.preview_limit,
    )
    print_json(
        "Verifier Outputs",
        {
            "records_path": str(records_path),
            "summary_path": str(summary_path),
        },
    )


if __name__ == "__main__":
    main()
