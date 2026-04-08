#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from post_training.config_utils import resolve_ppo_config_paths
from post_training.policy_model import PolicyValueModel, load_reference_model
from post_training.ppo_trainer import MoleculeWisePPOTrainer
from post_training.ppo_types import PPOConfig
from post_training.sft_dataset import MultiMoleculeDataset
from reward_utils.defaults import RewardConfig
from src.io_utils import ensure_dir, load_yaml, resolve_path, set_seed, write_json
from src.training import choose_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the molecule-wise PPO post-training stage.")
    parser.add_argument(
        "--config",
        default="configs/molecule_wise_ppo.yaml",
        help="Path to the YAML PPO config relative to the project root.",
    )
    return parser.parse_args()


def build_ppo_config(config: dict[str, Any]) -> PPOConfig:
    ppo_payload = dict(config.get("ppo", {}))
    training_config = config.get("training", {})
    model_config = config.get("model", {})

    ppo_payload.setdefault("output_dir", training_config["output_dir"])
    ppo_payload.setdefault(
        "save_every_iterations",
        training_config.get("save_every_iterations", PPOConfig.save_every_iterations),
    )
    ppo_payload.setdefault("use_lora", model_config.get("use_lora", PPOConfig.use_lora))
    ppo_payload.setdefault(
        "freeze_base_model_without_lora",
        model_config.get(
            "freeze_base_model_without_lora",
            PPOConfig.freeze_base_model_without_lora,
        ),
    )
    ppo_payload.setdefault("lora_rank", model_config.get("lora_rank", PPOConfig.lora_rank))
    ppo_payload.setdefault("lora_alpha", model_config.get("lora_alpha", PPOConfig.lora_alpha))
    ppo_payload.setdefault(
        "lora_dropout",
        model_config.get("lora_dropout", PPOConfig.lora_dropout),
    )
    ppo_payload.setdefault(
        "target_modules",
        model_config.get("target_modules", list(PPOConfig.target_modules)),
    )

    rollout_payload = dict(ppo_payload.get("rollout", {}))
    rollout_payload.setdefault(
        "max_source_length",
        config.get("data", {}).get("max_source_length", 512),
    )
    ppo_payload["rollout"] = rollout_payload
    return PPOConfig.from_dict(ppo_payload)


def sample_examples(dataset: MultiMoleculeDataset, count: int) -> list[dict[str, Any]]:
    return [dataset[random.randrange(len(dataset))] for _ in range(count)]


def main() -> None:
    args = parse_args()
    config_path = resolve_path(args.config, PROJECT_ROOT)
    config = resolve_ppo_config_paths(load_yaml(config_path), project_root=PROJECT_ROOT)
    set_seed(int(config.get("seed", 42)))

    checkpoint_path = resolve_path(config["model"]["checkpoint"], PROJECT_ROOT)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_path, use_fast=True)
    tokenizer.model_max_length = int(1e9)

    ppo_config = build_ppo_config(config)
    reward_config = RewardConfig(**config.get("reward", {}))
    device = choose_device(config["training"].get("device", "auto"))

    policy_model = PolicyValueModel.from_pretrained(
        checkpoint_path,
        tokenizer_size=len(tokenizer),
        use_lora=ppo_config.use_lora,
        lora_rank=ppo_config.lora_rank,
        lora_alpha=ppo_config.lora_alpha,
        lora_dropout=ppo_config.lora_dropout,
        target_modules=ppo_config.target_modules,
        freeze_base_model_without_lora=ppo_config.freeze_base_model_without_lora,
    )
    policy_model.to(device)
    reference_model = load_reference_model(checkpoint_path, tokenizer_size=len(tokenizer))
    reference_model.to(device)

    trainer = MoleculeWisePPOTrainer(
        policy_model=policy_model,
        reference_model=reference_model,
        tokenizer=tokenizer,
        config=ppo_config,
        reward_config=reward_config,
        device=device,
    )

    train_dataset = MultiMoleculeDataset.from_jsonl(config["data"]["train_file"])
    output_dir = ensure_dir(ppo_config.output_dir)

    history: list[dict[str, float]] = []
    for iteration in range(1, ppo_config.ppo_iterations + 1):
        iteration_examples = sample_examples(train_dataset, ppo_config.batch_size)
        metrics = trainer.train_iteration(iteration_examples, iteration_index=iteration)
        history.append(metrics)
        write_json(output_dir / "history.json", history)
        print(json.dumps(metrics, indent=2))

    summary = {
        "output_dir": str(output_dir),
        "num_iterations": ppo_config.ppo_iterations,
        "history": history,
    }
    write_json(output_dir / "run_summary.json", summary)


if __name__ == "__main__":
    main()
