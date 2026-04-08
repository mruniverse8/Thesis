from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, T5ForConditionalGeneration

from evaluation import MoleculeMetricConfig, build_prediction_metric_report, load_reference_index

from .datasets import TextToSelfiesCollator, TextToSelfiesDataset, move_tensor_batch_to_device
from .io_utils import resolve_path, write_json, write_jsonl
from .prompting import unwrap_selfies_target
from .selfies_utils import decode_selfies_to_smiles, normalize_generated_selfies
from .tokenizer_utils import build_decoder_tokenizer
from .training import choose_device, resolve_mixed_precision


@dataclass(frozen=True)
class CheckpointEvaluationResult:
    generation_metrics: dict[str, float]
    prediction_path: Path
    molecule_metrics: dict[str, Any] | None = None
    molecule_metrics_path: Path | None = None
    molecule_metrics_by_group: list[dict[str, Any]] | None = None
    group_metrics_path: Path | None = None


def load_decoder_tokenizer(checkpoint_dir: Path, config: dict[str, Any], training_tokenizer: Any) -> Any:
    saved_decoder_path = checkpoint_dir / "decoder_tokenizer"
    if saved_decoder_path.exists():
        tokenizer = AutoTokenizer.from_pretrained(saved_decoder_path, use_fast=True)
        tokenizer.model_max_length = int(1e9)
        return tokenizer
    return build_decoder_tokenizer(config["model"]["base_tokenizer_name"], training_tokenizer)


def generate_predictions(
    model: T5ForConditionalGeneration,
    training_tokenizer: Any,
    decoder_tokenizer: Any,
    dataloader: DataLoader,
    device: torch.device,
    mixed_precision: str,
    generation_config: dict[str, Any],
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    del training_tokenizer  # not needed yet, kept here for future task-specific extensions

    model.eval()

    predictions: list[dict[str, Any]] = []
    total_examples = 0
    exact_matches = 0
    valid_predictions = 0
    repaired_predictions = 0

    autocast_dtype = {
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
    }.get(mixed_precision)

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Generating", leave=False):
            tensor_batch = move_tensor_batch_to_device(batch, device)

            autocast_enabled = device.type == "cuda" and autocast_dtype is not None
            autocast_context = (
                torch.autocast(device_type="cuda", dtype=autocast_dtype)
                if autocast_enabled
                else nullcontext()
            )

            with autocast_context:
                generated_ids = model.generate(
                    input_ids=tensor_batch["input_ids"],
                    attention_mask=tensor_batch["attention_mask"],
                    max_new_tokens=int(generation_config["max_new_tokens"]),
                    num_beams=int(generation_config["num_beams"]),
                )

            decoded_outputs = decoder_tokenizer.batch_decode(
                generated_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True,
            )

            for example_id, description, target_text, raw_prediction in zip(
                batch["example_ids"],
                batch["descriptions"],
                batch["target_texts"],
                decoded_outputs,
            ):
                target_selfies = unwrap_selfies_target(target_text)
                predicted_selfies = normalize_generated_selfies(raw_prediction)
                predicted_smiles, repaired = decode_selfies_to_smiles(predicted_selfies)
                target_smiles, _ = decode_selfies_to_smiles(target_selfies)

                is_exact_match = predicted_selfies == target_selfies
                is_valid = predicted_smiles is not None

                total_examples += 1
                exact_matches += int(is_exact_match)
                valid_predictions += int(is_valid)
                repaired_predictions += int(repaired)

                predictions.append(
                    {
                        "id": example_id,
                        "description": description,
                        "target_selfies": target_selfies,
                        "target_smiles": target_smiles,
                        "prediction_text": raw_prediction.strip(),
                        "prediction_selfies": predicted_selfies,
                        "prediction_smiles": predicted_smiles,
                        "exact_match": is_exact_match,
                        "is_valid_selfies": is_valid,
                        "used_repair": repaired,
                    }
                )

    metrics = {
        "num_examples": total_examples,
        "exact_match": exact_matches / total_examples if total_examples else 0.0,
        "valid_selfies_rate": valid_predictions / total_examples if total_examples else 0.0,
        "repaired_selfies_rate": repaired_predictions / total_examples if total_examples else 0.0,
    }
    return metrics, predictions


def evaluate_prediction_molecules(
    predictions: Sequence[dict[str, Any]],
    *,
    dataset_path: str | Path,
    metric_config: MoleculeMetricConfig | None = None,
    summary_path: str | Path | None = None,
    by_group_path: str | Path | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]], Path | None, Path | None]:
    resolved_dataset_path = resolve_path(dataset_path)
    reference_index = load_reference_index(resolved_dataset_path)
    summary, by_group = build_prediction_metric_report(
        predictions,
        reference_index,
        config=metric_config,
    )

    resolved_summary_path = resolve_path(summary_path) if summary_path else None
    resolved_by_group_path = resolve_path(by_group_path) if by_group_path else None

    if resolved_summary_path is not None:
        write_json(resolved_summary_path, summary)
    if resolved_by_group_path is not None:
        write_jsonl(resolved_by_group_path, by_group)

    return summary, by_group, resolved_summary_path, resolved_by_group_path


def evaluate_checkpoint(
    config: dict[str, Any],
    checkpoint_path: str | Path,
    split: str,
    prediction_path: str | Path | None = None,
    molecule_metrics_path: str | Path | None = None,
    group_metrics_path: str | Path | None = None,
    metric_config: MoleculeMetricConfig | None = None,
    compute_molecule_metrics: bool = True,
) -> CheckpointEvaluationResult:
    checkpoint_dir = resolve_path(checkpoint_path)
    split_key = f"{split}_file"
    if split_key not in config["data"]:
        raise ValueError(f"Unknown split {split!r}; expected one of train/validation/test")

    dataset_path = resolve_path(config["data"][split_key])
    training_tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir, use_fast=True)
    decoder_tokenizer = load_decoder_tokenizer(checkpoint_dir, config, training_tokenizer)

    model = T5ForConditionalGeneration.from_pretrained(checkpoint_dir)
    device = choose_device(config["training"].get("device", "auto"))
    model.to(device)

    dataset = TextToSelfiesDataset.from_jsonl(dataset_path)
    collator = TextToSelfiesCollator(
        tokenizer=training_tokenizer,
        max_source_length=int(config["data"]["max_source_length"]),
        max_target_length=int(config["data"]["max_target_length"]),
    )
    dataloader = DataLoader(
        dataset,
        batch_size=int(config["training"]["per_device_eval_batch_size"]),
        shuffle=False,
        num_workers=int(config["data"].get("num_workers", 0)),
        collate_fn=collator,
        pin_memory=device.type == "cuda",
    )

    mixed_precision = resolve_mixed_precision(config["training"].get("mixed_precision", "auto"), device)
    metrics, predictions = generate_predictions(
        model=model,
        training_tokenizer=training_tokenizer,
        decoder_tokenizer=decoder_tokenizer,
        dataloader=dataloader,
        device=device,
        mixed_precision=mixed_precision,
        generation_config={
            "max_new_tokens": config["training"]["generation_max_new_tokens"],
            "num_beams": config["training"]["num_beams"],
        },
    )

    output_path = resolve_path(prediction_path) if prediction_path else checkpoint_dir / f"{split}_predictions.jsonl"
    write_jsonl(output_path, predictions)
    write_json(checkpoint_dir / f"{split}_generation_metrics.json", metrics)

    molecule_summary: dict[str, Any] | None = None
    molecule_by_group: list[dict[str, Any]] | None = None
    resolved_molecule_metrics_path: Path | None = None
    resolved_group_metrics_path: Path | None = None

    if compute_molecule_metrics:
        default_summary_path = checkpoint_dir / f"{split}_molecule_metrics.json"
        default_group_path = checkpoint_dir / f"{split}_molecule_metrics_by_group.jsonl"
        (
            molecule_summary,
            molecule_by_group,
            resolved_molecule_metrics_path,
            resolved_group_metrics_path,
        ) = evaluate_prediction_molecules(
            predictions,
            dataset_path=dataset_path,
            metric_config=metric_config,
            summary_path=molecule_metrics_path or default_summary_path,
            by_group_path=group_metrics_path or default_group_path,
        )

    return CheckpointEvaluationResult(
        generation_metrics=metrics,
        prediction_path=output_path,
        molecule_metrics=molecule_summary,
        molecule_metrics_path=resolved_molecule_metrics_path,
        molecule_metrics_by_group=molecule_by_group,
        group_metrics_path=resolved_group_metrics_path,
    )
