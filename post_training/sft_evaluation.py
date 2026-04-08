from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path
from typing import Any

import selfies as sf
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, T5ForConditionalGeneration

from reward_utils.validation import parse_molecule_text
from src.datasets import move_tensor_batch_to_device
from src.evaluation import load_decoder_tokenizer
from src.io_utils import resolve_path, write_json, write_jsonl
from src.training import choose_device, resolve_mixed_precision

from .sft_dataset import MultiMoleculeCollator, MultiMoleculeDataset
from .sequence_format import parse_molecule_sequence


def _canonicalize_selfies_list(selfies_list: list[str]) -> tuple[list[str | None], bool]:
    canonical_smiles: list[str | None] = []
    all_valid = True

    for selfies_text in selfies_list:
        try:
            record = parse_molecule_text(selfies_text, representation="selfies")
        except ImportError:
            try:
                canonical_smiles.append(sf.decoder(selfies_text))
            except Exception:
                canonical_smiles.append(None)
                all_valid = False
            continue

        if not record.is_valid or record.canonical_smiles is None:
            canonical_smiles.append(None)
            all_valid = False
            continue
        canonical_smiles.append(record.canonical_smiles)

    return canonical_smiles, all_valid


def _position_exact_match_rate(predicted: list[str], target: list[str]) -> float:
    denominator = max(len(predicted), len(target), 1)
    matches = sum(
        1 for predicted_item, target_item in zip(predicted, target) if predicted_item == target_item
    )
    return matches / denominator


def generate_multi_molecule_predictions(
    model: T5ForConditionalGeneration,
    decoder_tokenizer: Any,
    dataloader: DataLoader,
    device: torch.device,
    mixed_precision: str,
    generation_config: dict[str, Any],
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    model.eval()

    predictions: list[dict[str, Any]] = []
    total_examples = 0
    exact_sequence_matches = 0
    exact_set_matches = 0
    all_molecules_valid = 0
    duplicate_predictions = 0
    total_position_match_rate = 0.0
    total_predicted_molecule_count = 0
    total_unique_predicted_molecule_count = 0

    autocast_dtype = {"fp16": torch.float16, "bf16": torch.bfloat16}.get(mixed_precision)

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
                skip_special_tokens=False,
                clean_up_tokenization_spaces=True,
            )

            for example_id, description, target_selfies_list, raw_prediction in zip(
                batch["example_ids"],
                batch["descriptions"],
                batch["target_selfies_lists"],
                decoded_outputs,
            ):
                predicted_selfies_list = parse_molecule_sequence(raw_prediction)
                predicted_canonical_smiles, prediction_all_valid = _canonicalize_selfies_list(
                    predicted_selfies_list
                )
                target_canonical_smiles, _ = _canonicalize_selfies_list(target_selfies_list)

                predicted_valid_set = {item for item in predicted_canonical_smiles if item}
                target_valid_set = {item for item in target_canonical_smiles if item}
                has_duplicates = len(predicted_valid_set) < len(
                    [item for item in predicted_canonical_smiles if item]
                )

                exact_sequence_match = predicted_selfies_list == target_selfies_list
                exact_set_match = (
                    prediction_all_valid
                    and all(item is not None for item in target_canonical_smiles)
                    and predicted_valid_set == target_valid_set
                )
                position_match_rate = _position_exact_match_rate(
                    predicted_selfies_list,
                    target_selfies_list,
                )

                total_examples += 1
                exact_sequence_matches += int(exact_sequence_match)
                exact_set_matches += int(exact_set_match)
                all_molecules_valid += int(prediction_all_valid and bool(predicted_selfies_list))
                duplicate_predictions += int(has_duplicates)
                total_position_match_rate += position_match_rate
                total_predicted_molecule_count += len(predicted_selfies_list)
                total_unique_predicted_molecule_count += len(predicted_valid_set)

                predictions.append(
                    {
                        "id": example_id,
                        "description": description,
                        "target_selfies_list": list(target_selfies_list),
                        "target_canonical_smiles_list": target_canonical_smiles,
                        "prediction_text": raw_prediction.strip(),
                        "predicted_selfies_list": predicted_selfies_list,
                        "predicted_canonical_smiles_list": predicted_canonical_smiles,
                        "exact_sequence_match": exact_sequence_match,
                        "exact_set_match": exact_set_match,
                        "all_molecules_valid": prediction_all_valid,
                        "has_duplicate_prediction": has_duplicates,
                        "position_exact_match_rate": position_match_rate,
                    }
                )

    metrics = {
        "num_examples": total_examples,
        "exact_sequence_match": exact_sequence_matches / total_examples if total_examples else 0.0,
        "exact_set_match": exact_set_matches / total_examples if total_examples else 0.0,
        "all_molecules_valid_rate": all_molecules_valid / total_examples if total_examples else 0.0,
        "mean_predicted_molecule_count": (
            total_predicted_molecule_count / total_examples if total_examples else 0.0
        ),
        "mean_unique_predicted_molecule_count": (
            total_unique_predicted_molecule_count / total_examples if total_examples else 0.0
        ),
        "duplicate_prediction_rate": (
            duplicate_predictions / total_examples if total_examples else 0.0
        ),
        "position_exact_match_rate": (
            total_position_match_rate / total_examples if total_examples else 0.0
        ),
    }
    return metrics, predictions


def evaluate_multi_molecule_checkpoint(
    config: dict[str, Any],
    checkpoint_path: str | Path,
    split: str,
    prediction_path: str | Path | None = None,
) -> tuple[dict[str, float], Path]:
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

    dataset = MultiMoleculeDataset.from_jsonl(dataset_path)
    collator = MultiMoleculeCollator(
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
    metrics, predictions = generate_multi_molecule_predictions(
        model=model,
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
    return metrics, output_path
