from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import selfies as sf
import torch
from torch.utils.data import Dataset

from .io_utils import read_jsonl
from .prompting import (
    build_text2mol_prompt,
    normalize_free_text,
    normalize_selfies_text,
    wrap_selfies_target,
)


def _string_or_empty(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    return str(value)


def build_processed_record(
    raw_record: dict[str, Any],
    split: str,
    index: int,
    convert_missing_selfies_from_smiles: bool = True,
) -> dict[str, str]:
    description = normalize_free_text(_string_or_empty(raw_record.get("description")))
    smiles = _string_or_empty(raw_record.get("SMILES")).strip()
    selfies_text = normalize_selfies_text(_string_or_empty(raw_record.get("SELFIES")))

    if not selfies_text and convert_missing_selfies_from_smiles and smiles:
        selfies_text = normalize_selfies_text(sf.encoder(smiles))

    if not description:
        raise ValueError("Missing description")
    if not selfies_text:
        raise ValueError("Missing SELFIES")

    # Fail fast during preprocessing instead of discovering invalid strings later.
    sf.decoder(selfies_text)

    cid = _string_or_empty(raw_record.get("CID")).strip()
    processed = {
        "id": cid or f"{split}-{index:06d}",
        "description": description,
        "selfies": selfies_text,
    }
    if smiles:
        processed["source_smiles"] = smiles
    return processed


class TextToSelfiesDataset(Dataset):
    def __init__(self, records: list[dict[str, Any]], source_path: Path | None = None):
        if not records:
            raise ValueError(f"No records available for dataset: {source_path}")
        self.records = records
        self.source_path = source_path

    @classmethod
    def from_jsonl(cls, path_value: str | Path) -> "TextToSelfiesDataset":
        path = Path(path_value)
        return cls(read_jsonl(path), source_path=path)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, str]:
        record = self.records[index]
        description = normalize_free_text(record["description"])
        selfies_text = normalize_selfies_text(record["selfies"])
        return {
            "id": str(record["id"]),
            "description": description,
            "selfies": selfies_text,
            "prompt": build_text2mol_prompt(description),
            "target_text": wrap_selfies_target(selfies_text),
        }


class TextToSelfiesCollator:
    def __init__(
        self,
        tokenizer: Any,
        max_source_length: int,
        max_target_length: int,
    ) -> None:
        self.tokenizer = tokenizer
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length

    def __call__(self, batch: list[dict[str, str]]) -> dict[str, Any]:
        prompts = [example["prompt"] for example in batch]
        targets = [example["target_text"] for example in batch]

        model_inputs = self.tokenizer(
            prompts,
            padding=True,
            truncation=True,
            max_length=self.max_source_length,
            return_tensors="pt",
        )
        target_tokens = self.tokenizer(
            text_target=targets,
            padding=True,
            truncation=True,
            max_length=self.max_target_length,
            return_tensors="pt",
        )

        labels = target_tokens["input_ids"].clone()
        labels[labels == self.tokenizer.pad_token_id] = -100
        model_inputs["labels"] = labels

        model_inputs["example_ids"] = [example["id"] for example in batch]
        model_inputs["prompts"] = prompts
        model_inputs["target_texts"] = targets
        model_inputs["reference_selfies"] = [example["selfies"] for example in batch]
        model_inputs["descriptions"] = [example["description"] for example in batch]
        return model_inputs


def move_tensor_batch_to_device(batch: dict[str, Any], device: torch.device) -> dict[str, torch.Tensor]:
    return {
        key: value.to(device)
        for key, value in batch.items()
        if torch.is_tensor(value)
    }
