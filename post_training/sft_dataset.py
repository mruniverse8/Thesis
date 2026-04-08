from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import selfies as sf
import torch
from torch.utils.data import Dataset

from src.io_utils import read_jsonl
from src.prompting import normalize_free_text, normalize_selfies_text

from .sequence_format import serialize_molecule_sequence
from .sft_prompting import build_diverse_text2mol_prompt


def _string_or_empty(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    return str(value)


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


def _iter_candidate_fields(raw_record: dict[str, Any]) -> list[tuple[str, str]]:
    candidates: list[tuple[str, str]] = []

    for key in ("target_selfies_list", "selfies_list"):
        for item in _as_list(raw_record.get(key)):
            candidates.append((_string_or_empty(item), ""))

    for key in ("target_smiles_list", "smiles_list"):
        for item in _as_list(raw_record.get(key)):
            candidates.append(("", _string_or_empty(item)))

    for key in ("targets", "molecules"):
        for item in _as_list(raw_record.get(key)):
            if isinstance(item, dict):
                candidates.append(
                    (
                        _string_or_empty(item.get("selfies") or item.get("SELFIES")),
                        _string_or_empty(item.get("smiles") or item.get("SMILES")),
                    )
                )
            else:
                candidates.append((_string_or_empty(item), ""))

    single_selfies = _string_or_empty(raw_record.get("selfies") or raw_record.get("SELFIES"))
    single_smiles = _string_or_empty(raw_record.get("smiles") or raw_record.get("SMILES"))
    if single_selfies or single_smiles:
        candidates.append((single_selfies, single_smiles))

    return candidates


def _canonical_smiles_for_selfies(selfies_text: str) -> str | None:
    from reward_utils.validation import parse_molecule_text

    try:
        record = parse_molecule_text(selfies_text, representation="selfies")
    except ImportError:
        try:
            return sf.decoder(selfies_text)
        except Exception:
            return None

    if not record.is_valid or record.canonical_smiles is None:
        return None
    return record.canonical_smiles


def build_multi_molecule_processed_record(
    raw_record: dict[str, Any],
    split: str,
    index: int,
    max_molecules_per_sequence: int = 8,
    convert_missing_selfies_from_smiles: bool = True,
) -> dict[str, Any]:
    description = normalize_free_text(_string_or_empty(raw_record.get("description")))
    if not description:
        raise ValueError("Missing description")

    seen_keys: set[str] = set()
    target_selfies_list: list[str] = []
    target_smiles_list: list[str] = []

    for selfies_text, smiles_text in _iter_candidate_fields(raw_record):
        normalized_selfies = normalize_selfies_text(selfies_text)
        normalized_smiles = _string_or_empty(smiles_text).strip()

        if not normalized_selfies and convert_missing_selfies_from_smiles and normalized_smiles:
            try:
                normalized_selfies = normalize_selfies_text(sf.encoder(normalized_smiles))
            except Exception:
                normalized_selfies = ""

        if not normalized_selfies:
            continue

        try:
            sf.decoder(normalized_selfies)
        except Exception:
            continue

        canonical_smiles = _canonical_smiles_for_selfies(normalized_selfies)
        dedupe_key = canonical_smiles or normalized_selfies
        if dedupe_key in seen_keys:
            continue

        seen_keys.add(dedupe_key)
        target_selfies_list.append(normalized_selfies)
        if canonical_smiles:
            target_smiles_list.append(canonical_smiles)

        if len(target_selfies_list) >= max_molecules_per_sequence:
            break

    if not target_selfies_list:
        raise ValueError("Missing valid target molecules")

    cid = _string_or_empty(raw_record.get("id") or raw_record.get("CID")).strip()
    processed = {
        "id": cid or f"{split}-{index:06d}",
        "description": description,
        "target_selfies_list": target_selfies_list,
    }
    if target_smiles_list:
        processed["target_smiles_list"] = target_smiles_list
    return processed


class MultiMoleculeDataset(Dataset):
    def __init__(self, records: list[dict[str, Any]], source_path: Path | None = None):
        if not records:
            raise ValueError(f"No records available for dataset: {source_path}")
        self.records = records
        self.source_path = source_path

    @classmethod
    def from_jsonl(cls, path_value: str | Path) -> "MultiMoleculeDataset":
        path = Path(path_value)
        return cls(read_jsonl(path), source_path=path)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        description = normalize_free_text(record["description"])
        target_selfies_list = [
            normalize_selfies_text(item) for item in record["target_selfies_list"] if item
        ]
        return {
            "id": str(record["id"]),
            "description": description,
            "target_selfies_list": target_selfies_list,
            "prompt": build_diverse_text2mol_prompt(description),
            "target_text": serialize_molecule_sequence(target_selfies_list),
        }


class MultiMoleculeCollator:
    def __init__(
        self,
        tokenizer: Any,
        max_source_length: int,
        max_target_length: int,
    ) -> None:
        self.tokenizer = tokenizer
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length

    def __call__(self, batch: list[dict[str, Any]]) -> dict[str, Any]:
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
        model_inputs["descriptions"] = [example["description"] for example in batch]
        model_inputs["target_selfies_lists"] = [
            list(example["target_selfies_list"]) for example in batch
        ]
        return model_inputs
