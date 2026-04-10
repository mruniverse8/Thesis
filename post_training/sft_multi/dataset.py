from __future__ import annotations

from pathlib import Path
from typing import Any

from torch.utils.data import Dataset

from molecules.datasets.multi import build_multi_molecule_processed_record
from src.io_utils import read_jsonl

from post_training.shared.dataset_types import GroupedMoleculeRecord, coerce_grouped_molecule_record
from post_training.shared.sequence import serialize_staged_target

from .prompting import build_diverse_text2mol_prompt


def load_grouped_records(path_value: str | Path) -> list[GroupedMoleculeRecord]:
    return [coerce_grouped_molecule_record(record) for record in read_jsonl(path_value)]


class MultiMoleculeDataset(Dataset):
    def __init__(
        self,
        records: list[dict[str, Any] | GroupedMoleculeRecord],
        source_path: Path | None = None,
    ) -> None:
        if not records:
            raise ValueError(f"No records available for dataset: {source_path}")
        self.records = [coerce_grouped_molecule_record(record) for record in records]
        self.source_path = source_path

    @classmethod
    def from_jsonl(cls, path_value: str | Path) -> "MultiMoleculeDataset":
        path = Path(path_value)
        return cls(load_grouped_records(path), source_path=path)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        target_selfies_list = list(record.target_selfies_list)
        return {
            "id": record.id,
            "description": record.description,
            "target_selfies_list": target_selfies_list,
            "target_smiles_list": list(record.target_smiles_list),
            "prompt": build_diverse_text2mol_prompt(record.description),
            "target_text": serialize_staged_target(target_selfies_list),
        }
