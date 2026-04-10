from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from molecules.selfies import normalize_selfies_text
from src.prompting import normalize_free_text


@dataclass(frozen=True)
class GroupedMoleculeRecord:
    id: str
    description: str
    target_selfies_list: tuple[str, ...]
    target_smiles_list: tuple[str, ...] = ()

    @classmethod
    def from_mapping(cls, payload: dict[str, Any]) -> "GroupedMoleculeRecord":
        description = normalize_free_text(str(payload.get("description", "")))
        if not description:
            raise ValueError("Missing description")

        target_selfies = tuple(
            normalize_selfies_text(item)
            for item in payload.get("target_selfies_list", [])
            if normalize_selfies_text(str(item))
        )
        if not target_selfies:
            raise ValueError("Missing valid target molecules")

        target_smiles = tuple(str(item).strip() for item in payload.get("target_smiles_list", []) if str(item).strip())
        identifier = str(payload.get("id") or "").strip() or "unknown"
        return cls(
            id=identifier,
            description=description,
            target_selfies_list=target_selfies,
            target_smiles_list=target_smiles,
        )

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "id": self.id,
            "description": self.description,
            "target_selfies_list": list(self.target_selfies_list),
        }
        if self.target_smiles_list:
            payload["target_smiles_list"] = list(self.target_smiles_list)
        return payload


def coerce_grouped_molecule_record(payload: dict[str, Any] | GroupedMoleculeRecord) -> GroupedMoleculeRecord:
    if isinstance(payload, GroupedMoleculeRecord):
        return payload
    return GroupedMoleculeRecord.from_mapping(payload)
