from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


MoleculeRepresentation = Literal["auto", "smiles", "selfies", "rdkit"]


@dataclass(frozen=True)
class MoleculeRecord:
    input_text: str
    input_representation: str
    normalized_input: str
    smiles: str | None
    canonical_smiles: str | None
    is_valid: bool
    used_selfies_decoder: bool
    error: str | None = None
    mol: object | None = None


__all__ = ["MoleculeRecord", "MoleculeRepresentation"]
