from __future__ import annotations

from typing import Sequence

from .defaults import DEFAULT_FINGERPRINT_NUM_BITS, DEFAULT_FINGERPRINT_RADIUS
from .validation import MoleculeRecord, MoleculeRepresentation, ensure_rdkit, parse_molecule_text, record_from_rdkit_mol

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem
    from rdkit.Chem import rdFingerprintGenerator
except ImportError:  # pragma: no cover - guarded by ensure_rdkit()
    Chem = None
    AllChem = None
    rdFingerprintGenerator = None


def ensure_molecule_record(
    molecule: MoleculeRecord | object | str,
    *,
    representation: MoleculeRepresentation = "auto",
) -> MoleculeRecord:
    ensure_rdkit()
    if isinstance(molecule, MoleculeRecord):
        return molecule
    if isinstance(molecule, str):
        return parse_molecule_text(molecule, representation=representation)
    if Chem is not None and isinstance(molecule, Chem.Mol):
        return record_from_rdkit_mol(molecule)
    raise TypeError(f"Unsupported molecule input type: {type(molecule)!r}")


def build_morgan_fingerprint(
    molecule: MoleculeRecord | object | str,
    *,
    representation: MoleculeRepresentation = "auto",
    radius: int = DEFAULT_FINGERPRINT_RADIUS,
    n_bits: int = DEFAULT_FINGERPRINT_NUM_BITS,
):
    ensure_rdkit()
    record = ensure_molecule_record(molecule, representation=representation)
    if not record.is_valid or record.mol is None:
        return None
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)
    return generator.GetFingerprint(record.mol)


def build_fingerprints(
    molecules: Sequence[MoleculeRecord | object | str],
    *,
    representation: MoleculeRepresentation = "auto",
    radius: int = DEFAULT_FINGERPRINT_RADIUS,
    n_bits: int = DEFAULT_FINGERPRINT_NUM_BITS,
) -> list[object | None]:
    return [
        build_morgan_fingerprint(
            molecule,
            representation=representation,
            radius=radius,
            n_bits=n_bits,
        )
        for molecule in molecules
    ]
