from __future__ import annotations

from .defaults import DEFAULT_FINGERPRINT_NUM_BITS, DEFAULT_FINGERPRINT_RADIUS
from .fingerprints import build_morgan_fingerprint
from .parsing import ensure_rdkit
from .representations import MoleculeRepresentation

try:
    from rdkit import DataStructs
except ImportError:  # pragma: no cover - guarded by ensure_rdkit()
    DataStructs = None


def _safe_similarity(
    molecule_a,
    molecule_b,
    *,
    similarity_name: str,
    representation_a: MoleculeRepresentation = "auto",
    representation_b: MoleculeRepresentation = "auto",
    radius: int = DEFAULT_FINGERPRINT_RADIUS,
    n_bits: int = DEFAULT_FINGERPRINT_NUM_BITS,
) -> float:
    ensure_rdkit()
    fp_a = build_morgan_fingerprint(
        molecule_a,
        representation=representation_a,
        radius=radius,
        n_bits=n_bits,
    )
    fp_b = build_morgan_fingerprint(
        molecule_b,
        representation=representation_b,
        radius=radius,
        n_bits=n_bits,
    )
    if fp_a is None or fp_b is None:
        return 0.0

    if similarity_name == "dice":
        return float(DataStructs.DiceSimilarity(fp_a, fp_b))
    if similarity_name == "tanimoto":
        return float(DataStructs.TanimotoSimilarity(fp_a, fp_b))
    raise ValueError(f"Unsupported similarity: {similarity_name}")


def compute_dice_similarity(
    molecule_a,
    molecule_b,
    *,
    representation_a: MoleculeRepresentation = "auto",
    representation_b: MoleculeRepresentation = "auto",
    radius: int = DEFAULT_FINGERPRINT_RADIUS,
    n_bits: int = DEFAULT_FINGERPRINT_NUM_BITS,
) -> float:
    return _safe_similarity(
        molecule_a,
        molecule_b,
        similarity_name="dice",
        representation_a=representation_a,
        representation_b=representation_b,
        radius=radius,
        n_bits=n_bits,
    )


def compute_tanimoto_similarity(
    molecule_a,
    molecule_b,
    *,
    representation_a: MoleculeRepresentation = "auto",
    representation_b: MoleculeRepresentation = "auto",
    radius: int = DEFAULT_FINGERPRINT_RADIUS,
    n_bits: int = DEFAULT_FINGERPRINT_NUM_BITS,
) -> float:
    return _safe_similarity(
        molecule_a,
        molecule_b,
        similarity_name="tanimoto",
        representation_a=representation_a,
        representation_b=representation_b,
        radius=radius,
        n_bits=n_bits,
    )


__all__ = ["compute_dice_similarity", "compute_tanimoto_similarity"]
