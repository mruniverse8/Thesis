from __future__ import annotations

from .defaults import (
    DEFAULT_FINGERPRINT_NUM_BITS,
    DEFAULT_FINGERPRINT_RADIUS,
    DEFAULT_INVALID_SIMILARITY_NGRAM_SIZE,
    DEFAULT_PENALTY_INVALID,
)
from .fingerprints import build_morgan_fingerprint, ensure_molecule_record
from .parsing import ensure_rdkit
from .representations import MoleculeRepresentation
from .selfies import SELFIES_TOKEN_PATTERN, looks_like_selfies, normalize_selfies_text

try:
    from rdkit import DataStructs
except ImportError:  # pragma: no cover - guarded by ensure_rdkit()
    DataStructs = None


def _selfies_token_features(text: str) -> set[str] | None:
    compact = normalize_selfies_text(text)
    if not looks_like_selfies(compact):
        return None
    tokens = {token.rstrip(".") for token in SELFIES_TOKEN_PATTERN.findall(compact)}
    return tokens or None


def _character_ngram_features(text: str, *, ngram_size: int) -> set[str] | None:
    compact = "".join(str(text).split())
    if len(compact) < ngram_size:
        return None
    return {
        compact[index : index + ngram_size]
        for index in range(len(compact) - ngram_size + 1)
    }


def _token_features(text: str, *, ngram_size: int) -> set[str] | None:
    selfies_features = _selfies_token_features(text)
    if selfies_features is not None:
        return selfies_features
    return _character_ngram_features(text, ngram_size=ngram_size)


def token_ngram_similarity(
    text_a: str,
    text_b: str,
    *,
    similarity_name: str,
    ngram_size: int = DEFAULT_INVALID_SIMILARITY_NGRAM_SIZE,
) -> float | None:
    if int(ngram_size) < 1:
        raise ValueError("ngram_size must be at least 1.")

    features_a = _token_features(text_a, ngram_size=int(ngram_size))
    features_b = _token_features(text_b, ngram_size=int(ngram_size))
    if features_a is None or features_b is None:
        return None

    intersection_size = len(features_a & features_b)
    if similarity_name == "dice":
        denominator = len(features_a) + len(features_b)
        return 0.0 if denominator == 0 else (2.0 * intersection_size) / denominator
    if similarity_name == "tanimoto":
        union_size = len(features_a | features_b)
        return 0.0 if union_size == 0 else intersection_size / union_size
    raise ValueError(f"Unsupported similarity: {similarity_name}")


def _safe_similarity(
    molecule_a,
    molecule_b,
    *,
    similarity_name: str,
    representation_a: MoleculeRepresentation = "auto",
    representation_b: MoleculeRepresentation = "auto",
    radius: int = DEFAULT_FINGERPRINT_RADIUS,
    n_bits: int = DEFAULT_FINGERPRINT_NUM_BITS,
    invalid_fallback_penalty: float = DEFAULT_PENALTY_INVALID,
    invalid_similarity_ngram_size: int = DEFAULT_INVALID_SIMILARITY_NGRAM_SIZE,
) -> float:
    ensure_rdkit()
    record_a = ensure_molecule_record(molecule_a, representation=representation_a)
    record_b = ensure_molecule_record(molecule_b, representation=representation_b)
    fp_a = build_morgan_fingerprint(
        record_a,
        representation=representation_a,
        radius=radius,
        n_bits=n_bits,
    )
    fp_b = build_morgan_fingerprint(
        record_b,
        representation=representation_b,
        radius=radius,
        n_bits=n_bits,
    )
    if fp_a is None or fp_b is None:
        fallback_similarity = token_ngram_similarity(
            record_a.input_text,
            record_b.input_text,
            similarity_name=similarity_name,
            ngram_size=invalid_similarity_ngram_size,
        )
        if fallback_similarity is None:
            return 0.0
        return float(invalid_fallback_penalty) * fallback_similarity

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
    invalid_fallback_penalty: float = DEFAULT_PENALTY_INVALID,
    invalid_similarity_ngram_size: int = DEFAULT_INVALID_SIMILARITY_NGRAM_SIZE,
) -> float:
    return _safe_similarity(
        molecule_a,
        molecule_b,
        similarity_name="dice",
        representation_a=representation_a,
        representation_b=representation_b,
        radius=radius,
        n_bits=n_bits,
        invalid_fallback_penalty=invalid_fallback_penalty,
        invalid_similarity_ngram_size=invalid_similarity_ngram_size,
    )


def compute_tanimoto_similarity(
    molecule_a,
    molecule_b,
    *,
    representation_a: MoleculeRepresentation = "auto",
    representation_b: MoleculeRepresentation = "auto",
    radius: int = DEFAULT_FINGERPRINT_RADIUS,
    n_bits: int = DEFAULT_FINGERPRINT_NUM_BITS,
    invalid_fallback_penalty: float = DEFAULT_PENALTY_INVALID,
    invalid_similarity_ngram_size: int = DEFAULT_INVALID_SIMILARITY_NGRAM_SIZE,
) -> float:
    return _safe_similarity(
        molecule_a,
        molecule_b,
        similarity_name="tanimoto",
        representation_a=representation_a,
        representation_b=representation_b,
        radius=radius,
        n_bits=n_bits,
        invalid_fallback_penalty=invalid_fallback_penalty,
        invalid_similarity_ngram_size=invalid_similarity_ngram_size,
    )


__all__ = [
    "compute_dice_similarity",
    "compute_tanimoto_similarity",
    "token_ngram_similarity",
]
