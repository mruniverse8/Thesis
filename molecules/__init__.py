from __future__ import annotations

from .defaults import (
    CHEBI20_REWARD_CONFIG,
    DEFAULT_INVALID_SIMILARITY_NGRAM_SIZE,
    DEFAULT_REWARD_CONFIG,
    PPO_DEFAULTS,
    RewardConfig,
)

__all__ = [
    "CHEBI20_REWARD_CONFIG",
    "DEFAULT_INVALID_SIMILARITY_NGRAM_SIZE",
    "DEFAULT_REWARD_CONFIG",
    "PPO_DEFAULTS",
    "MoleculeRecord",
    "MoleculeRepresentation",
    "RewardConfig",
    "compute_dice_similarity",
    "compute_tanimoto_similarity",
    "ensure_rdkit",
    "is_duplicate_candidate",
    "looks_like_selfies",
    "parse_molecule_text",
]


def __getattr__(name: str):
    if name in {
        "MoleculeRecord",
        "MoleculeRepresentation",
        "ensure_rdkit",
        "is_duplicate_candidate",
        "looks_like_selfies",
        "parse_molecule_text",
    }:
        from . import parsing

        return getattr(parsing, name)

    if name in {
        "compute_dice_similarity",
        "compute_tanimoto_similarity",
    }:
        from . import similarity

        return getattr(similarity, name)

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
