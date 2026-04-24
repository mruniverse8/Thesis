from __future__ import annotations

from molecules.defaults import (
    CHEBI20_REWARD_CONFIG,
    DEFAULT_PENALTY_INVALID,
    DEFAULT_REWARD_CONFIG,
    PPO_DEFAULTS,
    RewardConfig,
)
from molecules.parsing import MoleculeRecord, is_duplicate_candidate, parse_molecule_text
from molecules.rewards import (
    RewardBreakdown,
    RewardComponent,
    compute_rdiv,
    compute_rmatch,
    compute_total_reward,
    score_candidate_sequence,
)
from molecules.similarity import compute_dice_similarity, compute_tanimoto_similarity

__all__ = [
    "CHEBI20_REWARD_CONFIG",
    "DEFAULT_PENALTY_INVALID",
    "DEFAULT_REWARD_CONFIG",
    "PPO_DEFAULTS",
    "MoleculeRecord",
    "RewardBreakdown",
    "RewardComponent",
    "RewardConfig",
    "compute_dice_similarity",
    "compute_tanimoto_similarity",
    "compute_rdiv",
    "compute_rmatch",
    "compute_total_reward",
    "is_duplicate_candidate",
    "parse_molecule_text",
    "score_candidate_sequence",
]
