from __future__ import annotations

from dataclasses import dataclass


DEFAULT_FINGERPRINT_RADIUS = 2
DEFAULT_FINGERPRINT_NUM_BITS = 2048

DEFAULT_MATCH_ALPHA = 0.5
DEFAULT_DIVERSITY_BETA = 1.0
CHEBI20_DIVERSITY_BETA = 2.0

DEFAULT_PLUS_VALID = 0.8
DEFAULT_REWARD_VARIANT = "reward_var2"

DEFAULT_MATCH_WEIGHT = 1.0
DEFAULT_DIVERSITY_WEIGHT = 1.0
DEFAULT_REWARD_AMPLIFICATION = 8.0
DEFAULT_DUPLICATE_PENALTY_FACTOR = 1.0
DEFAULT_PENALTY_INVALID = 0.5
DEFAULT_INVALID_SIMILARITY_NGRAM_SIZE = 3

DEFAULT_ACCEPTANCE_DICE_THRESHOLD = 0.7
DEFAULT_ACCEPTANCE_TANIMOTO_THRESHOLD = 0.6


@dataclass(frozen=True)
class RewardConfig:
    fingerprint_radius: int = DEFAULT_FINGERPRINT_RADIUS
    fingerprint_num_bits: int = DEFAULT_FINGERPRINT_NUM_BITS
    match_alpha: float = DEFAULT_MATCH_ALPHA
    plus_valid: float = DEFAULT_PLUS_VALID
    reward_variant: str = DEFAULT_REWARD_VARIANT
    diversity_beta: float = DEFAULT_DIVERSITY_BETA
    match_weight: float = DEFAULT_MATCH_WEIGHT
    diversity_weight: float = DEFAULT_DIVERSITY_WEIGHT
    reward_amplification: float = DEFAULT_REWARD_AMPLIFICATION
    duplicate_penalty_factor: float = DEFAULT_DUPLICATE_PENALTY_FACTOR
    penalty_invalid: float = DEFAULT_PENALTY_INVALID
    invalid_similarity_ngram_size: int = DEFAULT_INVALID_SIMILARITY_NGRAM_SIZE
    acceptance_dice_threshold: float = DEFAULT_ACCEPTANCE_DICE_THRESHOLD
    acceptance_tanimoto_threshold: float = DEFAULT_ACCEPTANCE_TANIMOTO_THRESHOLD

    def __post_init__(self) -> None:
        if self.reward_variant not in {"reward_var1", "reward_var2", "reward_var3"}:
            raise ValueError(
                "reward_variant must be one of {'reward_var1', 'reward_var2', 'reward_var3'}."
            )
        if not 0.0 <= float(self.duplicate_penalty_factor) <= 1.0:
            raise ValueError("duplicate_penalty_factor must be within [0.0, 1.0].")
        if not 0.0 < float(self.penalty_invalid) <= 1.0:
            raise ValueError("penalty_invalid must be within (0.0, 1.0].")
        if int(self.invalid_similarity_ngram_size) < 1:
            raise ValueError("invalid_similarity_ngram_size must be at least 1.")


DEFAULT_REWARD_CONFIG = RewardConfig()
CHEBI20_REWARD_CONFIG = RewardConfig(diversity_beta=CHEBI20_DIVERSITY_BETA)

PPO_DEFAULTS = {
    "ppo_iterations": 200,
    "mini_batch_size": 8,
    "batch_size": 128,
    "learning_rate": 5.0e-5,
    "kl_penalty": 0.01,
    "max_sequence_length": 2560,
    "reward_amplification": DEFAULT_REWARD_AMPLIFICATION,
    "lora_rank": 16,
    "lora_alpha": 32,
}


__all__ = [
    "CHEBI20_DIVERSITY_BETA",
    "CHEBI20_REWARD_CONFIG",
    "DEFAULT_ACCEPTANCE_DICE_THRESHOLD",
    "DEFAULT_ACCEPTANCE_TANIMOTO_THRESHOLD",
    "DEFAULT_DIVERSITY_BETA",
    "DEFAULT_DIVERSITY_WEIGHT",
    "DEFAULT_DUPLICATE_PENALTY_FACTOR",
    "DEFAULT_FINGERPRINT_NUM_BITS",
    "DEFAULT_FINGERPRINT_RADIUS",
    "DEFAULT_INVALID_SIMILARITY_NGRAM_SIZE",
    "DEFAULT_MATCH_ALPHA",
    "DEFAULT_MATCH_WEIGHT",
    "DEFAULT_PENALTY_INVALID",
    "DEFAULT_PLUS_VALID",
    "DEFAULT_REWARD_AMPLIFICATION",
    "DEFAULT_REWARD_CONFIG",
    "DEFAULT_REWARD_VARIANT",
    "PPO_DEFAULTS",
    "RewardConfig",
]
