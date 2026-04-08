from __future__ import annotations

from dataclasses import dataclass


DEFAULT_FINGERPRINT_RADIUS = 2
DEFAULT_FINGERPRINT_NUM_BITS = 2048

DEFAULT_MATCH_ALPHA = 0.5
DEFAULT_DIVERSITY_BETA = 1.0
CHEBI20_DIVERSITY_BETA = 2.0

DEFAULT_MATCH_WEIGHT = 1.0
DEFAULT_DIVERSITY_WEIGHT = 1.0
DEFAULT_REWARD_AMPLIFICATION = 8.0

DEFAULT_ACCEPTANCE_DICE_THRESHOLD = 0.7
DEFAULT_ACCEPTANCE_TANIMOTO_THRESHOLD = 0.6


@dataclass(frozen=True)
class RewardConfig:
    fingerprint_radius: int = DEFAULT_FINGERPRINT_RADIUS
    fingerprint_num_bits: int = DEFAULT_FINGERPRINT_NUM_BITS
    match_alpha: float = DEFAULT_MATCH_ALPHA
    diversity_beta: float = DEFAULT_DIVERSITY_BETA
    match_weight: float = DEFAULT_MATCH_WEIGHT
    diversity_weight: float = DEFAULT_DIVERSITY_WEIGHT
    reward_amplification: float = DEFAULT_REWARD_AMPLIFICATION
    acceptance_dice_threshold: float = DEFAULT_ACCEPTANCE_DICE_THRESHOLD
    acceptance_tanimoto_threshold: float = DEFAULT_ACCEPTANCE_TANIMOTO_THRESHOLD


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
