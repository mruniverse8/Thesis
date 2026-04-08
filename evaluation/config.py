from __future__ import annotations

from dataclasses import dataclass

from reward_utils.defaults import (
    DEFAULT_ACCEPTANCE_DICE_THRESHOLD,
    DEFAULT_ACCEPTANCE_TANIMOTO_THRESHOLD,
    DEFAULT_FINGERPRINT_NUM_BITS,
    DEFAULT_FINGERPRINT_RADIUS,
)


@dataclass(frozen=True)
class MoleculeMetricConfig:
    fingerprint_radius: int = DEFAULT_FINGERPRINT_RADIUS
    fingerprint_num_bits: int = DEFAULT_FINGERPRINT_NUM_BITS
    acceptance_dice_threshold: float = DEFAULT_ACCEPTANCE_DICE_THRESHOLD
    ncircles_tanimoto_threshold: float = DEFAULT_ACCEPTANCE_TANIMOTO_THRESHOLD
