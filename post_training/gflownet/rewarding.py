from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from molecules.representations import MoleculeRecord
from reward_utils.defaults import RewardConfig
from reward_utils.rewards import RewardBreakdown

from post_training.ppo.rewarding import build_reward_config, score_stage_reward


@dataclass(frozen=True)
class StageRewardSummary:
    reward_breakdown: dict[str, object]
    prefix_rewards: tuple[float, ...]
    terminal_reward: float
    is_valid_terminal: bool
    is_duplicate_terminal: bool


def _reward_breakdown_to_dict(breakdown: RewardBreakdown) -> dict[str, object]:
    return {
        "candidate": breakdown.candidate.canonical_smiles,
        "is_valid": breakdown.candidate.is_valid,
        "is_duplicate": breakdown.is_duplicate,
        "match_reward": breakdown.match.reward,
        "diversity_reward": breakdown.diversity.reward,
        "total_reward": breakdown.total_reward,
        "amplified_reward": breakdown.amplified_reward,
    }


def _invalid_candidate_record(text: str) -> MoleculeRecord:
    normalized = str(text).strip()
    return MoleculeRecord(
        input_text=str(text),
        input_representation="selfies",
        normalized_input=normalized,
        smiles=None,
        canonical_smiles=None,
        is_valid=False,
        used_selfies_decoder=True,
        error="Invalid generated stage used only for n-gram reward fallback.",
        mol=None,
    )


def score_stage_terminal_reward(
    candidate_selfies: str | None,
    *,
    targets: Sequence[str],
    previous_candidates: Sequence[str],
    num_prefix_states: int,
    reward_config: RewardConfig | None = None,
    invalid_terminal_reward: float = 1.0e-4,
    invalid_candidate_text: str | None = None,
) -> StageRewardSummary:
    if num_prefix_states <= 0:
        raise ValueError("num_prefix_states must be positive.")

    effective_reward_config = reward_config or RewardConfig()
    penalty_invalid = float(effective_reward_config.penalty_invalid)
    reward_floor = max(float(invalid_terminal_reward), 1.0e-12)
    candidate_for_reward: str | MoleculeRecord
    if candidate_selfies is not None:
        candidate_for_reward = candidate_selfies.strip()
    elif invalid_candidate_text is not None and str(invalid_candidate_text).strip():
        candidate_for_reward = _invalid_candidate_record(str(invalid_candidate_text))
    else:
        candidate_for_reward = ""
    breakdown = score_stage_reward(
        candidate_for_reward,
        targets=targets,
        previous_candidates=previous_candidates,
        config=effective_reward_config,
    )
    is_valid_terminal = bool(breakdown.candidate.is_valid)
    terminal_reward = max(float(breakdown.amplified_reward), reward_floor)
    if not is_valid_terminal:
        terminal_reward *= penalty_invalid

    prefix_rewards = tuple([reward_floor] * (num_prefix_states - 1) + [terminal_reward])
    return StageRewardSummary(
        reward_breakdown=_reward_breakdown_to_dict(breakdown),
        prefix_rewards=prefix_rewards,
        terminal_reward=terminal_reward,
        is_valid_terminal=is_valid_terminal,
        is_duplicate_terminal=bool(breakdown.is_duplicate),
    )


__all__ = [
    "StageRewardSummary",
    "build_reward_config",
    "score_stage_terminal_reward",
]
