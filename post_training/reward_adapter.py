from __future__ import annotations

from typing import Sequence

from reward_utils.defaults import RewardConfig
from reward_utils.rewards import RewardBreakdown, compute_total_reward, score_candidate_sequence


def score_stage_candidate(
    candidate_selfies: str,
    targets: Sequence[str],
    previous_candidates: Sequence[str],
    *,
    config: RewardConfig | None = None,
) -> RewardBreakdown:
    return compute_total_reward(
        candidate_selfies,
        targets=targets,
        previous_candidates=previous_candidates,
        config=config,
        candidate_representation="selfies",
        target_representation="selfies",
        previous_representation="selfies",
    )


def score_generated_sequence(
    candidates: Sequence[str],
    targets: Sequence[str],
    *,
    config: RewardConfig | None = None,
) -> list[RewardBreakdown]:
    return score_candidate_sequence(
        candidates,
        targets,
        config=config,
        candidate_representation="selfies",
        target_representation="selfies",
    )


def summarize_reward_breakdowns(breakdowns: Sequence[RewardBreakdown]) -> dict[str, float]:
    if not breakdowns:
        return {
            "mean_amplified_reward": 0.0,
            "mean_total_reward": 0.0,
            "mean_match_reward": 0.0,
            "mean_diversity_reward": 0.0,
            "duplicate_rate": 0.0,
            "valid_rate": 0.0,
        }

    count = len(breakdowns)
    return {
        "mean_amplified_reward": sum(item.amplified_reward for item in breakdowns) / count,
        "mean_total_reward": sum(item.total_reward for item in breakdowns) / count,
        "mean_match_reward": sum(item.match.reward for item in breakdowns) / count,
        "mean_diversity_reward": sum(item.diversity.reward for item in breakdowns) / count,
        "duplicate_rate": sum(int(item.is_duplicate) for item in breakdowns) / count,
        "valid_rate": sum(int(item.candidate.is_valid) for item in breakdowns) / count,
    }
