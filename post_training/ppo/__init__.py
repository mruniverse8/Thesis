"""Molecule-stage PPO package."""

from .config import PPOConfig, RolloutGenerationConfig, StageTrajectory, build_ppo_config
from .diagnostics import (
    build_trajectory_preview_payload,
    rollout_stage_metrics,
    termination_reason_metrics,
)
from .model import PolicyValueModel, gather_last_token_hidden_state, load_reference_model
from .rewarding import (
    build_reward_config,
    score_generated_sequence,
    score_stage_candidate,
    score_stage_reward,
    summarize_reward_breakdowns,
)

__all__ = [
    "PPOConfig",
    "PolicyValueModel",
    "RolloutGenerationConfig",
    "StageTrajectory",
    "build_trajectory_preview_payload",
    "build_ppo_config",
    "build_reward_config",
    "gather_last_token_hidden_state",
    "load_reference_model",
    "rollout_stage_metrics",
    "score_generated_sequence",
    "score_stage_candidate",
    "score_stage_reward",
    "summarize_reward_breakdowns",
    "termination_reason_metrics",
]
