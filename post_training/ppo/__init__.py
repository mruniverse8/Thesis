"""Molecule-stage PPO package."""

from .config import PPOConfig, RolloutGenerationConfig, StageTrajectory, build_ppo_config
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
    "build_ppo_config",
    "build_reward_config",
    "gather_last_token_hidden_state",
    "load_reference_model",
    "score_generated_sequence",
    "score_stage_candidate",
    "score_stage_reward",
    "summarize_reward_breakdowns",
]
