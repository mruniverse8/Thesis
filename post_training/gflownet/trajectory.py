from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Hashable, Sequence

import torch

ScalarLike = float | torch.Tensor
TokenLike = Hashable
PrefixState = tuple[TokenLike, ...]


def build_prefix_states(actions: Sequence[TokenLike]) -> tuple[PrefixState, ...]:
    prefixes: list[PrefixState] = [()]
    running_prefix: list[TokenLike] = []
    for action in actions:
        running_prefix.append(action)
        prefixes.append(tuple(running_prefix))
    return tuple(prefixes)


@dataclass(frozen=True)
class SampledStageTrajectory:
    rollout_id: str
    example_id: str
    prompt_text: str
    description: str
    target_selfies_list: tuple[str, ...]
    stage_index: int
    decoder_prefix_text: str
    previous_valid_selfies: tuple[str, ...]
    stage_text: str
    sampled_selfies: str | None
    action_token_ids: tuple[int, ...]
    reward_breakdown: dict[str, Any]
    prefix_rewards: tuple[float, ...]
    terminal_reward: float
    stop_token: str | None
    termination_reason: str
    is_valid: bool
    is_duplicate: bool
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.stage_index <= 0:
            raise ValueError("stage_index must be positive.")
        if len(self.prefix_rewards) != len(self.action_token_ids) + 1:
            raise ValueError("prefix_rewards must contain one value per token prefix.")
        if self.terminal_reward <= 0.0:
            raise ValueError("terminal_reward must be strictly positive.")
        if any(reward <= 0.0 for reward in self.prefix_rewards):
            raise ValueError("prefix_rewards must be strictly positive.")

    @property
    def prompt_id(self) -> str:
        return self.example_id

    @property
    def conditioning_text(self) -> str:
        return self.prompt_text

    @property
    def prefix_states(self) -> tuple[PrefixState, ...]:
        return build_prefix_states(self.action_token_ids)

    @property
    def num_actions(self) -> int:
        return len(self.action_token_ids)

    @property
    def terminal_prefix(self) -> PrefixState:
        return self.prefix_states[-1]

    def to_dict(self) -> dict[str, Any]:
        return {
            "rollout_id": self.rollout_id,
            "example_id": self.example_id,
            "prompt_text": self.prompt_text,
            "description": self.description,
            "target_selfies_list": list(self.target_selfies_list),
            "stage_index": self.stage_index,
            "decoder_prefix_text": self.decoder_prefix_text,
            "previous_valid_selfies": list(self.previous_valid_selfies),
            "stage_text": self.stage_text,
            "sampled_selfies": self.sampled_selfies,
            "action_token_ids": list(self.action_token_ids),
            "reward_breakdown": dict(self.reward_breakdown),
            "prefix_rewards": list(self.prefix_rewards),
            "terminal_reward": self.terminal_reward,
            "stop_token": self.stop_token,
            "termination_reason": self.termination_reason,
            "is_valid": self.is_valid,
            "is_duplicate": self.is_duplicate,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class ScoredStageTrajectory:
    sampled: SampledStageTrajectory
    log_pf_tokens: tuple[ScalarLike, ...]
    log_stop: tuple[ScalarLike, ...]
    log_state_flows: tuple[ScalarLike, ...]
    log_pb_tokens: tuple[ScalarLike, ...] = ()

    def __post_init__(self) -> None:
        if len(self.log_pf_tokens) != self.sampled.num_actions:
            raise ValueError("log_pf_tokens must align with sampled action_token_ids.")
        if len(self.log_stop) != self.sampled.num_actions + 1:
            raise ValueError("log_stop must contain one value per token prefix.")
        if len(self.log_state_flows) != self.sampled.num_actions + 1:
            raise ValueError("log_state_flows must contain one value per token prefix.")
        if self.log_pb_tokens and len(self.log_pb_tokens) != self.sampled.num_actions:
            raise ValueError("log_pb_tokens must align with sampled action_token_ids.")

    @property
    def prompt_id(self) -> str:
        return self.sampled.prompt_id

    @property
    def conditioning_text(self) -> str:
        return self.sampled.conditioning_text

    @property
    def prefix_states(self) -> tuple[PrefixState, ...]:
        return self.sampled.prefix_states

    @property
    def action_ids(self) -> tuple[int, ...]:
        return self.sampled.action_token_ids

    @property
    def prefix_rewards(self) -> tuple[float, ...]:
        return self.sampled.prefix_rewards

    @property
    def terminal_reward(self) -> float:
        return self.sampled.terminal_reward

    @property
    def is_valid(self) -> bool:
        return self.sampled.is_valid

    @property
    def is_duplicate(self) -> bool:
        return self.sampled.is_duplicate

    @property
    def termination_reason(self) -> str:
        return self.sampled.termination_reason

    def effective_log_pb_tokens(self) -> tuple[ScalarLike, ...]:
        if self.log_pb_tokens:
            return self.log_pb_tokens
        return tuple(0.0 for _ in self.sampled.action_token_ids)
