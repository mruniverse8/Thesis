from __future__ import annotations

from collections import deque
from collections.abc import Mapping
from dataclasses import dataclass, field
import random
from typing import Iterable, Protocol, Sequence

from .trajectory import SampledStageTrajectory, ScoredStageTrajectory


@dataclass(frozen=True)
class OnPolicyBatch:
    trajectories: tuple[SampledStageTrajectory, ...]

    @classmethod
    def from_trajectories(cls, trajectories: Iterable[SampledStageTrajectory]) -> "OnPolicyBatch":
        return cls(tuple(trajectories))

    def __len__(self) -> int:
        return len(self.trajectories)

    def mean_stage_reward(self) -> float:
        if not self.trajectories:
            return 0.0
        return sum(item.terminal_reward for item in self.trajectories) / len(self.trajectories)

    def valid_fraction(self) -> float:
        if not self.trajectories:
            return 0.0
        return sum(int(item.is_valid) for item in self.trajectories) / len(self.trajectories)

    def duplicate_fraction(self) -> float:
        if not self.trajectories:
            return 0.0
        return sum(int(item.is_duplicate) for item in self.trajectories) / len(self.trajectories)

    def mean_num_actions(self) -> float:
        if not self.trajectories:
            return 0.0
        return sum(item.num_actions for item in self.trajectories) / len(self.trajectories)

    def mean_stage_index(self) -> float:
        if not self.trajectories:
            return 0.0
        return sum(item.stage_index for item in self.trajectories) / len(self.trajectories)


@dataclass(frozen=True)
class ReplaySampleBatch:
    trajectories: tuple[SampledStageTrajectory, ...]
    source_counts: Mapping[str, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "source_counts",
            {
                str(source): int(count)
                for source, count in dict(self.source_counts).items()
                if int(count) > 0
            },
        )

    def __len__(self) -> int:
        return len(self.trajectories)


class ReplayBuffer(Protocol):
    @property
    def total_action_tokens(self) -> int:
        ...

    def __len__(self) -> int:
        ...

    def add(self, trajectory: SampledStageTrajectory) -> None:
        ...

    def extend(self, trajectories: Sequence[SampledStageTrajectory]) -> None:
        ...

    def snapshot(self) -> tuple[SampledStageTrajectory, ...]:
        ...

    def sample(
        self,
        batch_size: int,
        *,
        rng: random.Random | None = None,
        with_replacement: bool = False,
    ) -> ReplaySampleBatch:
        ...

    def observe_scored(self, trajectories: Sequence[ScoredStageTrajectory]) -> None:
        ...


class _BaseReplayBuffer:
    def __init__(self, capacity: int, *, max_total_action_tokens: int) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be positive.")
        if max_total_action_tokens <= 0:
            raise ValueError("max_total_action_tokens must be positive.")
        self.capacity = int(capacity)
        self.max_total_action_tokens = int(max_total_action_tokens)
        self._items: deque[SampledStageTrajectory] = deque()
        self._total_action_tokens = 0

    def __len__(self) -> int:
        return len(self._items)

    @property
    def total_action_tokens(self) -> int:
        return self._total_action_tokens

    def _evict_oldest(self) -> None:
        removed = self._items.popleft()
        self._total_action_tokens -= removed.num_actions

    def add(self, trajectory: SampledStageTrajectory) -> None:
        self._items.append(trajectory)
        self._total_action_tokens += trajectory.num_actions
        while (
            len(self._items) > self.capacity
            or self._total_action_tokens > self.max_total_action_tokens
        ):
            self._evict_oldest()

    def extend(self, trajectories: Sequence[SampledStageTrajectory]) -> None:
        for trajectory in trajectories:
            self.add(trajectory)

    def snapshot(self) -> tuple[SampledStageTrajectory, ...]:
        return tuple(self._items)

    def observe_scored(self, trajectories: Sequence[ScoredStageTrajectory]) -> None:
        del trajectories


class UniformReplayBuffer(_BaseReplayBuffer):
    def sample(
        self,
        batch_size: int,
        *,
        rng: random.Random | None = None,
        with_replacement: bool = False,
    ) -> ReplaySampleBatch:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive.")
        if not self._items:
            return ReplaySampleBatch(())

        generator = rng or random
        population = list(self._items)
        if with_replacement:
            sampled = tuple(generator.choice(population) for _ in range(batch_size))
            return ReplaySampleBatch(sampled, source_counts={"uniform": len(sampled)})
        if batch_size >= len(population):
            return ReplaySampleBatch(tuple(population), source_counts={"uniform": len(population)})
        sampled = tuple(generator.sample(population, batch_size))
        return ReplaySampleBatch(sampled, source_counts={"uniform": len(sampled)})


def build_replay_buffer(
    *,
    capacity: int,
    max_total_action_tokens: int,
    buffer_type: str,
    recent_fraction: float = 0.40,
    reward_fraction: float = 0.40,
    uniform_fraction: float = 0.20,
    tb_residual_fraction: float = 0.20,
    reward_temperature: float = 1.0,
    tb_residual_temperature: float = 1.0,
    recent_window_size: int = 64,
    max_invalid_fraction: float = 0.20,
    max_duplicate_fraction: float = 0.10,
) -> ReplayBuffer:
    normalized_buffer_type = str(buffer_type).strip().lower()
    if normalized_buffer_type in {"uniform", "random"}:
        return UniformReplayBuffer(
            capacity,
            max_total_action_tokens=max_total_action_tokens,
        )
    if normalized_buffer_type == "experimental_mixture":
        from .experimental_buffers import ExperimentalMixtureReplayBuffer

        return ExperimentalMixtureReplayBuffer(
            capacity,
            recent_fraction=recent_fraction,
            reward_fraction=reward_fraction,
            uniform_fraction=uniform_fraction,
            reward_temperature=reward_temperature,
            recent_window_size=recent_window_size,
            max_invalid_fraction=max_invalid_fraction,
            max_duplicate_fraction=max_duplicate_fraction,
        )
    if normalized_buffer_type == "experimental_tb_mixture":
        from .experimental_buffers import ExperimentalTBMixtureReplayBuffer

        return ExperimentalTBMixtureReplayBuffer(
            capacity,
            recent_fraction=recent_fraction,
            reward_fraction=reward_fraction,
            uniform_fraction=uniform_fraction,
            tb_residual_fraction=tb_residual_fraction,
            reward_temperature=reward_temperature,
            tb_residual_temperature=tb_residual_temperature,
            recent_window_size=recent_window_size,
            max_invalid_fraction=max_invalid_fraction,
            max_duplicate_fraction=max_duplicate_fraction,
        )
    raise ValueError(f"Unsupported replay buffer_type: {buffer_type!r}")


TrajectoryReplayBuffer = UniformReplayBuffer


__all__ = [
    "OnPolicyBatch",
    "ReplayBuffer",
    "ReplaySampleBatch",
    "TrajectoryReplayBuffer",
    "UniformReplayBuffer",
    "build_replay_buffer",
]
