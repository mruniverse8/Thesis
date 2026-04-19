from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import random
from typing import Iterable, Sequence

from .trajectory import SampledStageTrajectory


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


class TrajectoryReplayBuffer:
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

    def sample(
        self,
        batch_size: int,
        *,
        rng: random.Random | None = None,
        with_replacement: bool = False,
    ) -> list[SampledStageTrajectory]:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive.")
        if not self._items:
            return []
        generator = rng or random
        population = list(self._items)
        if with_replacement:
            return [generator.choice(population) for _ in range(batch_size)]
        if batch_size >= len(population):
            return population
        return generator.sample(population, batch_size)
