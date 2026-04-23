from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import math
import random
from typing import Iterable, Protocol, Sequence

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


@dataclass(frozen=True)
class ReplaySampleBatch:
    trajectories: tuple[SampledStageTrajectory, ...]
    top_reward_count: int = 0
    hard_positive_count: int = 0
    hard_negative_count: int = 0

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
            return ReplaySampleBatch(sampled)
        if batch_size >= len(population):
            return ReplaySampleBatch(tuple(population))
        return ReplaySampleBatch(tuple(generator.sample(population, batch_size)))


class PriorityReplayBuffer(_BaseReplayBuffer):
    def __init__(
        self,
        capacity: int,
        *,
        max_total_action_tokens: int,
        invalid_terminal_reward: float,
        top_reward_fraction: float,
        hard_positive_fraction: float,
        hard_negative_fraction: float,
    ) -> None:
        super().__init__(capacity, max_total_action_tokens=max_total_action_tokens)
        self.invalid_terminal_reward = max(float(invalid_terminal_reward), 1.0e-12)
        self.top_reward_fraction = max(float(top_reward_fraction), 0.0)
        self.hard_positive_fraction = max(float(hard_positive_fraction), 0.0)
        self.hard_negative_fraction = max(float(hard_negative_fraction), 0.0)

    def _is_hard_negative(self, trajectory: SampledStageTrajectory) -> bool:
        return (
            not trajectory.is_valid
            or trajectory.is_duplicate
            or abs(float(trajectory.terminal_reward) - self.invalid_terminal_reward) <= 1.0e-12
        )

    def _bucket_population(
        self,
    ) -> tuple[
        list[SampledStageTrajectory],
        list[SampledStageTrajectory],
        list[SampledStageTrajectory],
    ]:
        top_reward: list[SampledStageTrajectory] = []
        hard_positive: list[SampledStageTrajectory] = []
        hard_negative: list[SampledStageTrajectory] = []
        valid_non_hard_negative: list[SampledStageTrajectory] = []

        for trajectory in self._items:
            if self._is_hard_negative(trajectory):
                hard_negative.append(trajectory)
            else:
                valid_non_hard_negative.append(trajectory)

        ordered_valid = sorted(
            valid_non_hard_negative,
            key=lambda item: float(item.terminal_reward),
            reverse=True,
        )
        cutoff = math.ceil(len(ordered_valid) / 2)
        top_reward.extend(ordered_valid[:cutoff])
        hard_positive.extend(ordered_valid[cutoff:])
        return top_reward, hard_positive, hard_negative

    def _normalized_bucket_fractions(self) -> tuple[float, float, float]:
        total = self.top_reward_fraction + self.hard_positive_fraction + self.hard_negative_fraction
        if total <= 0.0:
            return 0.5, 0.25, 0.25
        return (
            self.top_reward_fraction / total,
            self.hard_positive_fraction / total,
            self.hard_negative_fraction / total,
        )

    def _target_bucket_counts(self, batch_size: int) -> tuple[int, int, int]:
        fractions = self._normalized_bucket_fractions()
        exact_counts = [batch_size * fraction for fraction in fractions]
        bucket_counts = [int(math.floor(count)) for count in exact_counts]
        remaining = batch_size - sum(bucket_counts)
        if remaining <= 0:
            return tuple(bucket_counts)
        remainders = sorted(
            (
                exact_counts[index] - bucket_counts[index],
                index,
            )
            for index in range(len(bucket_counts))
        )
        for _remainder, index in reversed(remainders[-remaining:]):
            bucket_counts[index] += 1
        return tuple(bucket_counts)

    def _count_selected_buckets(
        self,
        sampled: Sequence[SampledStageTrajectory],
        *,
        top_reward: Sequence[SampledStageTrajectory],
        hard_positive: Sequence[SampledStageTrajectory],
        hard_negative: Sequence[SampledStageTrajectory],
    ) -> tuple[int, int, int]:
        top_ids = {id(item) for item in top_reward}
        hard_positive_ids = {id(item) for item in hard_positive}
        hard_negative_ids = {id(item) for item in hard_negative}

        top_count = 0
        hard_positive_count = 0
        hard_negative_count = 0
        for trajectory in sampled:
            trajectory_id = id(trajectory)
            if trajectory_id in top_ids:
                top_count += 1
            elif trajectory_id in hard_positive_ids:
                hard_positive_count += 1
            elif trajectory_id in hard_negative_ids:
                hard_negative_count += 1
        return top_count, hard_positive_count, hard_negative_count

    def _draw_without_replacement(
        self,
        pool: list[SampledStageTrajectory],
        count: int,
        *,
        generator: random.Random,
    ) -> tuple[list[SampledStageTrajectory], list[SampledStageTrajectory]]:
        if count <= 0 or not pool:
            return [], pool
        if count >= len(pool):
            return list(pool), []
        selected = generator.sample(pool, count)
        selected_ids = {id(item) for item in selected}
        remaining = [item for item in pool if id(item) not in selected_ids]
        return selected, remaining

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
        top_reward, hard_positive, hard_negative = self._bucket_population()
        population = [*top_reward, *hard_positive, *hard_negative]
        if with_replacement:
            target_top, target_hard_positive, target_hard_negative = self._target_bucket_counts(
                batch_size
            )
            selected_top = (
                [generator.choice(top_reward) for _ in range(target_top)] if top_reward else []
            )
            selected_hard_positive = (
                [generator.choice(hard_positive) for _ in range(target_hard_positive)]
                if hard_positive
                else []
            )
            selected_hard_negative = (
                [generator.choice(hard_negative) for _ in range(target_hard_negative)]
                if hard_negative
                else []
            )
            sampled = [*selected_top, *selected_hard_positive, *selected_hard_negative]
            while len(sampled) < batch_size and population:
                sampled.append(generator.choice(population))
            top_count, hard_positive_count, hard_negative_count = self._count_selected_buckets(
                sampled,
                top_reward=top_reward,
                hard_positive=hard_positive,
                hard_negative=hard_negative,
            )
            return ReplaySampleBatch(
                tuple(sampled),
                top_reward_count=top_count,
                hard_positive_count=hard_positive_count,
                hard_negative_count=hard_negative_count,
            )

        if batch_size >= len(population):
            return ReplaySampleBatch(
                tuple(population),
                top_reward_count=len(top_reward),
                hard_positive_count=len(hard_positive),
                hard_negative_count=len(hard_negative),
            )

        target_top, target_hard_positive, target_hard_negative = self._target_bucket_counts(batch_size)
        selected_top, remaining_top = self._draw_without_replacement(
            list(top_reward),
            target_top,
            generator=generator,
        )
        selected_hard_positive, remaining_hard_positive = self._draw_without_replacement(
            list(hard_positive),
            target_hard_positive,
            generator=generator,
        )
        selected_hard_negative, remaining_hard_negative = self._draw_without_replacement(
            list(hard_negative),
            target_hard_negative,
            generator=generator,
        )

        sampled = [*selected_top, *selected_hard_positive, *selected_hard_negative]
        remaining_needed = batch_size - len(sampled)
        if remaining_needed > 0:
            remainder_pool = [
                *remaining_top,
                *remaining_hard_positive,
                *remaining_hard_negative,
            ]
            sampled.extend(generator.sample(remainder_pool, remaining_needed))

        top_count, hard_positive_count, hard_negative_count = self._count_selected_buckets(
            sampled,
            top_reward=top_reward,
            hard_positive=hard_positive,
            hard_negative=hard_negative,
        )

        return ReplaySampleBatch(
            tuple(sampled),
            top_reward_count=top_count,
            hard_positive_count=hard_positive_count,
            hard_negative_count=hard_negative_count,
        )


def build_replay_buffer(
    *,
    capacity: int,
    max_total_action_tokens: int,
    buffer_type: str,
    invalid_terminal_reward: float,
    top_reward_fraction: float,
    hard_positive_fraction: float,
    hard_negative_fraction: float,
) -> ReplayBuffer:
    normalized_buffer_type = str(buffer_type).strip().lower()
    if normalized_buffer_type in {"uniform", "random"}:
        return UniformReplayBuffer(
            capacity,
            max_total_action_tokens=max_total_action_tokens,
        )
    if normalized_buffer_type == "priority":
        return PriorityReplayBuffer(
            capacity,
            max_total_action_tokens=max_total_action_tokens,
            invalid_terminal_reward=invalid_terminal_reward,
            top_reward_fraction=top_reward_fraction,
            hard_positive_fraction=hard_positive_fraction,
            hard_negative_fraction=hard_negative_fraction,
        )
    raise ValueError(f"Unsupported replay buffer_type: {buffer_type!r}")


TrajectoryReplayBuffer = UniformReplayBuffer


__all__ = [
    "OnPolicyBatch",
    "PriorityReplayBuffer",
    "ReplayBuffer",
    "ReplaySampleBatch",
    "TrajectoryReplayBuffer",
    "UniformReplayBuffer",
    "build_replay_buffer",
]
