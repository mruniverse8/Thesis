from __future__ import annotations

from collections import Counter, deque
import math
import random
from typing import Sequence

from .buffer import ReplaySampleBatch
from .losses import trajectory_balance_residual
from .trajectory import SampledStageTrajectory, ScoredStageTrajectory


class _FenwickTree:
    def __init__(self, capacity: int) -> None:
        self._size = int(capacity)
        self._tree = [0.0] * (self._size + 1)
        self._values = [0.0] * self._size
        self._total = 0.0

    @property
    def total(self) -> float:
        return self._total

    def set(self, index: int, value: float) -> None:
        value = max(0.0, float(value))
        delta = value - self._values[index]
        if delta == 0.0:
            return
        self._values[index] = value
        self._total += delta
        cursor = index + 1
        while cursor <= self._size:
            self._tree[cursor] += delta
            cursor += cursor & -cursor

    def sample(self, generator: random.Random) -> int | None:
        if self._total <= 0.0:
            return None
        target = float(generator.random()) * self._total
        index = 0
        bit = 1 << (self._size.bit_length() - 1)
        while bit:
            next_index = index + bit
            if next_index <= self._size and target >= self._tree[next_index]:
                index = next_index
                target -= self._tree[next_index]
            bit >>= 1
        return min(index, self._size - 1)


class ExperimentalMixtureReplayBuffer:
    def __init__(
        self,
        capacity: int,
        *,
        recent_fraction: float = 0.40,
        reward_fraction: float = 0.40,
        uniform_fraction: float = 0.20,
        tb_residual_fraction: float = 0.0,
        reward_temperature: float = 1.0,
        tb_residual_temperature: float = 1.0,
        recent_window_size: int = 64,
        max_invalid_fraction: float = 0.20,
        max_duplicate_fraction: float = 0.10,
    ) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be positive.")
        self.capacity = int(capacity)
        self.recent_fraction = max(0.0, float(recent_fraction))
        self.reward_fraction = max(0.0, float(reward_fraction))
        self.uniform_fraction = max(0.0, float(uniform_fraction))
        self.tb_residual_fraction = max(0.0, float(tb_residual_fraction))
        self.reward_temperature = max(float(reward_temperature), 1.0e-6)
        self.tb_residual_temperature = max(float(tb_residual_temperature), 1.0e-6)
        self.recent_window_size = max(1, int(recent_window_size))
        self.max_invalid_fraction = min(1.0, max(0.0, float(max_invalid_fraction)))
        self.max_duplicate_fraction = min(1.0, max(0.0, float(max_duplicate_fraction)))

        self._items: list[SampledStageTrajectory | None] = [None] * self.capacity
        self._free_indices = list(range(self.capacity - 1, -1, -1))
        self._order: deque[int] = deque()
        self._recent_indices: set[int] = set()
        self._index_by_object_id: dict[int, int] = {}
        self._insert_sequences = [0] * self.capacity
        self._tb_residuals = [0.0] * self.capacity
        self._next_sequence = 1
        self._total_action_tokens = 0

        self._uniform_tree = _FenwickTree(self.capacity)
        self._recent_tree = _FenwickTree(self.capacity)
        self._reward_tree = _FenwickTree(self.capacity)
        self._tb_residual_tree = _FenwickTree(self.capacity)

    def __len__(self) -> int:
        return len(self._order)

    @property
    def total_action_tokens(self) -> int:
        return self._total_action_tokens

    def snapshot(self) -> tuple[SampledStageTrajectory, ...]:
        return tuple(self._items[index] for index in self._order if self._items[index] is not None)

    def extend(self, trajectories: Sequence[SampledStageTrajectory]) -> None:
        for trajectory in trajectories:
            self.add(trajectory)

    def add(self, trajectory: SampledStageTrajectory) -> None:
        while len(self) >= self.capacity:
            self._evict_one()
        index = self._free_indices.pop()
        self._items[index] = trajectory
        self._index_by_object_id[id(trajectory)] = index
        self._insert_sequences[index] = self._next_sequence
        self._next_sequence += 1
        self._tb_residuals[index] = 0.0
        self._total_action_tokens += trajectory.num_actions
        self._order.append(index)
        self._set_active_weights(index, active=True)
        self._refresh_recent_tree()

    def observe_scored(self, trajectories: Sequence[ScoredStageTrajectory]) -> None:
        del trajectories

    def sample(
        self,
        batch_size: int,
        *,
        rng: random.Random | None = None,
        with_replacement: bool = False,
    ) -> ReplaySampleBatch:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive.")
        if not self._order:
            return ReplaySampleBatch(())

        if not with_replacement and batch_size >= len(self):
            population = self.snapshot()
            return ReplaySampleBatch(population, source_counts={"uniform": len(population)})

        generator = rng or random
        selected: list[SampledStageTrajectory] = []
        source_counts: Counter[str] = Counter()
        muted_indices: list[int] = []
        try:
            for _ in range(batch_size):
                source = self._choose_source(generator)
                if source is None:
                    break
                index = self._source_tree(source).sample(generator)
                if index is None:
                    break
                trajectory = self._items[index]
                if trajectory is None:
                    self._set_active_weights(index, active=False)
                    continue
                selected.append(trajectory)
                source_counts[source] += 1
                if not with_replacement:
                    self._set_active_weights(index, active=False)
                    muted_indices.append(index)
        finally:
            for index in muted_indices:
                self._set_active_weights(index, active=True)

        return ReplaySampleBatch(tuple(selected), source_counts=dict(source_counts))

    def _choose_source(self, generator: random.Random) -> str | None:
        available = [
            (source, fraction)
            for source, fraction in self._source_fractions().items()
            if fraction > 0.0 and self._source_tree(source).total > 0.0
        ]
        if not available:
            return None
        total = sum(fraction for _source, fraction in available)
        target = float(generator.random()) * total
        cumulative = 0.0
        for source, fraction in available:
            cumulative += fraction
            if target <= cumulative:
                return source
        return available[-1][0]

    def _source_fractions(self) -> dict[str, float]:
        fractions = {
            "recent": self.recent_fraction,
            "reward": self.reward_fraction,
            "uniform": self.uniform_fraction,
        }
        if self.tb_residual_fraction > 0.0:
            fractions["tb_residual"] = self.tb_residual_fraction
        return fractions

    def _source_tree(self, source: str) -> _FenwickTree:
        if source == "recent":
            return self._recent_tree
        if source == "reward":
            return self._reward_tree
        if source == "tb_residual":
            return self._tb_residual_tree
        if source == "uniform":
            return self._uniform_tree
        raise ValueError(f"Unsupported replay sample source: {source!r}")

    def _evict_one(self) -> None:
        occupied = list(self._order)
        if not occupied:
            return
        invalid_indices = [
            index for index in occupied if self._items[index] is not None and not self._items[index].is_valid
        ]
        duplicate_indices = [
            index
            for index in occupied
            if self._items[index] is not None and self._items[index].is_duplicate
        ]
        if invalid_indices and len(invalid_indices) / len(occupied) > self.max_invalid_fraction:
            victim = min(invalid_indices, key=self._keep_score)
        elif duplicate_indices and len(duplicate_indices) / len(occupied) > self.max_duplicate_fraction:
            victim = min(duplicate_indices, key=self._keep_score)
        else:
            victim = min(occupied, key=self._keep_score)
        self._evict_index(victim)

    def _evict_index(self, index: int) -> None:
        trajectory = self._items[index]
        if trajectory is None:
            return
        self._order.remove(index)
        self._items[index] = None
        self._index_by_object_id.pop(id(trajectory), None)
        self._insert_sequences[index] = 0
        self._tb_residuals[index] = 0.0
        self._total_action_tokens -= trajectory.num_actions
        self._set_active_weights(index, active=False)
        self._free_indices.append(index)
        self._refresh_recent_tree()

    def _keep_score(self, index: int) -> float:
        trajectory = self._items[index]
        if trajectory is None:
            return float("-inf")
        reward_score = math.log(max(float(trajectory.terminal_reward), 1.0e-12))
        validity_score = 3.0 if trajectory.is_valid else -3.0
        duplicate_score = -2.0 if trajectory.is_duplicate else 0.0
        recency_score = self._insert_sequences[index] / max(1.0, float(self._next_sequence))
        return reward_score + validity_score + duplicate_score + recency_score

    def _refresh_recent_tree(self) -> None:
        self._recent_indices = set(list(self._order)[-self.recent_window_size :])
        for index in range(self.capacity):
            self._recent_tree.set(index, 0.0)
        for index in self._recent_indices:
            if self._items[index] is not None:
                self._recent_tree.set(index, 1.0)

    def _set_active_weights(self, index: int, *, active: bool) -> None:
        trajectory = self._items[index]
        if not active or trajectory is None:
            self._uniform_tree.set(index, 0.0)
            self._reward_tree.set(index, 0.0)
            self._tb_residual_tree.set(index, 0.0)
            self._recent_tree.set(index, 0.0)
            return
        self._uniform_tree.set(index, 1.0)
        self._reward_tree.set(index, self._reward_weight(trajectory))
        self._tb_residual_tree.set(index, self._tb_residual_weight(index))
        self._recent_tree.set(index, 1.0 if index in self._recent_indices else 0.0)

    def _reward_weight(self, trajectory: SampledStageTrajectory) -> float:
        log_weight = math.log(max(float(trajectory.terminal_reward), 1.0e-12))
        log_weight /= self.reward_temperature
        return math.exp(min(60.0, max(-60.0, log_weight)))

    def _tb_residual_weight(self, index: int) -> float:
        log_weight = math.log1p(max(0.0, float(self._tb_residuals[index])))
        log_weight /= self.tb_residual_temperature
        return math.exp(min(60.0, max(-60.0, log_weight)))


class ExperimentalTBMixtureReplayBuffer(ExperimentalMixtureReplayBuffer):
    def __init__(
        self,
        capacity: int,
        *,
        recent_fraction: float = 0.40,
        reward_fraction: float = 0.40,
        uniform_fraction: float = 0.20,
        tb_residual_fraction: float = 0.20,
        reward_temperature: float = 1.0,
        tb_residual_temperature: float = 1.0,
        recent_window_size: int = 64,
        max_invalid_fraction: float = 0.20,
        max_duplicate_fraction: float = 0.10,
    ) -> None:
        super().__init__(
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

    def observe_scored(self, trajectories: Sequence[ScoredStageTrajectory]) -> None:
        for trajectory in trajectories:
            index = self._index_by_object_id.get(id(trajectory.sampled))
            if index is None or self._items[index] is None:
                continue
            try:
                residual = float(trajectory_balance_residual(trajectory).detach().abs().item())
            except (RuntimeError, ValueError, TypeError):
                residual = 0.0
            if not math.isfinite(residual):
                residual = 0.0
            self._tb_residuals[index] = residual
            self._tb_residual_tree.set(index, self._tb_residual_weight(index))


__all__ = [
    "ExperimentalMixtureReplayBuffer",
    "ExperimentalTBMixtureReplayBuffer",
]
