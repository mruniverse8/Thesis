"""Streaming helpers for exact notebook-local GFlowNet evaluation metrics."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import DefaultDict, Sequence

from rdkit import DataStructs

from evaluation_metrics import EvaluationMetricConfig, EvaluationMetricsResult, GenerationGroup, MoleculeInput
from evaluation_metrics.metrics import (
    PreparedMolecule,
    _best_dice_similarity,
    _parse_molecule_with_fingerprint,
    _prepare_targets,
    n_circles,
)
from post_training.gflownet.trajectory import SampledStageTrajectory


_BASE_TERMINATION_REASONS = (
    "stop_token",
    "eos_token",
    "max_stage_new_tokens",
    "max_sequence_length",
)


def _normalize_metric_key_component(value: str) -> str:
    return value.strip().lower().replace("-", "_").replace(" ", "_")


def _extend_pairwise_similarity_state(
    fingerprint: object,
    *,
    fingerprints: list[object],
    pairwise_similarity_chunks: list[float],
) -> int:
    if fingerprints:
        similarities = DataStructs.BulkTanimotoSimilarity(fingerprint, list(fingerprints))
        for index, similarity in enumerate(similarities):
            pairwise_similarity_chunks[index] += float(similarity)
        pair_count_increment = len(similarities)
    else:
        pair_count_increment = 0
    fingerprints.append(fingerprint)
    pairwise_similarity_chunks.append(0.0)
    return pair_count_increment


def _internal_diversity_from_pairwise_state(
    *,
    pairwise_similarity_chunks: Sequence[float],
    pair_count: int,
) -> float:
    if pair_count == 0:
        return 0.0
    pairwise_similarity_sum = 0.0
    for chunk_sum in pairwise_similarity_chunks:
        pairwise_similarity_sum += chunk_sum
    return 1.0 - (pairwise_similarity_sum / pair_count)


@dataclass(slots=True)
class IncrementalGenerationMetrics:
    """Accumulate exact grouped generation metrics without rescanning prefixes."""

    config: EvaluationMetricConfig
    num_groups: int = 0
    num_candidates: int = 0
    num_valid_candidates: int = 0
    num_duplicate_valid_candidates: int = 0
    num_accepted: int = 0
    _max_dice_sum: float = 0.0
    _valid_seen_smiles: set[str] = field(default_factory=set)
    _valid_fingerprints: list[object] = field(default_factory=list)
    _valid_pairwise_similarity_chunks: list[float] = field(default_factory=list)
    _valid_pair_count: int = 0
    _target_seen_smiles: set[str] = field(default_factory=set)
    _novel_accepted_smiles: set[str] = field(default_factory=set)
    _accepted_by_smiles: dict[str, PreparedMolecule] = field(default_factory=dict)
    _accepted_fingerprints: list[object] = field(default_factory=list)
    _accepted_pairwise_similarity_chunks: list[float] = field(default_factory=list)
    _accepted_pair_count: int = 0
    _prepared_targets_cache: dict[tuple[MoleculeInput, ...], tuple[PreparedMolecule, ...]] = field(
        default_factory=dict
    )

    def _prepared_targets(self, targets: tuple[MoleculeInput, ...]) -> tuple[PreparedMolecule, ...]:
        cached = self._prepared_targets_cache.get(targets)
        if cached is not None:
            return cached
        prepared = _prepare_targets(targets, config=self.config)
        self._prepared_targets_cache[targets] = prepared
        return prepared

    def _update_target_seen_smiles(self, prepared_targets: Sequence[PreparedMolecule]) -> None:
        for target in prepared_targets:
            canonical_smiles = target.record.canonical_smiles
            if canonical_smiles is None or canonical_smiles in self._target_seen_smiles:
                continue
            self._target_seen_smiles.add(canonical_smiles)
            self._novel_accepted_smiles.discard(canonical_smiles)

    def _add_accepted_unique(self, canonical_smiles: str, prepared: PreparedMolecule) -> None:
        if canonical_smiles in self._accepted_by_smiles:
            return
        self._accepted_pair_count += _extend_pairwise_similarity_state(
            prepared.fingerprint,
            fingerprints=self._accepted_fingerprints,
            pairwise_similarity_chunks=self._accepted_pairwise_similarity_chunks,
        )
        self._accepted_by_smiles[canonical_smiles] = prepared
        if canonical_smiles not in self._target_seen_smiles:
            self._novel_accepted_smiles.add(canonical_smiles)

    def _add_valid_candidate(self, fingerprint: object) -> None:
        self._valid_pair_count += _extend_pairwise_similarity_state(
            fingerprint,
            fingerprints=self._valid_fingerprints,
            pairwise_similarity_chunks=self._valid_pairwise_similarity_chunks,
        )

    def update(self, group: GenerationGroup) -> None:
        self.num_groups += 1
        prepared_targets = self._prepared_targets(group.targets)
        self._update_target_seen_smiles(prepared_targets)

        for candidate in group.candidates:
            self.num_candidates += 1
            candidate_record, candidate_fp = _parse_molecule_with_fingerprint(
                candidate,
                config=self.config,
            )
            canonical_smiles = candidate_record.canonical_smiles
            is_valid = (
                candidate_record.is_valid
                and canonical_smiles is not None
                and candidate_fp is not None
            )
            if not is_valid:
                continue

            assert canonical_smiles is not None
            assert candidate_fp is not None
            self.num_valid_candidates += 1
            self.num_duplicate_valid_candidates += int(canonical_smiles in self._valid_seen_smiles)
            self._valid_seen_smiles.add(canonical_smiles)
            self._add_valid_candidate(candidate_fp)

            if not prepared_targets:
                continue

            max_dice_similarity, _ = _best_dice_similarity(candidate_fp, prepared_targets)
            self._max_dice_sum += max_dice_similarity
            if max_dice_similarity <= self.config.acceptance_dice_threshold:
                continue

            self.num_accepted += 1
            prepared_candidate = PreparedMolecule(
                molecule_input=candidate,
                record=candidate_record,
                fingerprint=candidate_fp,
            )
            self._add_accepted_unique(canonical_smiles, prepared_candidate)

    @property
    def accepted_unique_count(self) -> int:
        return len(self._accepted_by_smiles)

    @property
    def internal_diversity(self) -> float:
        return _internal_diversity_from_pairwise_state(
            pairwise_similarity_chunks=self._accepted_pairwise_similarity_chunks,
            pair_count=self._accepted_pair_count,
        )

    @property
    def valid_internal_diversity(self) -> float:
        return _internal_diversity_from_pairwise_state(
            pairwise_similarity_chunks=self._valid_pairwise_similarity_chunks,
            pair_count=self._valid_pair_count,
        )

    def to_result(self) -> EvaluationMetricsResult:
        if self.config.compute_n_circles:
            n_circles_value, n_circles_exact_value = n_circles(
                tuple(self._accepted_fingerprints),
                tanimoto_threshold=self.config.n_circles_tanimoto_threshold,
                exact_max_molecules=self.config.n_circles_exact_max_molecules,
            )
        else:
            n_circles_value, n_circles_exact_value = 0, False
        novelty_count = len(self._novel_accepted_smiles)
        novelty_fraction = novelty_count / max(self.accepted_unique_count, 1)
        # Prefix-average metric: cumulative over all generated samples seen so far.
        prefix_valid_fraction = self.num_valid_candidates / max(self.num_candidates, 1)
        prefix_duplicate_fraction = self.num_duplicate_valid_candidates / max(
            self.num_candidates,
            1,
        )
        prefix_duplicate_valid_fraction = self.num_duplicate_valid_candidates / max(
            self.num_valid_candidates,
            1,
        )
        prefix_average_max_dice_similarity = self._max_dice_sum / max(self.num_candidates, 1)
        return EvaluationMetricsResult(
            config=self.config,
            num_groups=self.num_groups,
            num_candidates=self.num_candidates,
            num_valid_candidates=self.num_valid_candidates,
            num_unique_valid_molecules=len(self._valid_seen_smiles),
            num_accepted=self.num_accepted,
            accepted_unique_count=self.accepted_unique_count,
            n_circles=n_circles_value,
            n_circles_exact=n_circles_exact_value,
            valid_fraction=prefix_valid_fraction,
            internal_diversity=self.internal_diversity,
            mean_max_dice_similarity=prefix_average_max_dice_similarity,
            prefix_valid_fraction=prefix_valid_fraction,
            prefix_duplicate_fraction=prefix_duplicate_fraction,
            prefix_duplicate_valid_fraction=prefix_duplicate_valid_fraction,
            prefix_average_max_dice_similarity=prefix_average_max_dice_similarity,
            prefix_accepted_unique_internal_diversity=self.internal_diversity,
            prefix_valid_internal_diversity=self.valid_internal_diversity,
            accepted_unique_smiles=tuple(sorted(self._accepted_by_smiles)),
            novelty_count=novelty_count,
            novelty_fraction=novelty_fraction,
            novel_accepted_unique_smiles=tuple(sorted(self._novel_accepted_smiles)),
            candidate_assessments=(),
        )


@dataclass(slots=True)
class _StageAccumulator:
    num_trajectories: int = 0
    num_valid: int = 0
    stage_reward_sum: float = 0.0
    num_actions_sum: float = 0.0
    invalid_reward_floor_count: int = 0
    termination_counts: DefaultDict[str, int] = field(default_factory=lambda: defaultdict(int))

    def update(
        self,
        trajectory: SampledStageTrajectory,
        *,
        invalid_terminal_reward: float | None,
    ) -> None:
        self.num_trajectories += 1
        self.num_valid += int(trajectory.is_valid)
        self.stage_reward_sum += float(trajectory.terminal_reward)
        self.num_actions_sum += float(trajectory.num_actions)
        self.termination_counts[str(trajectory.termination_reason)] += 1
        if invalid_terminal_reward is not None and (
            abs(float(trajectory.terminal_reward) - float(invalid_terminal_reward)) <= 1.0e-12
        ):
            self.invalid_reward_floor_count += 1

    def to_state_dict(self) -> dict[str, object]:
        return {
            "num_trajectories": int(self.num_trajectories),
            "num_valid": int(self.num_valid),
            "stage_reward_sum": float(self.stage_reward_sum),
            "num_actions_sum": float(self.num_actions_sum),
            "invalid_reward_floor_count": int(self.invalid_reward_floor_count),
            "termination_counts": {
                str(reason): int(count)
                for reason, count in sorted(self.termination_counts.items())
            },
        }

    @classmethod
    def from_state_dict(cls, payload: dict[str, object]) -> "_StageAccumulator":
        accumulator = cls()
        accumulator.num_trajectories = int(payload.get("num_trajectories", 0))
        accumulator.num_valid = int(payload.get("num_valid", 0))
        accumulator.stage_reward_sum = float(payload.get("stage_reward_sum", 0.0))
        accumulator.num_actions_sum = float(payload.get("num_actions_sum", 0.0))
        accumulator.invalid_reward_floor_count = int(payload.get("invalid_reward_floor_count", 0))
        termination_payload = payload.get("termination_counts", {})
        if isinstance(termination_payload, dict):
            for reason, count in termination_payload.items():
                accumulator.termination_counts[str(reason)] = int(count)
        return accumulator

    def merge(self, other: "_StageAccumulator") -> "_StageAccumulator":
        self.num_trajectories += int(other.num_trajectories)
        self.num_valid += int(other.num_valid)
        self.stage_reward_sum += float(other.stage_reward_sum)
        self.num_actions_sum += float(other.num_actions_sum)
        self.invalid_reward_floor_count += int(other.invalid_reward_floor_count)
        for reason, count in other.termination_counts.items():
            self.termination_counts[str(reason)] += int(count)
        return self

    def metrics(self, *, stage_index: int, invalid_terminal_reward: float | None) -> dict[str, float]:
        stage_prefix = f"stage{stage_index}_"
        metrics: dict[str, float] = {
            f"{stage_prefix}num_trajectories": float(self.num_trajectories),
            f"{stage_prefix}valid_fraction": (
                float(self.num_valid / self.num_trajectories) if self.num_trajectories > 0 else 0.0
            ),
            f"{stage_prefix}mean_stage_reward": (
                float(self.stage_reward_sum / self.num_trajectories)
                if self.num_trajectories > 0
                else 0.0
            ),
            f"{stage_prefix}mean_num_actions": (
                float(self.num_actions_sum / self.num_trajectories)
                if self.num_trajectories > 0
                else 0.0
            ),
            f"{stage_prefix}invalid_reward_floor_fraction": (
                float(self.invalid_reward_floor_count / self.num_trajectories)
                if invalid_terminal_reward is not None and self.num_trajectories > 0
                else 0.0
            ),
        }
        for reason in _BASE_TERMINATION_REASONS:
            metrics[f"{stage_prefix}termination_fraction_{reason}"] = 0.0
        if self.num_trajectories == 0:
            return metrics
        for reason, count in sorted(self.termination_counts.items()):
            metrics[
                f"{stage_prefix}termination_fraction_{_normalize_metric_key_component(reason)}"
            ] = count / self.num_trajectories
        return metrics


@dataclass(slots=True)
class IncrementalRolloutMetrics:
    """Accumulate exact rollout diagnostics without rescanning all trajectories."""

    max_molecules_per_sequence: int | None = None
    invalid_terminal_reward: float | None = None
    stage_indices: tuple[int, ...] = (1, 2)
    num_rollouts: int = 0
    total_trajectories: int = 0
    planned_trajectory_length_sum: int = 0
    max_planned_trajectory_length: int = 0
    trajectory_length_sum: int = 0
    max_trajectory_length: int = 0
    planned_trajectory_length_2_plus_count: int = 0
    trajectory_length_2_plus_count: int = 0
    trajectory_length_3_plus_count: int = 0
    reached_planned_trajectory_length_count: int = 0
    trajectory_length_1_count: int = 0
    trajectory_length_2_count: int = 0
    trajectory_length_3_plus_rollout_count: int = 0
    invalid_reward_floor_count: int = 0
    _stage_accumulators: dict[int, _StageAccumulator] = field(default_factory=dict)

    def update(self, trajectories: Sequence[SampledStageTrajectory]) -> None:
        grouped_rollouts: dict[str, list[SampledStageTrajectory]] = defaultdict(list)
        for trajectory in trajectories:
            grouped_rollouts[trajectory.rollout_id].append(trajectory)
        for rollout in grouped_rollouts.values():
            self._update_rollout(rollout)

    def _update_rollout(self, rollout: Sequence[SampledStageTrajectory]) -> None:
        if not rollout:
            return
        ordered_rollout = sorted(rollout, key=lambda item: item.stage_index)
        trajectory_length = len(ordered_rollout)
        planned_trajectory_length = (
            max(1, int(self.max_molecules_per_sequence))
            if self.max_molecules_per_sequence is not None
            else max(1, trajectory_length)
        )

        self.num_rollouts += 1
        self.planned_trajectory_length_sum += planned_trajectory_length
        self.max_planned_trajectory_length = max(
            self.max_planned_trajectory_length,
            planned_trajectory_length,
        )
        self.trajectory_length_sum += trajectory_length
        self.max_trajectory_length = max(self.max_trajectory_length, trajectory_length)
        self.planned_trajectory_length_2_plus_count += int(planned_trajectory_length >= 2)
        self.trajectory_length_2_plus_count += int(trajectory_length >= 2)
        self.trajectory_length_3_plus_count += int(trajectory_length >= 3)
        self.reached_planned_trajectory_length_count += int(trajectory_length >= planned_trajectory_length)
        self.trajectory_length_1_count += int(trajectory_length == 1)
        self.trajectory_length_2_count += int(trajectory_length == 2)
        self.trajectory_length_3_plus_rollout_count += int(trajectory_length >= 3)

        for trajectory in ordered_rollout:
            self.total_trajectories += 1
            if self.invalid_terminal_reward is not None and (
                abs(float(trajectory.terminal_reward) - float(self.invalid_terminal_reward)) <= 1.0e-12
            ):
                self.invalid_reward_floor_count += 1
            stage_state = self._stage_accumulators.setdefault(
                int(trajectory.stage_index),
                _StageAccumulator(),
            )
            stage_state.update(
                trajectory,
                invalid_terminal_reward=self.invalid_terminal_reward,
            )

    def to_metrics(self) -> dict[str, float]:
        metrics: dict[str, float] = {
            "num_rollouts": float(self.num_rollouts),
            "mean_planned_trajectory_length": (
                float(self.planned_trajectory_length_sum / self.num_rollouts)
                if self.num_rollouts > 0
                else 0.0
            ),
            "max_planned_trajectory_length": float(self.max_planned_trajectory_length),
            "mean_trajectory_length": (
                float(self.trajectory_length_sum / self.num_rollouts) if self.num_rollouts > 0 else 0.0
            ),
            "max_trajectory_length": float(self.max_trajectory_length),
            "fraction_rollouts_planned_trajectory_length_2_plus": (
                float(self.planned_trajectory_length_2_plus_count / self.num_rollouts)
                if self.num_rollouts > 0
                else 0.0
            ),
            "fraction_rollouts_trajectory_length_2_plus": (
                float(self.trajectory_length_2_plus_count / self.num_rollouts)
                if self.num_rollouts > 0
                else 0.0
            ),
            "fraction_rollouts_trajectory_length_3_plus": (
                float(self.trajectory_length_3_plus_count / self.num_rollouts)
                if self.num_rollouts > 0
                else 0.0
            ),
            "fraction_rollouts_reaching_planned_trajectory_length": (
                float(self.reached_planned_trajectory_length_count / self.num_rollouts)
                if self.num_rollouts > 0
                else 0.0
            ),
            "trajectory_length_1_fraction": (
                float(self.trajectory_length_1_count / self.num_rollouts) if self.num_rollouts > 0 else 0.0
            ),
            "trajectory_length_2_fraction": (
                float(self.trajectory_length_2_count / self.num_rollouts) if self.num_rollouts > 0 else 0.0
            ),
            "trajectory_length_3_plus_fraction": (
                float(self.trajectory_length_3_plus_rollout_count / self.num_rollouts)
                if self.num_rollouts > 0
                else 0.0
            ),
            "invalid_reward_floor_fraction": (
                float(self.invalid_reward_floor_count / self.total_trajectories)
                if self.invalid_terminal_reward is not None and self.total_trajectories > 0
                else 0.0
            ),
        }
        for stage_index in self.stage_indices:
            metrics.update(
                self._stage_accumulators.get(
                    stage_index,
                    _StageAccumulator(),
                ).metrics(
                    stage_index=stage_index,
                    invalid_terminal_reward=self.invalid_terminal_reward,
                )
            )
        return metrics

    def to_state_dict(self) -> dict[str, object]:
        return {
            "max_molecules_per_sequence": self.max_molecules_per_sequence,
            "invalid_terminal_reward": self.invalid_terminal_reward,
            "stage_indices": [int(stage_index) for stage_index in self.stage_indices],
            "num_rollouts": int(self.num_rollouts),
            "total_trajectories": int(self.total_trajectories),
            "planned_trajectory_length_sum": int(self.planned_trajectory_length_sum),
            "max_planned_trajectory_length": int(self.max_planned_trajectory_length),
            "trajectory_length_sum": int(self.trajectory_length_sum),
            "max_trajectory_length": int(self.max_trajectory_length),
            "planned_trajectory_length_2_plus_count": int(self.planned_trajectory_length_2_plus_count),
            "trajectory_length_2_plus_count": int(self.trajectory_length_2_plus_count),
            "trajectory_length_3_plus_count": int(self.trajectory_length_3_plus_count),
            "reached_planned_trajectory_length_count": int(self.reached_planned_trajectory_length_count),
            "trajectory_length_1_count": int(self.trajectory_length_1_count),
            "trajectory_length_2_count": int(self.trajectory_length_2_count),
            "trajectory_length_3_plus_rollout_count": int(self.trajectory_length_3_plus_rollout_count),
            "invalid_reward_floor_count": int(self.invalid_reward_floor_count),
            "stage_accumulators": {
                str(stage_index): accumulator.to_state_dict()
                for stage_index, accumulator in sorted(self._stage_accumulators.items())
            },
        }

    @classmethod
    def from_state_dict(cls, payload: dict[str, object]) -> "IncrementalRolloutMetrics":
        stage_indices_payload = payload.get("stage_indices", (1, 2))
        stage_indices = tuple(int(stage_index) for stage_index in stage_indices_payload)  # type: ignore[arg-type]
        metrics = cls(
            max_molecules_per_sequence=(
                None
                if payload.get("max_molecules_per_sequence") is None
                else int(payload["max_molecules_per_sequence"])  # type: ignore[index]
            ),
            invalid_terminal_reward=(
                None
                if payload.get("invalid_terminal_reward") is None
                else float(payload["invalid_terminal_reward"])  # type: ignore[index]
            ),
            stage_indices=stage_indices or (1, 2),
        )
        metrics.num_rollouts = int(payload.get("num_rollouts", 0))
        metrics.total_trajectories = int(payload.get("total_trajectories", 0))
        metrics.planned_trajectory_length_sum = int(payload.get("planned_trajectory_length_sum", 0))
        metrics.max_planned_trajectory_length = int(payload.get("max_planned_trajectory_length", 0))
        metrics.trajectory_length_sum = int(payload.get("trajectory_length_sum", 0))
        metrics.max_trajectory_length = int(payload.get("max_trajectory_length", 0))
        metrics.planned_trajectory_length_2_plus_count = int(
            payload.get("planned_trajectory_length_2_plus_count", 0)
        )
        metrics.trajectory_length_2_plus_count = int(payload.get("trajectory_length_2_plus_count", 0))
        metrics.trajectory_length_3_plus_count = int(payload.get("trajectory_length_3_plus_count", 0))
        metrics.reached_planned_trajectory_length_count = int(
            payload.get("reached_planned_trajectory_length_count", 0)
        )
        metrics.trajectory_length_1_count = int(payload.get("trajectory_length_1_count", 0))
        metrics.trajectory_length_2_count = int(payload.get("trajectory_length_2_count", 0))
        metrics.trajectory_length_3_plus_rollout_count = int(
            payload.get("trajectory_length_3_plus_rollout_count", 0)
        )
        metrics.invalid_reward_floor_count = int(payload.get("invalid_reward_floor_count", 0))
        stage_accumulators_payload = payload.get("stage_accumulators", {})
        if isinstance(stage_accumulators_payload, dict):
            metrics._stage_accumulators = {
                int(stage_index): _StageAccumulator.from_state_dict(stage_payload)
                for stage_index, stage_payload in stage_accumulators_payload.items()
                if isinstance(stage_payload, dict)
            }
        return metrics

    def merge(self, other: "IncrementalRolloutMetrics") -> "IncrementalRolloutMetrics":
        if (
            self.max_molecules_per_sequence is not None
            and other.max_molecules_per_sequence is not None
            and int(self.max_molecules_per_sequence) != int(other.max_molecules_per_sequence)
        ):
            raise ValueError("Cannot merge rollout metrics with different max_molecules_per_sequence values.")
        if (
            self.invalid_terminal_reward is not None
            and other.invalid_terminal_reward is not None
            and abs(float(self.invalid_terminal_reward) - float(other.invalid_terminal_reward)) > 1.0e-12
        ):
            raise ValueError("Cannot merge rollout metrics with different invalid_terminal_reward values.")

        if self.max_molecules_per_sequence is None:
            self.max_molecules_per_sequence = other.max_molecules_per_sequence
        if self.invalid_terminal_reward is None:
            self.invalid_terminal_reward = other.invalid_terminal_reward

        self.stage_indices = tuple(
            sorted({int(stage_index) for stage_index in self.stage_indices + other.stage_indices})
        )
        self.num_rollouts += int(other.num_rollouts)
        self.total_trajectories += int(other.total_trajectories)
        self.planned_trajectory_length_sum += int(other.planned_trajectory_length_sum)
        self.max_planned_trajectory_length = max(
            int(self.max_planned_trajectory_length),
            int(other.max_planned_trajectory_length),
        )
        self.trajectory_length_sum += int(other.trajectory_length_sum)
        self.max_trajectory_length = max(
            int(self.max_trajectory_length),
            int(other.max_trajectory_length),
        )
        self.planned_trajectory_length_2_plus_count += int(other.planned_trajectory_length_2_plus_count)
        self.trajectory_length_2_plus_count += int(other.trajectory_length_2_plus_count)
        self.trajectory_length_3_plus_count += int(other.trajectory_length_3_plus_count)
        self.reached_planned_trajectory_length_count += int(other.reached_planned_trajectory_length_count)
        self.trajectory_length_1_count += int(other.trajectory_length_1_count)
        self.trajectory_length_2_count += int(other.trajectory_length_2_count)
        self.trajectory_length_3_plus_rollout_count += int(other.trajectory_length_3_plus_rollout_count)
        self.invalid_reward_floor_count += int(other.invalid_reward_floor_count)

        for stage_index, other_accumulator in other._stage_accumulators.items():
            stage_state = self._stage_accumulators.setdefault(
                int(stage_index),
                _StageAccumulator(),
            )
            stage_state.merge(other_accumulator)
        return self
