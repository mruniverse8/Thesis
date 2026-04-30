from __future__ import annotations

import pytest

pytest.importorskip("rdkit")

from evaluation_metrics import EvaluationMetricConfig, GenerationGroup, MoleculeInput, evaluate_generation_groups
from notebooks.gflownet_eval_streaming import IncrementalGenerationMetrics, IncrementalRolloutMetrics
from post_training.gflownet.diagnostics import rollout_stage_metrics
from post_training.gflownet.trajectory import SampledStageTrajectory
from src.constants import EOM_TOKEN


def _group(
    group_id: str,
    *,
    candidates: tuple[str, ...],
    targets: tuple[str, ...],
) -> GenerationGroup:
    return GenerationGroup(
        group_id=group_id,
        candidates=tuple(
            MoleculeInput(text=selfies, representation="selfies")
            for selfies in candidates
        ),
        targets=tuple(
            MoleculeInput(text=selfies, representation="selfies")
            for selfies in targets
        ),
    )


def _trajectory(
    *,
    rollout_id: str,
    stage_index: int,
    terminal_reward: float,
    termination_reason: str = "stop_token",
    is_valid: bool = True,
    action_token_ids: tuple[int, ...] = (1, 2),
    sampled_selfies: str | None = "[C][C][O]",
) -> SampledStageTrajectory:
    return SampledStageTrajectory(
        rollout_id=rollout_id,
        example_id=f"example-{rollout_id}",
        prompt_text="prompt",
        description="description",
        target_selfies_list=("[C][C][O]", "[C][C][N]"),
        stage_index=stage_index,
        decoder_prefix_text="",
        previous_sampled_selfies=(),
        stage_text="<bom>[C][C][O]<eom>",
        sampled_selfies=sampled_selfies if is_valid else None,
        action_token_ids=action_token_ids,
        reward_breakdown={"amplified_reward": terminal_reward},
        prefix_rewards=tuple([1.0e-4] * len(action_token_ids) + [terminal_reward]),
        terminal_reward=terminal_reward,
        stop_token=EOM_TOKEN,
        termination_reason=termination_reason,
        is_valid=is_valid,
        is_duplicate=False,
        metadata={},
    )


def _compare_metric_dicts(actual: dict[str, float], expected: dict[str, float]) -> None:
    assert actual.keys() == expected.keys()
    for key in actual:
        assert actual[key] == pytest.approx(expected[key]), key


def _build_groups_and_rollouts() -> tuple[list[GenerationGroup], list[list[SampledStageTrajectory]]]:
    groups: list[GenerationGroup] = []
    rollout_batches: list[list[SampledStageTrajectory]] = []
    for index in range(123):
        group_id = f"group-{index:03d}"
        rollout_id = f"rollout-{index:03d}"
        pattern = index % 6
        if pattern == 0:
            groups.append(
                _group(
                    group_id,
                    candidates=("[C][C][O]", "[Bad]"),
                    targets=("[C][C][C]",),
                )
            )
            rollout_batches.append(
                [
                    _trajectory(
                        rollout_id=rollout_id,
                        stage_index=1,
                        terminal_reward=2.0,
                        action_token_ids=(1, 2, 3),
                        sampled_selfies="[C][C][O]",
                    ),
                    _trajectory(
                        rollout_id=rollout_id,
                        stage_index=2,
                        terminal_reward=3.0,
                        action_token_ids=(4, 5),
                        sampled_selfies="[C][C][N]",
                    ),
                ]
            )
        elif pattern == 1:
            groups.append(
                _group(
                    group_id,
                    candidates=("[C][C][O]", "[C][C][O]"),
                    targets=("[C][C][O]",),
                )
            )
            rollout_batches.append(
                [
                    _trajectory(
                        rollout_id=rollout_id,
                        stage_index=1,
                        terminal_reward=2.5,
                        action_token_ids=(1, 2),
                        sampled_selfies="[C][C][O]",
                    )
                ]
            )
        elif pattern == 2:
            groups.append(
                _group(
                    group_id,
                    candidates=("[C][C][N]", "[C][O]", "[C][C][O]"),
                    targets=("[C][C][N]", "[C][O]"),
                )
            )
            rollout_batches.append(
                [
                    _trajectory(
                        rollout_id=rollout_id,
                        stage_index=1,
                        terminal_reward=1.5,
                        action_token_ids=(1,),
                        sampled_selfies="[C][C][N]",
                    ),
                    _trajectory(
                        rollout_id=rollout_id,
                        stage_index=2,
                        terminal_reward=2.2,
                        action_token_ids=(2, 3, 4),
                        sampled_selfies="[C][O]",
                    ),
                    _trajectory(
                        rollout_id=rollout_id,
                        stage_index=3,
                        terminal_reward=4.1,
                        action_token_ids=(5, 6),
                        termination_reason="max_sequence_length",
                        sampled_selfies="[C][C][O]",
                    ),
                ]
            )
        elif pattern == 3:
            groups.append(
                _group(
                    group_id,
                    candidates=("[O]", "[Bad]"),
                    targets=("[Bad]",),
                )
            )
            rollout_batches.append(
                [
                    _trajectory(
                        rollout_id=rollout_id,
                        stage_index=1,
                        terminal_reward=1.0e-4,
                        termination_reason="max_stage_new_tokens",
                        is_valid=False,
                        action_token_ids=(1, 2, 3, 4),
                    )
                ]
            )
        elif pattern == 4:
            groups.append(
                _group(
                    group_id,
                    candidates=(),
                    targets=("[O]",),
                )
            )
            rollout_batches.append([])
        else:
            groups.append(
                _group(
                    group_id,
                    candidates=("[C][N]",),
                    targets=("[C][O]",),
                )
            )
            rollout_batches.append(
                [
                    _trajectory(
                        rollout_id=rollout_id,
                        stage_index=1,
                        terminal_reward=1.8,
                        action_token_ids=(7, 8),
                        sampled_selfies="[C][N]",
                    ),
                    _trajectory(
                        rollout_id=rollout_id,
                        stage_index=2,
                        terminal_reward=1.0e-4,
                        termination_reason="eos_token",
                        is_valid=False,
                        action_token_ids=(9,),
                    ),
                ]
            )
    return groups, rollout_batches


def test_incremental_generation_and_rollout_metrics_match_reference_snapshots() -> None:
    config = EvaluationMetricConfig(
        acceptance_dice_threshold=0.0,
        compute_n_circles=False,
        n_circles_tanimoto_threshold=0.6,
    )
    generation_metrics = IncrementalGenerationMetrics(config)
    rollout_metrics = IncrementalRolloutMetrics(
        max_molecules_per_sequence=8,
        invalid_terminal_reward=1.0e-4,
    )
    groups, rollout_batches = _build_groups_and_rollouts()
    all_trajectories: list[SampledStageTrajectory] = []

    for index, (group, trajectories) in enumerate(zip(groups, rollout_batches), start=1):
        generation_metrics.update(group)
        rollout_metrics.update(trajectories)
        all_trajectories.extend(trajectories)

        if index not in {50, 100, len(groups)}:
            continue

        expected_generation = evaluate_generation_groups(groups[:index], config=config)
        actual_generation = generation_metrics.to_result()
        assert actual_generation.to_dict(include_assessments=False) == expected_generation.to_dict(
            include_assessments=False
        )

        expected_rollout = rollout_stage_metrics(
            all_trajectories,
            max_molecules_per_sequence=8,
            invalid_terminal_reward=1.0e-4,
        )
        actual_rollout = rollout_metrics.to_metrics()
        _compare_metric_dicts(actual_rollout, expected_rollout)


def test_incremental_generation_metrics_removes_novelty_when_target_appears_later() -> None:
    config = EvaluationMetricConfig(
        acceptance_dice_threshold=0.0,
        compute_n_circles=False,
        n_circles_tanimoto_threshold=0.6,
    )
    groups = [
        _group(
            "group-early",
            candidates=("[C][C][O]",),
            targets=("[C][C][C]",),
        ),
        _group(
            "group-late",
            candidates=(),
            targets=("[C][C][O]",),
        ),
    ]
    metrics = IncrementalGenerationMetrics(config)

    metrics.update(groups[0])
    first_result = metrics.to_result()
    assert first_result.novelty_count == 1
    assert first_result.novelty_fraction == pytest.approx(1.0)

    metrics.update(groups[1])
    second_result = metrics.to_result()
    expected = evaluate_generation_groups(groups, config=config)
    assert second_result.to_dict(include_assessments=False) == expected.to_dict(
        include_assessments=False
    )
    assert second_result.novelty_count == 0
    assert second_result.novelty_fraction == pytest.approx(0.0)
