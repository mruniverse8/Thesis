from __future__ import annotations

import pytest

pytest.importorskip("rdkit")

from evaluation_metrics import EvaluationMetricConfig, GenerationGroup, MoleculeInput, evaluate_generation_groups
from notebooks.gflownet_eval_parallel import evaluate_generation_rows, shard_selected_examples
from notebooks.gflownet_eval_streaming import IncrementalRolloutMetrics
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
        candidates=tuple(MoleculeInput(text=selfies, representation="selfies") for selfies in candidates),
        targets=tuple(MoleculeInput(text=selfies, representation="selfies") for selfies in targets),
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


def _build_generation_rows(groups: list[GenerationGroup]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for global_index, group in enumerate(groups):
        rows.append(
            {
                "id": group.group_id,
                "split": "validation",
                "description": f"description-{global_index}",
                "prompt": f"prompt-{global_index}",
                "target_selfies_list": [target.text for target in group.targets],
                "sampled_selfies_list": [candidate.text for candidate in group.candidates],
                "global_example_index": global_index,
                "selected_dataset_index": 100 + global_index,
            }
        )
    return rows


def _build_rollout_batches() -> list[list[SampledStageTrajectory]]:
    return [
        [
            _trajectory(
                rollout_id="rollout-0",
                stage_index=1,
                terminal_reward=2.0,
                action_token_ids=(1, 2, 3),
                sampled_selfies="[C][C][O]",
            ),
            _trajectory(
                rollout_id="rollout-0",
                stage_index=2,
                terminal_reward=3.0,
                action_token_ids=(4, 5),
                sampled_selfies="[C][C][N]",
            ),
        ],
        [
            _trajectory(
                rollout_id="rollout-1",
                stage_index=1,
                terminal_reward=2.5,
                action_token_ids=(1, 2),
                sampled_selfies="[C][C][O]",
            )
        ],
        [
            _trajectory(
                rollout_id="rollout-2",
                stage_index=1,
                terminal_reward=1.5,
                action_token_ids=(1,),
                sampled_selfies="[C][C][N]",
            ),
            _trajectory(
                rollout_id="rollout-2",
                stage_index=2,
                terminal_reward=2.2,
                action_token_ids=(2, 3, 4),
                sampled_selfies="[C][O]",
            ),
            _trajectory(
                rollout_id="rollout-2",
                stage_index=3,
                terminal_reward=4.1,
                termination_reason="max_sequence_length",
                action_token_ids=(5, 6),
                sampled_selfies="[C][C][O]",
            ),
        ],
        [
            _trajectory(
                rollout_id="rollout-3",
                stage_index=1,
                terminal_reward=1.0e-4,
                termination_reason="max_stage_new_tokens",
                is_valid=False,
                action_token_ids=(1, 2, 3, 4),
            )
        ],
    ]


def test_shard_selected_examples_covers_selected_order_exactly_once() -> None:
    examples = [
        {"id": "selected-0"},
        {"id": "selected-1"},
        {"id": "selected-2"},
        {"id": "selected-3"},
        {"id": "selected-4"},
    ]
    selected_indices = [7, 2, 2, 11, 5]

    shard0_examples, shard0_indices, shard0 = shard_selected_examples(
        examples,
        selected_indices,
        shard_index=0,
        num_shards=2,
    )
    shard1_examples, shard1_indices, shard1 = shard_selected_examples(
        examples,
        selected_indices,
        shard_index=1,
        num_shards=2,
    )

    assert shard0.start == 0
    assert shard0.end == 3
    assert shard1.start == 3
    assert shard1.end == 5
    assert [example["id"] for example in shard0_examples + shard1_examples] == [
        "selected-0",
        "selected-1",
        "selected-2",
        "selected-3",
        "selected-4",
    ]
    assert shard0_indices + shard1_indices == selected_indices
    assert shard0.global_example_indices + shard1.global_example_indices == tuple(range(len(examples)))


def test_merged_generation_rows_match_serial_generation_metrics() -> None:
    metric_config = EvaluationMetricConfig(
        acceptance_dice_threshold=0.0,
        compute_n_circles=False,
        n_circles_tanimoto_threshold=0.6,
    )
    groups = [
        _group("group-0", candidates=("[C][C][O]", "[Bad]"), targets=("[C][C][C]",)),
        _group("group-1", candidates=("[C][C][O]", "[C][C][O]"), targets=("[C][C][O]",)),
        _group("group-2", candidates=("[C][C][N]", "[C][O]", "[C][C][O]"), targets=("[C][C][N]", "[C][O]")),
        _group("group-3", candidates=("[O]", "[Bad]"), targets=("[Bad]",)),
        _group("group-4", candidates=(), targets=("[O]",)),
        _group("group-5", candidates=("[C][N]",), targets=("[C][O]",)),
    ]
    rows = _build_generation_rows(groups)
    shard0 = rows[:3]
    shard1 = rows[3:]

    expected = evaluate_generation_groups(groups, config=metric_config)
    actual = evaluate_generation_rows(
        [*shard1, *shard0],
        metric_config=metric_config,
    )

    assert actual.to_dict(include_assessments=False) == expected.to_dict(include_assessments=False)


def test_rollout_state_roundtrip_and_merge_match_serial_metrics() -> None:
    rollout_batches = _build_rollout_batches()
    serial_metrics = IncrementalRolloutMetrics(
        max_molecules_per_sequence=8,
        invalid_terminal_reward=1.0e-4,
    )
    for batch in rollout_batches:
        serial_metrics.update(batch)

    left_metrics = IncrementalRolloutMetrics(
        max_molecules_per_sequence=8,
        invalid_terminal_reward=1.0e-4,
    )
    for batch in rollout_batches[:2]:
        left_metrics.update(batch)

    right_metrics = IncrementalRolloutMetrics(
        max_molecules_per_sequence=8,
        invalid_terminal_reward=1.0e-4,
    )
    for batch in rollout_batches[2:]:
        right_metrics.update(batch)

    roundtrip_metrics = IncrementalRolloutMetrics.from_state_dict(serial_metrics.to_state_dict())
    merged_metrics = IncrementalRolloutMetrics.from_state_dict(left_metrics.to_state_dict())
    merged_metrics.merge(IncrementalRolloutMetrics.from_state_dict(right_metrics.to_state_dict()))

    _compare_metric_dicts(roundtrip_metrics.to_metrics(), serial_metrics.to_metrics())
    _compare_metric_dicts(merged_metrics.to_metrics(), serial_metrics.to_metrics())
