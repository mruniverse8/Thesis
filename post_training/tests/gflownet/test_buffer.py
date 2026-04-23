import random

from src.constants import EOM_TOKEN

from post_training.gflownet.buffer import (
    OnPolicyBatch,
    PriorityReplayBuffer,
    TrajectoryReplayBuffer,
    UniformReplayBuffer,
)
from post_training.gflownet.trajectory import SampledStageTrajectory


def _make_sampled(
    rollout_id: str,
    *,
    stage_index: int,
    num_actions: int,
    terminal_reward: float = 1.0,
    is_valid: bool = True,
    is_duplicate: bool = False,
) -> SampledStageTrajectory:
    action_token_ids = tuple(range(1, num_actions + 1))
    return SampledStageTrajectory(
        rollout_id=rollout_id,
        example_id=f"example-{rollout_id}",
        prompt_text="prompt",
        description="description",
        target_selfies_list=("[C][C][O]",),
        stage_index=stage_index,
        decoder_prefix_text="",
        previous_sampled_selfies=(),
        stage_text="<bom>[C][C][O]<eom>",
        sampled_selfies="[C][C][O]" if is_valid else None,
        action_token_ids=action_token_ids,
        reward_breakdown={"amplified_reward": terminal_reward},
        prefix_rewards=tuple([1.0e-4] * num_actions + [max(terminal_reward, 1.0e-4)]),
        terminal_reward=max(terminal_reward, 1.0e-4),
        stop_token=EOM_TOKEN,
        termination_reason="stop_token",
        is_valid=is_valid,
        is_duplicate=is_duplicate,
    )


def test_on_policy_batch_reports_stage_local_aggregates() -> None:
    batch = OnPolicyBatch.from_trajectories(
        [
            _make_sampled("one", stage_index=1, num_actions=2),
            _make_sampled("two", stage_index=3, num_actions=4),
        ]
    )

    assert batch.mean_stage_reward() == 1.0
    assert batch.mean_num_actions() == 3.0
    assert batch.mean_stage_index() == 2.0


def test_uniform_replay_buffer_evicts_oldest_items_by_token_budget() -> None:
    buffer = UniformReplayBuffer(capacity=3, max_total_action_tokens=4)

    first = _make_sampled("one", stage_index=1, num_actions=2)
    second = _make_sampled("two", stage_index=2, num_actions=2)
    third = _make_sampled("three", stage_index=3, num_actions=3)

    buffer.add(first)
    buffer.add(second)
    assert buffer.total_action_tokens == 4

    buffer.add(third)

    assert len(buffer) == 1
    assert buffer.total_action_tokens == 3
    assert [item.rollout_id for item in buffer.snapshot()] == ["three"]


def test_trajectory_replay_buffer_alias_points_to_uniform_buffer() -> None:
    buffer = TrajectoryReplayBuffer(capacity=2, max_total_action_tokens=8)

    assert isinstance(buffer, UniformReplayBuffer)


def test_priority_replay_buffer_classifies_trajectories_into_expected_buckets() -> None:
    buffer = PriorityReplayBuffer(
        capacity=8,
        max_total_action_tokens=64,
        invalid_terminal_reward=1.0e-4,
        top_reward_fraction=0.5,
        hard_positive_fraction=0.25,
        hard_negative_fraction=0.25,
    )
    trajectories = [
        _make_sampled("top-a", stage_index=1, num_actions=2, terminal_reward=5.0),
        _make_sampled("top-b", stage_index=1, num_actions=2, terminal_reward=4.0),
        _make_sampled("hard-pos", stage_index=1, num_actions=2, terminal_reward=2.0),
        _make_sampled(
            "hard-neg-invalid",
            stage_index=1,
            num_actions=2,
            terminal_reward=1.0e-4,
            is_valid=False,
        ),
        _make_sampled(
            "hard-neg-dup",
            stage_index=1,
            num_actions=2,
            terminal_reward=3.0,
            is_duplicate=True,
        ),
    ]
    buffer.extend(trajectories)

    sample = buffer.sample(5, rng=random.Random(0), with_replacement=False)

    assert len(sample) == 5
    assert sample.top_reward_count == 2
    assert sample.hard_positive_count == 1
    assert sample.hard_negative_count == 2


def test_priority_replay_buffer_respects_bucket_mix_when_buckets_are_populated() -> None:
    buffer = PriorityReplayBuffer(
        capacity=8,
        max_total_action_tokens=64,
        invalid_terminal_reward=1.0e-4,
        top_reward_fraction=0.5,
        hard_positive_fraction=0.25,
        hard_negative_fraction=0.25,
    )
    buffer.extend(
        [
            _make_sampled("top-a", stage_index=1, num_actions=2, terminal_reward=5.0),
            _make_sampled("top-b", stage_index=1, num_actions=2, terminal_reward=4.5),
            _make_sampled("top-c", stage_index=1, num_actions=2, terminal_reward=4.0),
            _make_sampled("hard-pos-a", stage_index=1, num_actions=2, terminal_reward=2.0),
            _make_sampled("hard-pos-b", stage_index=1, num_actions=2, terminal_reward=1.5),
            _make_sampled(
                "hard-neg-a",
                stage_index=1,
                num_actions=2,
                terminal_reward=1.0e-4,
                is_valid=False,
            ),
            _make_sampled(
                "hard-neg-b",
                stage_index=1,
                num_actions=2,
                terminal_reward=2.5,
                is_duplicate=True,
            ),
        ]
    )

    sample = buffer.sample(4, rng=random.Random(0), with_replacement=False)

    assert len(sample) == 4
    assert sample.top_reward_count == 2
    assert sample.hard_positive_count == 1
    assert sample.hard_negative_count == 1


def test_priority_replay_buffer_backfills_from_remaining_buckets_when_a_bucket_is_sparse() -> None:
    buffer = PriorityReplayBuffer(
        capacity=8,
        max_total_action_tokens=64,
        invalid_terminal_reward=1.0e-4,
        top_reward_fraction=0.5,
        hard_positive_fraction=0.25,
        hard_negative_fraction=0.25,
    )
    buffer.extend(
        [
            _make_sampled("top-a", stage_index=1, num_actions=2, terminal_reward=5.0),
            _make_sampled("top-b", stage_index=1, num_actions=2, terminal_reward=4.0),
            _make_sampled("hard-pos-a", stage_index=1, num_actions=2, terminal_reward=2.0),
            _make_sampled("hard-pos-b", stage_index=1, num_actions=2, terminal_reward=1.5),
        ]
    )

    sample = buffer.sample(4, rng=random.Random(0), with_replacement=False)

    assert len(sample) == 4
    assert sample.top_reward_count == 2
    assert sample.hard_positive_count == 2
    assert sample.hard_negative_count == 0
