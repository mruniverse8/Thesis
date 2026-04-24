import random

from src.constants import EOM_TOKEN

from post_training.gflownet.buffer import (
    OnPolicyBatch,
    TrajectoryReplayBuffer,
    UniformReplayBuffer,
)
from post_training.gflownet.experimental_buffers import (
    ExperimentalMixtureReplayBuffer,
    ExperimentalTBMixtureReplayBuffer,
)
from post_training.gflownet.trajectory import SampledStageTrajectory
from post_training.gflownet.trajectory import ScoredStageTrajectory


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


def test_experimental_mixture_replay_buffer_respects_element_capacity() -> None:
    buffer = ExperimentalMixtureReplayBuffer(capacity=2)
    buffer.extend(
        [
            _make_sampled("one", stage_index=1, num_actions=2, terminal_reward=1.0),
            _make_sampled("two", stage_index=1, num_actions=3, terminal_reward=2.0),
            _make_sampled("three", stage_index=1, num_actions=4, terminal_reward=3.0),
        ]
    )

    assert len(buffer) == 2
    assert buffer.total_action_tokens == 7
    assert [item.rollout_id for item in buffer.snapshot()] == ["two", "three"]


def test_experimental_mixture_replay_buffer_samples_recent_source() -> None:
    buffer = ExperimentalMixtureReplayBuffer(
        capacity=4,
        recent_fraction=1.0,
        reward_fraction=0.0,
        uniform_fraction=0.0,
        recent_window_size=1,
    )
    buffer.extend(
        [
            _make_sampled("old", stage_index=1, num_actions=2),
            _make_sampled("recent", stage_index=1, num_actions=2),
        ]
    )

    sample = buffer.sample(5, rng=random.Random(0), with_replacement=True)

    assert [item.rollout_id for item in sample.trajectories] == ["recent"] * 5
    assert sample.source_counts == {"recent": 5}


def test_experimental_mixture_replay_buffer_samples_reward_source() -> None:
    buffer = ExperimentalMixtureReplayBuffer(
        capacity=4,
        recent_fraction=0.0,
        reward_fraction=1.0,
        uniform_fraction=0.0,
    )
    buffer.extend(
        [
            _make_sampled("low", stage_index=1, num_actions=2, terminal_reward=1.0e-6),
            _make_sampled("high", stage_index=1, num_actions=2, terminal_reward=1000.0),
        ]
    )

    sample = buffer.sample(8, rng=random.Random(0), with_replacement=True)

    assert {item.rollout_id for item in sample.trajectories} == {"high"}
    assert sample.source_counts == {"reward": 8}


def test_experimental_mixture_replay_buffer_samples_uniform_without_replacement() -> None:
    buffer = ExperimentalMixtureReplayBuffer(
        capacity=4,
        recent_fraction=0.0,
        reward_fraction=0.0,
        uniform_fraction=1.0,
    )
    buffer.extend(
        [
            _make_sampled("one", stage_index=1, num_actions=2),
            _make_sampled("two", stage_index=1, num_actions=2),
            _make_sampled("three", stage_index=1, num_actions=2),
        ]
    )

    sample = buffer.sample(2, rng=random.Random(0), with_replacement=False)

    assert len(sample) == 2
    assert len({item.rollout_id for item in sample.trajectories}) == 2
    assert sample.source_counts == {"uniform": 2}


def test_experimental_tb_mixture_replay_buffer_updates_residual_source() -> None:
    buffer = ExperimentalTBMixtureReplayBuffer(
        capacity=4,
        recent_fraction=0.0,
        reward_fraction=0.0,
        uniform_fraction=0.0,
        tb_residual_fraction=1.0,
    )
    low = _make_sampled("low", stage_index=1, num_actions=1, terminal_reward=1.0)
    high = _make_sampled("high", stage_index=1, num_actions=1, terminal_reward=1.0)
    buffer.extend([low, high])
    buffer.observe_scored(
        [
            ScoredStageTrajectory(
                sampled=low,
                log_pf_tokens=(0.0,),
                log_stop=(0.0, 0.0),
                log_state_flows=(0.0, 0.0),
            ),
            ScoredStageTrajectory(
                sampled=high,
                log_pf_tokens=(1000.0,),
                log_stop=(0.0, 0.0),
                log_state_flows=(0.0, 0.0),
            ),
        ]
    )

    sample = buffer.sample(8, rng=random.Random(0), with_replacement=True)

    assert {item.rollout_id for item in sample.trajectories} == {"high"}
    assert sample.source_counts == {"tb_residual": 8}


def test_experimental_mixture_replay_buffer_evicts_invalid_over_quota_first() -> None:
    buffer = ExperimentalMixtureReplayBuffer(
        capacity=3,
        max_invalid_fraction=0.25,
        max_duplicate_fraction=1.0,
    )
    buffer.extend(
        [
            _make_sampled("valid-a", stage_index=1, num_actions=2, terminal_reward=3.0),
            _make_sampled(
                "invalid-a",
                stage_index=1,
                num_actions=2,
                terminal_reward=1.0e-4,
                is_valid=False,
            ),
            _make_sampled(
                "invalid-b",
                stage_index=1,
                num_actions=2,
                terminal_reward=1.0e-4,
                is_valid=False,
            ),
            _make_sampled("valid-b", stage_index=1, num_actions=2, terminal_reward=2.0),
        ]
    )

    assert len(buffer) == 3
    assert [item.rollout_id for item in buffer.snapshot()].count("invalid-a") == 0
    assert {item.rollout_id for item in buffer.snapshot()} == {
        "valid-a",
        "invalid-b",
        "valid-b",
    }


def test_experimental_mixture_replay_buffer_evicts_duplicates_over_quota_first() -> None:
    buffer = ExperimentalMixtureReplayBuffer(
        capacity=3,
        max_invalid_fraction=1.0,
        max_duplicate_fraction=0.25,
    )
    buffer.extend(
        [
            _make_sampled("valid-a", stage_index=1, num_actions=2, terminal_reward=3.0),
            _make_sampled(
                "duplicate-a",
                stage_index=1,
                num_actions=2,
                terminal_reward=1.0,
                is_duplicate=True,
            ),
            _make_sampled(
                "duplicate-b",
                stage_index=1,
                num_actions=2,
                terminal_reward=1.0,
                is_duplicate=True,
            ),
            _make_sampled("valid-b", stage_index=1, num_actions=2, terminal_reward=2.0),
        ]
    )

    assert len(buffer) == 3
    assert [item.rollout_id for item in buffer.snapshot()].count("duplicate-a") == 0
    assert {item.rollout_id for item in buffer.snapshot()} == {
        "valid-a",
        "duplicate-b",
        "valid-b",
    }
