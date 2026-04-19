from src.constants import EOM_TOKEN

from post_training.gflownet.buffer import OnPolicyBatch, TrajectoryReplayBuffer
from post_training.gflownet.trajectory import SampledStageTrajectory


def _make_sampled(rollout_id: str, *, stage_index: int, num_actions: int) -> SampledStageTrajectory:
    action_token_ids = tuple(range(1, num_actions + 1))
    return SampledStageTrajectory(
        rollout_id=rollout_id,
        example_id=f"example-{rollout_id}",
        prompt_text="prompt",
        description="description",
        target_selfies_list=("[C][C][O]",),
        stage_index=stage_index,
        decoder_prefix_text="",
        previous_valid_selfies=(),
        stage_text="<bom>[C][C][O]<eom>",
        sampled_selfies="[C][C][O]",
        action_token_ids=action_token_ids,
        reward_breakdown={"amplified_reward": 1.0},
        prefix_rewards=tuple([1.0e-4] * num_actions + [1.0]),
        terminal_reward=1.0,
        stop_token=EOM_TOKEN,
        termination_reason="stop_token",
        is_valid=True,
        is_duplicate=False,
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


def test_replay_buffer_evicts_oldest_items_by_token_budget() -> None:
    buffer = TrajectoryReplayBuffer(capacity=3, max_total_action_tokens=4)

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
