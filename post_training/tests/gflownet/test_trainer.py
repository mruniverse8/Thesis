import json
from pathlib import Path
import random
from types import SimpleNamespace
from zipfile import ZipFile

import pytest
import torch

from src.constants import EOM_TOKEN

from post_training.gflownet.config import (
    GFlowNetConfig,
    GFlowNetRolloutConfig,
    ReplayConfig,
    TargetGuidanceConfig,
)
from post_training.gflownet.diagnostics import GFlowNetTrainIterationResult
from post_training.gflownet.trajectory import SampledStageTrajectory, ScoredStageTrajectory
from post_training.gflownet.trainer import (
    EpochBatchSampler,
    MultiMoleculeGFlowNetTrainer,
    _compute_replay_target_count,
    _select_optimization_trajectories,
    run_multi_molecule_gflownet,
)
from post_training.shared.decoding import StageTokenConstraints
from post_training.shared.config import DEFAULT_PPO_FALLBACK_CHECKPOINT, resolve_gflownet_config_paths


def _make_sampled_trajectory(
    *,
    rollout_id: str,
    terminal_reward: float,
    prompt_text: str = "prompt",
    stage_index: int = 1,
    action_token_ids: tuple[int, ...] = (1, 2),
    target_selfies_list: tuple[str, ...] = ("[C][C][O]",),
    termination_reason: str = "stop_token",
    is_valid: bool = True,
    sampled_selfies: str | None = "[C][C][O]",
    is_duplicate: bool = False,
    metadata_source: str | None = None,
) -> SampledStageTrajectory:
    return SampledStageTrajectory(
        rollout_id=rollout_id,
        example_id=f"example-{rollout_id}",
        prompt_text=prompt_text,
        description="description",
        target_selfies_list=target_selfies_list,
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
        is_duplicate=is_duplicate,
        metadata=(
            {"trajectory_source": metadata_source}
            if metadata_source is not None
            else {}
        ),
    )


class _BatchScoringTokenizer:
    def __init__(self) -> None:
        self.batch_calls: list[dict[str, object]] = []

    def __call__(
        self,
        text,
        *,
        padding=False,
        truncation=False,
        max_length=None,
        return_tensors=None,
        add_special_tokens=True,
    ):
        del add_special_tokens
        if isinstance(text, list):
            self.batch_calls.append(
                {
                    "texts": tuple(str(value) for value in text),
                    "padding": padding,
                    "truncation": truncation,
                    "max_length": max_length,
                    "return_tensors": return_tensors,
                }
            )
            encoded = [self._encode_text(str(value)) for value in text]
            max_encoded_length = max(len(token_ids) for token_ids in encoded)
            padded = [
                token_ids + [0] * (max_encoded_length - len(token_ids))
                for token_ids in encoded
            ]
            masks = [
                [1] * len(token_ids) + [0] * (max_encoded_length - len(token_ids))
                for token_ids in encoded
            ]
            return {
                "input_ids": torch.tensor(padded, dtype=torch.long),
                "attention_mask": torch.tensor(masks, dtype=torch.long),
            }

        token_ids = self._encode_text(str(text))
        return {
            "input_ids": torch.tensor([token_ids], dtype=torch.long),
            "attention_mask": torch.ones((1, len(token_ids)), dtype=torch.long),
        }

    def _encode_text(self, text: str) -> list[int]:
        token_ids = [2]
        token_ids.extend(3 + (ord(character) % 17) for character in text[:6])
        return token_ids

    def convert_tokens_to_ids(self, token: str) -> int:
        if token == EOM_TOKEN:
            return 7
        return 0


class _RecordingScoreModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(1.0))
        self.policy_model = SimpleNamespace(config=SimpleNamespace(decoder_start_token_id=0))
        self.calls: list[tuple[tuple[int, ...], ...]] = []

    def score_action_sequences(
        self,
        *,
        input_ids,
        attention_mask,
        decoder_prefix_ids,
        action_token_ids,
        stop_token_id,
    ):
        del input_ids, attention_mask, decoder_prefix_ids, stop_token_id
        normalized_action_ids = tuple(
            tuple(int(token_id) for token_id in row_action_ids)
            for row_action_ids in action_token_ids
        )
        self.calls.append(normalized_action_ids)
        scores = []
        for row_action_ids in normalized_action_ids:
            anchor = float(row_action_ids[0] if row_action_ids else 1)
            base = self.weight * (anchor / 100.0)
            scores.append(
                (
                    tuple(base + 0.01 * index for index, _ in enumerate(row_action_ids)),
                    tuple(base - 0.02 * index for index in range(len(row_action_ids) + 1)),
                    tuple(base + 0.03 * index for index in range(len(row_action_ids) + 1)),
                )
            )
        return scores


def test_replay_fraction_converts_to_replay_dominant_target_count() -> None:
    assert _compute_replay_target_count(
        on_policy_count=4,
        replay_fraction=0.75,
        legacy_replay_batch_size=0,
    ) == 12


def test_epoch_batch_sampler_covers_epoch_before_repeating_and_allows_partial_batch() -> None:
    dataset = [{"id": f"example-{index}"} for index in range(5)]
    sampler = EpochBatchSampler(dataset, batch_size=2, rng=random.Random(7))

    batches = [sampler.next_batch() for _ in range(3)]
    first_epoch_ids = [
        example["id"]
        for batch, _epoch in batches
        for example in batch
    ]

    assert [epoch for _batch, epoch in batches] == [1, 1, 1]
    assert [len(batch) for batch, _epoch in batches] == [2, 2, 1]
    assert set(first_epoch_ids) == {f"example-{index}" for index in range(5)}
    assert len(first_epoch_ids) == len(set(first_epoch_ids))

    next_epoch_batch, next_epoch = sampler.next_batch()
    assert next_epoch == 2
    assert len(next_epoch_batch) == 2


def test_epoch_batch_sampler_reshuffles_at_epoch_boundary() -> None:
    class RecordingRng:
        def __init__(self) -> None:
            self.calls = 0

        def shuffle(self, values: list[int]) -> None:
            self.calls += 1
            if self.calls == 1:
                values.reverse()
            else:
                values.sort()

    dataset = [{"id": f"example-{index}"} for index in range(4)]
    rng = RecordingRng()
    sampler = EpochBatchSampler(dataset, batch_size=4, rng=rng)

    first_epoch_batch, first_epoch = sampler.next_batch()
    second_epoch_batch, second_epoch = sampler.next_batch()

    assert first_epoch == 1
    assert second_epoch == 2
    assert rng.calls == 2
    assert [example["id"] for example in first_epoch_batch] == [
        "example-3",
        "example-2",
        "example-1",
        "example-0",
    ]
    assert [example["id"] for example in second_epoch_batch] == [
        "example-0",
        "example-1",
        "example-2",
        "example-3",
    ]


def test_epoch_batch_sampler_is_seeded_deterministic() -> None:
    dataset = [{"id": f"example-{index}"} for index in range(7)]

    def collect(seed: int) -> list[tuple[tuple[str, ...], int]]:
        sampler = EpochBatchSampler(dataset, batch_size=3, rng=random.Random(seed))
        return [
            (tuple(str(example["id"]) for example in batch), epoch)
            for batch, epoch in (sampler.next_batch() for _ in range(6))
        ]

    assert collect(123) == collect(123)


def test_select_optimization_trajectories_honors_priority_and_optional_cap() -> None:
    teacher = [
        _make_sampled_trajectory(rollout_id=f"teacher-{index}", terminal_reward=10.0)
        for index in range(2)
    ]
    on_policy = [
        _make_sampled_trajectory(rollout_id=f"on-policy-{index}", terminal_reward=2.0)
        for index in range(2)
    ]
    prefix = [
        _make_sampled_trajectory(rollout_id=f"prefix-{index}", terminal_reward=5.0)
        for index in range(2)
    ]
    replay = [
        _make_sampled_trajectory(rollout_id=f"replay-{index}", terminal_reward=1.0)
        for index in range(2)
    ]

    candidates, uncapped = _select_optimization_trajectories(
        target_teacher_trajectories=teacher,
        on_policy_trajectories=on_policy,
        target_prefix_trajectories=prefix,
        replay_trajectories=replay,
        max_optimization_trajectories_per_iter=None,
    )
    capped_candidates, capped = _select_optimization_trajectories(
        target_teacher_trajectories=teacher,
        on_policy_trajectories=on_policy,
        target_prefix_trajectories=prefix,
        replay_trajectories=replay,
        max_optimization_trajectories_per_iter=5,
    )

    expected_priority = [
        "teacher-0",
        "teacher-1",
        "on-policy-0",
        "on-policy-1",
        "prefix-0",
        "prefix-1",
        "replay-0",
        "replay-1",
    ]
    assert [trajectory.rollout_id for trajectory in candidates] == expected_priority
    assert [trajectory.rollout_id for trajectory in uncapped] == expected_priority
    assert [trajectory.rollout_id for trajectory in capped_candidates] == expected_priority
    assert [trajectory.rollout_id for trajectory in capped] == expected_priority[:5]


def test_score_trajectories_preserves_order_across_scoring_microbatches() -> None:
    tokenizer = _BatchScoringTokenizer()
    model = _RecordingScoreModel()
    trainer = MultiMoleculeGFlowNetTrainer(
        model=model,
        tokenizer=tokenizer,
        config=GFlowNetConfig(
            scoring_microbatch_size=2,
            rollout=GFlowNetRolloutConfig(max_source_length=12),
        ),
        device=torch.device("cpu"),
    )
    trajectories = [
        _make_sampled_trajectory(
            rollout_id=f"trajectory-{index}",
            prompt_text=f"prompt-{index}",
            terminal_reward=1.0 + index,
            action_token_ids=tuple(range(10 + index, 11 + index + (index % 3))),
        )
        for index in range(5)
    ]

    scored = trainer.score_trajectories(trajectories)

    assert [trajectory.sampled.rollout_id for trajectory in scored] == [
        trajectory.rollout_id for trajectory in trajectories
    ]
    assert [len(call) for call in model.calls] == [2, 2, 1]
    assert model.calls[0] == (
        trajectories[0].action_token_ids,
        trajectories[1].action_token_ids,
    )
    assert tokenizer.batch_calls[0]["padding"] is True
    assert tokenizer.batch_calls[0]["truncation"] is True
    assert tokenizer.batch_calls[0]["max_length"] == 12
    assert tokenizer.batch_calls[0]["return_tensors"] == "pt"


@pytest.mark.parametrize("objective", ["tb", "db", "subtb"])
def test_objective_loss_is_scoring_microbatch_invariant(objective: str) -> None:
    trajectories = [
        _make_sampled_trajectory(
            rollout_id=f"trajectory-{index}",
            terminal_reward=1.0 + 0.25 * index,
            action_token_ids=tuple(range(10 + index, 12 + index + (index % 2))),
        )
        for index in range(4)
    ]
    losses: list[float] = []

    for microbatch_size in (1, 2, 4):
        trainer = MultiMoleculeGFlowNetTrainer(
            model=_RecordingScoreModel(),
            tokenizer=_BatchScoringTokenizer(),
            config=GFlowNetConfig(
                objective=objective,
                scoring_microbatch_size=microbatch_size,
            ),
            device=torch.device("cpu"),
        )
        scored = trainer.score_trajectories(trajectories)
        loss, _diagnostics = trainer._compute_objective_loss(scored)
        losses.append(float(loss.detach().item()))

    assert losses[1] == pytest.approx(losses[0], abs=1.0e-7)
    assert losses[2] == pytest.approx(losses[0], abs=1.0e-7)


def test_scheduled_learning_rate_ramps_over_configured_warmup_iterations() -> None:
    trainer = MultiMoleculeGFlowNetTrainer(
        model=_RecordingScoreModel(),
        tokenizer=_BatchScoringTokenizer(),
        config=GFlowNetConfig(
            gflownet_iterations=100,
            learning_rate=1.0e-6,
            warmup_ratio=0.03,
        ),
        device=torch.device("cpu"),
    )

    assert trainer._scheduled_learning_rate(1) == pytest.approx(1.0e-6 / 3.0)
    assert trainer._scheduled_learning_rate(2) == pytest.approx(2.0e-6 / 3.0)
    assert trainer._scheduled_learning_rate(3) == pytest.approx(1.0e-6)
    assert trainer._scheduled_learning_rate(4) == pytest.approx(1.0e-6)


def test_scheduled_learning_rate_stays_constant_without_warmup() -> None:
    trainer = MultiMoleculeGFlowNetTrainer(
        model=_RecordingScoreModel(),
        tokenizer=_BatchScoringTokenizer(),
        config=GFlowNetConfig(
            gflownet_iterations=100,
            learning_rate=1.0e-6,
            warmup_ratio=0.0,
        ),
        device=torch.device("cpu"),
    )

    assert trainer._scheduled_learning_rate(1) == pytest.approx(1.0e-6)
    assert trainer._scheduled_learning_rate(50) == pytest.approx(1.0e-6)


def test_scheduled_learning_rate_uses_absolute_iteration_for_restart_warmup() -> None:
    trainer = MultiMoleculeGFlowNetTrainer(
        model=_RecordingScoreModel(),
        tokenizer=_BatchScoringTokenizer(),
        config=GFlowNetConfig(
            start_iteration=1500,
            gflownet_iterations=100,
            learning_rate=1.0e-6,
            warmup_ratio=0.03,
        ),
        device=torch.device("cpu"),
    )

    assert trainer._scheduled_learning_rate(1501) == pytest.approx(1.0e-6)


def test_train_iteration_reports_current_scheduled_learning_rate(monkeypatch) -> None:
    trainer = MultiMoleculeGFlowNetTrainer(
        model=_RecordingScoreModel(),
        tokenizer=_BatchScoringTokenizer(),
        config=GFlowNetConfig(
            gflownet_iterations=100,
            learning_rate=9.0e-6,
            warmup_ratio=0.03,
            replay=ReplayConfig(enabled=False),
            target_guidance=TargetGuidanceConfig(enabled=False),
        ),
        device=torch.device("cpu"),
    )
    monkeypatch.setattr(
        trainer,
        "collect_on_policy_trajectories",
        lambda _examples, *, iteration_index: [],
    )

    warmup_result = trainer.train_iteration([{"id": "example-1"}], iteration_index=2)
    post_warmup_result = trainer.train_iteration([{"id": "example-1"}], iteration_index=4)

    assert warmup_result.metrics["learning_rate"] == pytest.approx(6.0e-6)
    assert post_warmup_result.metrics["learning_rate"] == pytest.approx(9.0e-6)
    assert trainer.optimizer.param_groups[0]["lr"] == pytest.approx(9.0e-6)


def test_train_iteration_saves_final_additional_iteration_with_absolute_index(
    monkeypatch,
) -> None:
    trainer = MultiMoleculeGFlowNetTrainer(
        model=_RecordingScoreModel(),
        tokenizer=_BatchScoringTokenizer(),
        config=GFlowNetConfig(
            start_iteration=1500,
            gflownet_iterations=2,
            learning_rate=1.0e-6,
            warmup_ratio=0.03,
            save_every_iterations=999,
            replay=ReplayConfig(enabled=False),
            target_guidance=TargetGuidanceConfig(enabled=False),
        ),
        device=torch.device("cpu"),
    )
    trajectory = _make_sampled_trajectory(
        rollout_id="restart-trajectory",
        terminal_reward=2.0,
    )

    monkeypatch.setattr(
        trainer,
        "collect_on_policy_trajectories",
        lambda _examples, *, iteration_index: [trajectory],
    )

    def fake_score(trajectories):
        base = trainer.model.weight
        return [
            ScoredStageTrajectory(
                sampled=item,
                log_pf_tokens=tuple(base * 0.1 for _ in item.action_token_ids),
                log_stop=tuple(base * -0.2 for _ in range(len(item.action_token_ids) + 1)),
                log_state_flows=tuple(
                    base * 0.3 for _ in range(len(item.action_token_ids) + 1)
                ),
            )
            for item in trajectories
        ]

    saved_iterations: list[tuple[int, float]] = []
    monkeypatch.setattr(trainer, "score_trajectories", fake_score)
    monkeypatch.setattr(
        trainer,
        "save_checkpoint",
        lambda **kwargs: saved_iterations.append(
            (
                int(kwargs["iteration_index"]),
                float(kwargs["metrics"]["iteration"]),
            )
        ),
    )
    monkeypatch.setattr(trainer, "save_best_checkpoint", lambda **kwargs: None)

    first_result = trainer.train_iteration([{"id": "example-1"}], iteration_index=1501)
    final_result = trainer.train_iteration([{"id": "example-1"}], iteration_index=1502)

    assert first_result.metrics["iteration"] == pytest.approx(1501.0)
    assert final_result.metrics["iteration"] == pytest.approx(1502.0)
    assert final_result.metrics["learning_rate"] == pytest.approx(1.0e-6)
    assert saved_iterations == [(1502, 1502.0)]


@pytest.mark.parametrize(
    ("objective", "expected_return_last_valid_trajectory_only"),
    [
        ("tb", False),
        ("db", False),
        ("subtb", True),
    ],
)
def test_collect_on_policy_trajectories_passes_last_valid_only_flag_for_subtb(
    monkeypatch,
    objective: str,
    expected_return_last_valid_trajectory_only: bool,
) -> None:
    class DummyModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(1.0))

    calls: list[bool] = []

    def fake_sample_stage_trajectories_for_example(*args, **kwargs):
        del args
        calls.append(bool(kwargs["return_last_valid_trajectory_only"]))
        return []

    monkeypatch.setattr(
        "post_training.gflownet.trainer.sample_stage_trajectories_for_example",
        fake_sample_stage_trajectories_for_example,
    )

    trainer = MultiMoleculeGFlowNetTrainer(
        model=DummyModel(),
        tokenizer=None,
        config=GFlowNetConfig(
            objective=objective,
            replay=ReplayConfig(enabled=False),
        ),
        device=torch.device("cpu"),
    )

    trajectories = trainer.collect_on_policy_trajectories(
        [{"id": "example-1"}],
        iteration_index=1,
    )

    assert trajectories == []
    assert calls == [expected_return_last_valid_trajectory_only]


def test_collect_on_policy_trajectories_stops_after_retained_cap(monkeypatch) -> None:
    class DummyModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(1.0))

    sampled_example_ids: list[str] = []

    def fake_sample_stage_trajectories_for_example(*args, **kwargs):
        del args
        sampled_example_ids.append(str(kwargs["rollout_id"]))
        return [
            _make_sampled_trajectory(
                rollout_id=f"{kwargs['rollout_id']}-stage-{stage_index}",
                terminal_reward=1.0,
                stage_index=stage_index,
            )
            for stage_index in (1, 2)
        ]

    monkeypatch.setattr(
        "post_training.gflownet.trainer.sample_stage_trajectories_for_example",
        fake_sample_stage_trajectories_for_example,
    )

    trainer = MultiMoleculeGFlowNetTrainer(
        model=DummyModel(),
        tokenizer=None,
        config=GFlowNetConfig(
            rollout=GFlowNetRolloutConfig(max_molecules_per_sequence=3),
            replay=ReplayConfig(enabled=False),
        ),
        device=torch.device("cpu"),
    )

    trajectories = trainer.collect_on_policy_trajectories(
        [{"id": "a"}, {"id": "b"}, {"id": "c"}],
        iteration_index=1,
    )

    assert len(trajectories) == 3
    assert sampled_example_ids == [
        "iter-0001-sample-0000-a",
        "iter-0001-sample-0001-b",
    ]


def test_train_iteration_mixes_on_policy_and_target_guided_sources(monkeypatch) -> None:
    class DummyModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(1.0))
            self.policy_model = SimpleNamespace(config=SimpleNamespace(decoder_start_token_id=0))

        def to(self, device):
            return super().to(device)

        def save_checkpoint(self, *args, **kwargs):
            del args, kwargs
            return None

    model = DummyModel()
    trainer = MultiMoleculeGFlowNetTrainer(
        model=model,
        tokenizer=None,
        config=GFlowNetConfig(
            batch_size=3,
            objective="tb",
            save_every_iterations=99,
            diagnostic_log_every_iterations=1,
            trajectory_preview_every_iterations=1,
            trajectory_preview_num_samples=3,
            replay=ReplayConfig(
                enabled=True,
                capacity=32,
                replay_fraction=0.75,
                max_total_action_tokens=32,
            ),
            target_guidance=TargetGuidanceConfig(
                enabled=True,
                on_policy_fraction=0.25,
                target_prefix_rollout_fraction=0.50,
                target_teacher_fraction=0.25,
            ),
        ),
        device=torch.device("cpu"),
    )

    on_policy = [
        _make_sampled_trajectory(
            rollout_id="fresh-1",
            terminal_reward=2.0,
            target_selfies_list=("[C][C][O]", "[C][C][N]"),
        ),
        _make_sampled_trajectory(
            rollout_id="fresh-1",
            terminal_reward=3.0,
            stage_index=2,
            target_selfies_list=("[C][C][O]", "[C][C][N]"),
        ),
        _make_sampled_trajectory(
            rollout_id="fresh-2",
            terminal_reward=1.0e-4,
            action_token_ids=(1, 2, 3, 4),
            target_selfies_list=("[C][C][O]", "[C][C][N]", "[C][O][O]"),
            termination_reason="max_stage_new_tokens",
            is_valid=False,
        ),
        _make_sampled_trajectory(rollout_id="fresh-3", terminal_reward=4.0),
    ]
    assert trainer.replay_buffer is None

    monkeypatch.setattr(
        trainer,
        "collect_on_policy_trajectories",
        lambda _examples, iteration_index: on_policy,
    )
    target_prefix_items = [
        _make_sampled_trajectory(
            rollout_id=f"target-prefix-{index}",
            terminal_reward=5.0,
            metadata_source="target_prefix_rollout",
        )
        for index in range(8)
    ]
    target_teacher_items = [
        _make_sampled_trajectory(
            rollout_id=f"target-teacher-{index}",
            terminal_reward=6.0,
            metadata_source="target_teacher",
        )
        for index in range(4)
    ]
    prefix_calls: list[int] = []
    teacher_calls: list[int] = []

    def fake_collect_target_prefix(_examples, *, iteration_index, count):
        del iteration_index
        prefix_calls.append(int(count))
        return target_prefix_items[:count]

    def fake_collect_target_teacher(_examples, *, iteration_index, count):
        del iteration_index
        teacher_calls.append(int(count))
        return target_teacher_items[:count]

    monkeypatch.setattr(
        trainer,
        "collect_target_prefix_trajectories",
        fake_collect_target_prefix,
    )
    monkeypatch.setattr(
        trainer,
        "collect_target_teacher_trajectories",
        fake_collect_target_teacher,
    )

    def fake_score(trajectories):
        scored: list[ScoredStageTrajectory] = []
        for index, trajectory in enumerate(trajectories, start=1):
            base = trainer.model.weight
            scored.append(
                ScoredStageTrajectory(
                    sampled=trajectory,
                    log_pf_tokens=tuple(
                        base * (0.1 * index + 0.05 * position)
                        for position, _token_id in enumerate(trajectory.action_token_ids)
                    ),
                    log_stop=tuple(
                        base * (-0.2 * index - 0.05 * position)
                        for position in range(len(trajectory.action_token_ids) + 1)
                    ),
                    log_state_flows=tuple(
                        base * (0.3 * index + 0.05 * position)
                        for position in range(len(trajectory.action_token_ids) + 1)
                    ),
                )
            )
        return scored

    monkeypatch.setattr(trainer, "score_trajectories", fake_score)

    result = trainer.train_iteration([{"id": "unused"}], iteration_index=1)
    metrics = result.metrics

    assert metrics["num_on_policy_trajectories"] == 4.0
    assert metrics["num_target_prefix_trajectories"] == 8.0
    assert metrics["num_target_teacher_trajectories"] == 4.0
    assert metrics["num_target_guided_trajectories"] == 12.0
    assert prefix_calls == [8]
    assert teacher_calls == [4]
    assert metrics["target_guidance_on_policy_fraction"] == pytest.approx(0.25)
    assert metrics["target_guidance_prefix_fraction"] == pytest.approx(0.50)
    assert metrics["target_guidance_teacher_fraction"] == pytest.approx(0.25)
    assert metrics["target_prefix_valid_fraction"] == pytest.approx(1.0)
    assert metrics["target_teacher_valid_fraction"] == pytest.approx(1.0)
    assert metrics["target_prefix_mean_stage_reward"] == pytest.approx(5.0)
    assert metrics["target_teacher_mean_stage_reward"] == pytest.approx(6.0)
    assert metrics["num_replay_trajectories"] == 0.0
    assert metrics["configured_replay_fraction"] == pytest.approx(0.0)
    assert metrics["replay_fraction"] == pytest.approx(0.0)
    assert metrics["replay_buffer_type"] == "disabled"
    assert metrics["replay_recent_count"] == pytest.approx(0.0)
    assert metrics["replay_reward_count"] == pytest.approx(0.0)
    assert metrics["replay_uniform_count"] == pytest.approx(0.0)
    assert metrics["replay_tb_residual_count"] == pytest.approx(0.0)
    assert metrics["replay_size"] == 0.0
    assert metrics["replay_total_action_tokens"] == 0.0
    assert metrics["rollout_append_probability"] == pytest.approx(0.30)
    assert metrics["rollout_return_last_valid_trajectory_only"] == pytest.approx(0.0)
    assert metrics["mean_stage_reward"] == pytest.approx((2.0 + 3.0 + 1.0e-4 + 4.0) / 4.0)
    assert metrics["mean_training_stage_reward"] == pytest.approx(
        (2.0 + 3.0 + 1.0e-4 + 4.0 + 8 * 5.0 + 4 * 6.0) / 16.0
    )
    assert metrics["valid_fraction"] == pytest.approx(0.75)
    assert metrics["duplicate_count_on_policy"] == pytest.approx(0.0)
    assert metrics["num_valid_on_policy_for_novelty"] == pytest.approx(3.0)
    assert metrics["num_novel_on_policy"] == pytest.approx(0.0)
    assert metrics["novelty_fraction_on_policy"] == pytest.approx(0.0)
    assert metrics["mean_num_actions"] == pytest.approx(2.5)
    assert metrics["max_num_actions"] == pytest.approx(4.0)
    assert metrics["mean_stage_index"] == pytest.approx(1.25)
    assert metrics["termination_fraction_stop_token"] == pytest.approx(0.75)
    assert metrics["termination_fraction_max_stage_new_tokens"] == pytest.approx(0.25)
    assert metrics["num_rollouts"] == pytest.approx(3.0)
    assert metrics["mean_planned_trajectory_length"] == pytest.approx(8.0)
    assert metrics["max_planned_trajectory_length"] == pytest.approx(8.0)
    assert metrics["mean_trajectory_length"] == pytest.approx(4.0 / 3.0)
    assert metrics["max_trajectory_length"] == pytest.approx(2.0)
    assert metrics["fraction_rollouts_planned_trajectory_length_2_plus"] == pytest.approx(1.0)
    assert metrics["fraction_rollouts_trajectory_length_2_plus"] == pytest.approx(1.0 / 3.0)
    assert metrics["fraction_rollouts_reaching_planned_trajectory_length"] == pytest.approx(0.0)
    assert metrics["trajectory_length_1_fraction"] == pytest.approx(2.0 / 3.0)
    assert metrics["trajectory_length_2_fraction"] == pytest.approx(1.0 / 3.0)
    assert metrics["invalid_reward_floor_fraction"] == pytest.approx(0.25)
    assert metrics["stage1_num_trajectories"] == pytest.approx(3.0)
    assert metrics["stage1_valid_fraction"] == pytest.approx(2.0 / 3.0)
    assert metrics["stage1_invalid_reward_floor_fraction"] == pytest.approx(1.0 / 3.0)
    assert metrics["stage1_termination_fraction_stop_token"] == pytest.approx(2.0 / 3.0)
    assert metrics["stage1_termination_fraction_max_stage_new_tokens"] == pytest.approx(1.0 / 3.0)
    assert metrics["stage2_num_trajectories"] == pytest.approx(1.0)
    assert metrics["stage2_valid_fraction"] == pytest.approx(1.0)
    assert metrics["stage2_termination_fraction_stop_token"] == pytest.approx(1.0)
    assert metrics["all_finite"] is True
    assert "objective_loss" in metrics
    assert "grad_norm" in metrics
    assert "sampling_duration_sec" in metrics
    assert "target_guidance_sampling_duration_sec" in metrics
    assert "action_tokens_per_sec" in metrics
    assert "mean_log_pf_token" in metrics
    assert result.diagnostic_metrics is not None
    assert result.categorized_diagnostic_metrics is not None
    assert result.categorized_diagnostic_metrics["sec_timer"]["sampling_duration_sec"] > 0.0
    assert result.categorized_diagnostic_metrics["stage_rollout"]["mean_trajectory_length"] == pytest.approx(
        4.0 / 3.0
    )
    assert result.trajectory_preview is not None
    assert len(result.trajectory_preview["records"]) == 3
    assert {record["preview_slot"] for record in result.trajectory_preview["records"]} == {
        "best",
        "median",
        "worst",
    }


def test_train_iteration_trims_on_policy_before_target_guidance_counts(monkeypatch) -> None:
    class DummyModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(1.0))
            self.policy_model = SimpleNamespace(config=SimpleNamespace(decoder_start_token_id=0))

        def save_checkpoint(self, *args, **kwargs):
            del args, kwargs
            return None

    trainer = MultiMoleculeGFlowNetTrainer(
        model=DummyModel(),
        tokenizer=None,
        config=GFlowNetConfig(
            batch_size=1,
            objective="tb",
            save_every_iterations=99,
            diagnostic_log_every_iterations=1,
            trajectory_preview_every_iterations=99,
            rollout=GFlowNetRolloutConfig(max_molecules_per_sequence=2),
            target_guidance=TargetGuidanceConfig(
                enabled=True,
                on_policy_fraction=0.25,
                target_prefix_rollout_fraction=0.50,
                target_teacher_fraction=0.25,
            ),
        ),
        device=torch.device("cpu"),
    )

    raw_on_policy = [
        _make_sampled_trajectory(rollout_id=f"fresh-{index}", terminal_reward=1.0 + index)
        for index in range(4)
    ]
    target_prefix_items = [
        _make_sampled_trajectory(
            rollout_id=f"target-prefix-{index}",
            terminal_reward=5.0,
            metadata_source="target_prefix_rollout",
        )
        for index in range(4)
    ]
    target_teacher_items = [
        _make_sampled_trajectory(
            rollout_id=f"target-teacher-{index}",
            terminal_reward=6.0,
            metadata_source="target_teacher",
        )
        for index in range(2)
    ]
    sampling_modes: list[bool] = []
    score_modes: list[bool] = []
    prefix_calls: list[int] = []
    teacher_calls: list[int] = []

    def fake_collect_on_policy(_examples, *, iteration_index):
        del _examples, iteration_index
        sampling_modes.append(trainer.model.training)
        return raw_on_policy

    def fake_collect_target_prefix(_examples, *, iteration_index, count):
        del _examples, iteration_index
        sampling_modes.append(trainer.model.training)
        prefix_calls.append(int(count))
        return target_prefix_items[:count]

    def fake_collect_target_teacher(_examples, *, iteration_index, count):
        del _examples, iteration_index
        sampling_modes.append(trainer.model.training)
        teacher_calls.append(int(count))
        return target_teacher_items[:count]

    def fake_score(trajectories):
        score_modes.append(trainer.model.training)
        base = trainer.model.weight
        return [
            ScoredStageTrajectory(
                sampled=trajectory,
                log_pf_tokens=tuple(base * 0.1 for _ in trajectory.action_token_ids),
                log_stop=tuple(base * -0.2 for _ in range(len(trajectory.action_token_ids) + 1)),
                log_state_flows=tuple(
                    base * 0.3 for _ in range(len(trajectory.action_token_ids) + 1)
                ),
            )
            for trajectory in trajectories
        ]

    monkeypatch.setattr(trainer, "collect_on_policy_trajectories", fake_collect_on_policy)
    monkeypatch.setattr(trainer, "collect_target_prefix_trajectories", fake_collect_target_prefix)
    monkeypatch.setattr(trainer, "collect_target_teacher_trajectories", fake_collect_target_teacher)
    monkeypatch.setattr(trainer, "score_trajectories", fake_score)

    result = trainer.train_iteration([{"id": "unused"}], iteration_index=1)

    assert result.metrics["num_on_policy_trajectories_raw"] == pytest.approx(4.0)
    assert result.metrics["num_on_policy_trajectories"] == pytest.approx(2.0)
    assert result.metrics["num_on_policy_trajectories_trimmed"] == pytest.approx(2.0)
    assert result.metrics["num_target_prefix_trajectories"] == pytest.approx(4.0)
    assert result.metrics["num_target_teacher_trajectories"] == pytest.approx(2.0)
    assert prefix_calls == [4]
    assert teacher_calls == [2]
    assert sampling_modes == [False, False, False]
    assert score_modes == [True]


def test_train_iteration_caps_optimization_trajectories_after_priority_selection(
    monkeypatch,
) -> None:
    class DummyModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(1.0))
            self.policy_model = SimpleNamespace(config=SimpleNamespace(decoder_start_token_id=0))

        def save_checkpoint(self, *args, **kwargs):
            del args, kwargs
            return None

    trainer = MultiMoleculeGFlowNetTrainer(
        model=DummyModel(),
        tokenizer=None,
        config=GFlowNetConfig(
            batch_size=1,
            objective="tb",
            max_optimization_trajectories_per_iter=3,
            save_every_iterations=99,
            diagnostic_log_every_iterations=1,
            trajectory_preview_every_iterations=99,
            rollout=GFlowNetRolloutConfig(max_molecules_per_sequence=2),
            target_guidance=TargetGuidanceConfig(
                enabled=True,
                on_policy_fraction=0.25,
                target_prefix_rollout_fraction=0.50,
                target_teacher_fraction=0.25,
            ),
        ),
        device=torch.device("cpu"),
    )

    on_policy = [
        _make_sampled_trajectory(rollout_id=f"fresh-{index}", terminal_reward=1.0)
        for index in range(2)
    ]
    target_prefix_items = [
        _make_sampled_trajectory(
            rollout_id=f"target-prefix-{index}",
            terminal_reward=5.0,
            metadata_source="target_prefix_rollout",
        )
        for index in range(4)
    ]
    target_teacher_items = [
        _make_sampled_trajectory(
            rollout_id=f"target-teacher-{index}",
            terminal_reward=6.0,
            metadata_source="target_teacher",
        )
        for index in range(2)
    ]
    scored_rollout_ids: list[str] = []

    monkeypatch.setattr(
        trainer,
        "collect_on_policy_trajectories",
        lambda _examples, iteration_index: on_policy,
    )
    monkeypatch.setattr(
        trainer,
        "collect_target_prefix_trajectories",
        lambda _examples, *, iteration_index, count: target_prefix_items[:count],
    )
    monkeypatch.setattr(
        trainer,
        "collect_target_teacher_trajectories",
        lambda _examples, *, iteration_index, count: target_teacher_items[:count],
    )

    def fake_score(trajectories):
        scored_rollout_ids.extend(trajectory.rollout_id for trajectory in trajectories)
        base = trainer.model.weight
        return [
            ScoredStageTrajectory(
                sampled=trajectory,
                log_pf_tokens=tuple(base * 0.1 for _ in trajectory.action_token_ids),
                log_stop=tuple(base * -0.2 for _ in range(len(trajectory.action_token_ids) + 1)),
                log_state_flows=tuple(
                    base * 0.3 for _ in range(len(trajectory.action_token_ids) + 1)
                ),
            )
            for trajectory in trajectories
        ]

    monkeypatch.setattr(trainer, "score_trajectories", fake_score)
    monkeypatch.setattr(trainer, "save_best_checkpoint", lambda **kwargs: None)

    result = trainer.train_iteration([{"id": "unused"}], iteration_index=1)

    assert scored_rollout_ids == ["target-teacher-0", "target-teacher-1", "fresh-0"]
    assert result.metrics["num_optimization_trajectories_raw"] == pytest.approx(8.0)
    assert result.metrics["num_optimization_trajectories"] == pytest.approx(3.0)
    assert result.metrics["num_optimization_trajectories_trimmed"] == pytest.approx(5.0)
    assert result.metrics["max_optimization_trajectories_per_iter"] == pytest.approx(3.0)
    assert result.metrics["num_target_teacher_trajectories"] == pytest.approx(2.0)
    assert result.metrics["num_target_prefix_trajectories"] == pytest.approx(4.0)


def test_train_iteration_proceeds_with_target_teacher_when_on_policy_is_empty(
    monkeypatch,
) -> None:
    class DummyModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(1.0))
            self.policy_model = SimpleNamespace(config=SimpleNamespace(decoder_start_token_id=0))

        def to(self, device):
            return super().to(device)

        def save_checkpoint(self, *args, **kwargs):
            del args, kwargs
            return None

    trainer = MultiMoleculeGFlowNetTrainer(
        model=DummyModel(),
        tokenizer=None,
        config=GFlowNetConfig(
            objective="db",
            save_every_iterations=99,
            diagnostic_log_every_iterations=1,
            trajectory_preview_every_iterations=99,
            replay=ReplayConfig(enabled=False),
            target_guidance=TargetGuidanceConfig(
                enabled=True,
                on_policy_fraction=0.25,
                target_prefix_rollout_fraction=0.50,
                target_teacher_fraction=0.25,
            ),
        ),
        device=torch.device("cpu"),
    )
    teacher = [_make_sampled_trajectory(rollout_id="teacher-1", terminal_reward=6.0)]
    prefix_counts: list[int] = []
    teacher_counts: list[int] = []

    monkeypatch.setattr(
        trainer,
        "collect_on_policy_trajectories",
        lambda _examples, iteration_index: [],
    )

    def fake_collect_target_prefix(_examples, *, iteration_index, count):
        del iteration_index
        prefix_counts.append(int(count))
        return []

    def fake_collect_target_teacher(_examples, *, iteration_index, count):
        del iteration_index
        teacher_counts.append(int(count))
        return teacher[:1]

    monkeypatch.setattr(
        trainer,
        "collect_target_prefix_trajectories",
        fake_collect_target_prefix,
    )
    monkeypatch.setattr(
        trainer,
        "collect_target_teacher_trajectories",
        fake_collect_target_teacher,
    )

    def fake_score(trajectories):
        base = trainer.model.weight
        return [
            ScoredStageTrajectory(
                sampled=trajectory,
                log_pf_tokens=tuple(base * 0.1 for _ in trajectory.action_token_ids),
                log_stop=tuple(base * -0.2 for _ in range(len(trajectory.action_token_ids) + 1)),
                log_state_flows=tuple(
                    base * 0.3 for _ in range(len(trajectory.action_token_ids) + 1)
                ),
            )
            for trajectory in trajectories
        ]

    monkeypatch.setattr(trainer, "score_trajectories", fake_score)
    monkeypatch.setattr(trainer, "save_best_checkpoint", lambda **kwargs: None)

    result = trainer.train_iteration([{"id": "example-1"}], iteration_index=1)

    assert prefix_counts == [2]
    assert teacher_counts == [1]
    assert result.metrics["num_on_policy_trajectories"] == 0.0
    assert result.metrics["num_target_prefix_trajectories"] == 0.0
    assert result.metrics["num_target_teacher_trajectories"] == 1.0
    assert result.metrics["num_target_guided_trajectories"] == 1.0
    assert result.metrics["mean_stage_reward"] == pytest.approx(0.0)
    assert result.metrics["target_teacher_mean_stage_reward"] == pytest.approx(6.0)
    assert result.metrics["all_finite"] is True
    assert "objective_loss" in result.metrics


def test_save_checkpoint_overwrites_last_archive_on_every_save(
    tmp_path: Path,
) -> None:
    class DummyTokenizer:
        def save_pretrained(self, output_dir: str | Path) -> None:
            path = Path(output_dir)
            path.mkdir(parents=True, exist_ok=True)
            (path / "tokenizer_config.json").write_text("{}", encoding="utf-8")
            (path / "spiece.model").write_text("spiece", encoding="utf-8")

    class DummyModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(1.0))
            self.policy_model = SimpleNamespace(config=SimpleNamespace(decoder_start_token_id=0))

        def save_checkpoint(
            self,
            output_dir: str | Path,
            *,
            tokenizer=None,
            config=None,
            metrics=None,
            create_archive: bool = False,
        ) -> None:
            del create_archive
            path = Path(output_dir)
            path.mkdir(parents=True, exist_ok=True)
            (path / "model.bin").write_text("weights", encoding="utf-8")
            if tokenizer is not None:
                tokenizer.save_pretrained(path)
            if config is not None:
                (path / "training_config.json").write_text(
                    json.dumps(config),
                    encoding="utf-8",
                )
            if metrics is not None:
                (path / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")

    output_dir = tmp_path / "outputs"
    trainer = MultiMoleculeGFlowNetTrainer(
        model=DummyModel(),
        tokenizer=DummyTokenizer(),
        config=GFlowNetConfig(
            output_dir=str(output_dir),
            start_iteration=10,
            gflownet_iterations=2,
            save_every_iterations=99,
        ),
        device=torch.device("cpu"),
    )
    trajectories = [_make_sampled_trajectory(rollout_id="last-1", terminal_reward=2.0)]

    non_final_checkpoint_dir = trainer.save_checkpoint(
        iteration_index=11,
        metrics={"objective_loss": 1.5},
        trajectories=trajectories,
    )

    last_checkpoint_dir = output_dir / "checkpoints" / "last"
    last_checkpoint_zip = output_dir / "checkpoints" / "last.zip"

    assert non_final_checkpoint_dir == output_dir / "checkpoints" / "iteration-0011"
    assert last_checkpoint_dir.exists()
    assert last_checkpoint_zip.exists()
    assert trainer.last_checkpoint_iteration == 11
    assert trainer.last_checkpoint_dir == str(last_checkpoint_dir)
    assert trainer.last_checkpoint_zip == str(last_checkpoint_zip)
    assert json.loads((last_checkpoint_dir / "iteration_metrics.json").read_text())[
        "objective_loss"
    ] == 1.5
    with ZipFile(last_checkpoint_zip) as archive:
        assert "last/iteration_metrics.json" in archive.namelist()

    final_checkpoint_dir = trainer.save_checkpoint(
        iteration_index=12,
        metrics={"objective_loss": 1.25},
        trajectories=trajectories,
    )

    assert final_checkpoint_dir == output_dir / "checkpoints" / "iteration-0012"
    assert final_checkpoint_dir.with_suffix(".zip").exists()
    assert last_checkpoint_dir.exists()
    assert last_checkpoint_zip.exists()
    assert trainer.last_checkpoint_iteration == 12
    assert trainer.last_checkpoint_dir == str(last_checkpoint_dir)
    assert trainer.last_checkpoint_zip == str(last_checkpoint_zip)
    assert json.loads((last_checkpoint_dir / "iteration_metrics.json").read_text())[
        "objective_loss"
    ] == 1.25
    with ZipFile(last_checkpoint_zip) as archive:
        assert "last/iteration_metrics.json" in archive.namelist()


def test_save_best_checkpoint_tracks_lowest_objective_loss_and_writes_zip(
    tmp_path: Path,
) -> None:
    class DummyTokenizer:
        def save_pretrained(self, output_dir: str | Path) -> None:
            path = Path(output_dir)
            path.mkdir(parents=True, exist_ok=True)
            (path / "tokenizer_config.json").write_text("{}", encoding="utf-8")
            (path / "spiece.model").write_text("spiece", encoding="utf-8")

    class DummyModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(1.0))
            self.policy_model = SimpleNamespace(config=SimpleNamespace(decoder_start_token_id=0))

        def save_checkpoint(
            self,
            output_dir: str | Path,
            *,
            tokenizer=None,
            config=None,
            metrics=None,
            create_archive: bool = False,
        ) -> None:
            del create_archive
            path = Path(output_dir)
            path.mkdir(parents=True, exist_ok=True)
            (path / "model.bin").write_text("weights", encoding="utf-8")
            if tokenizer is not None:
                tokenizer.save_pretrained(path)
            if config is not None:
                (path / "training_config.json").write_text(
                    json.dumps(config),
                    encoding="utf-8",
                )
            if metrics is not None:
                (path / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")

    trainer = MultiMoleculeGFlowNetTrainer(
        model=DummyModel(),
        tokenizer=DummyTokenizer(),
        config=GFlowNetConfig(output_dir=str(tmp_path / "outputs"), save_every_iterations=99),
        device=torch.device("cpu"),
    )
    trajectories = [_make_sampled_trajectory(rollout_id="best-1", terminal_reward=2.0)]

    checkpoint_dir = trainer.save_best_checkpoint(
        iteration_index=1,
        metrics={"objective_loss": 1.25},
        trajectories=trajectories,
    )

    assert checkpoint_dir == tmp_path / "outputs" / "checkpoints" / "best"
    assert trainer.best_objective_loss == 1.25
    assert trainer.best_checkpoint_iteration == 1
    assert trainer.best_checkpoint_dir == str(checkpoint_dir)
    assert trainer.best_checkpoint_zip == str(checkpoint_dir.with_suffix(".zip"))
    assert json.loads((checkpoint_dir / "iteration_metrics.json").read_text())["objective_loss"] == 1.25
    with ZipFile(checkpoint_dir.with_suffix(".zip")) as archive:
        assert "best/iteration_metrics.json" in archive.namelist()

    unchanged = trainer.save_best_checkpoint(
        iteration_index=2,
        metrics={"objective_loss": 1.25},
        trajectories=trajectories,
    )
    assert unchanged is None
    assert trainer.best_checkpoint_iteration == 1

    updated = trainer.save_best_checkpoint(
        iteration_index=3,
        metrics={"objective_loss": 1.0},
        trajectories=trajectories,
    )
    assert updated == checkpoint_dir
    assert trainer.best_objective_loss == 1.0
    assert trainer.best_checkpoint_iteration == 3
    assert json.loads((checkpoint_dir / "iteration_metrics.json").read_text())["objective_loss"] == 1.0


def test_run_multi_molecule_gflownet_uses_epoch_batches_for_configured_iterations(
    monkeypatch,
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "outputs" / "multi_molecule_gflownet"
    trainer_instances = []

    class DummyTokenizer:
        def __init__(self) -> None:
            self.model_max_length = 0

    class DummyGFlowNetModel:
        def __init__(self) -> None:
            self.policy_model = object()

        def to(self, device) -> None:
            self.device = device

    class DummyTracker:
        def log_config(self, payload) -> None:
            self.payload = payload

        def log_metrics(self, metrics, *, step: int, prefix: str) -> None:
            del metrics, step, prefix

        def log_summary(self, summary, *, prefix: str) -> None:
            del summary, prefix

        def finish(self, *, status: str) -> None:
            self.status = status

    class DummyTrainer:
        def __init__(self, **kwargs) -> None:
            self.kwargs = kwargs
            self.best_objective_loss = None
            self.best_checkpoint_iteration = None
            self.best_checkpoint_dir = None
            self.best_checkpoint_zip = None
            self.last_checkpoint_iteration = None
            self.last_checkpoint_dir = None
            self.last_checkpoint_zip = None
            self.calls: list[tuple[int, list[str]]] = []
            trainer_instances.append(self)

        def train_iteration(self, examples, *, iteration_index: int) -> GFlowNetTrainIterationResult:
            self.calls.append(
                (iteration_index, [str(example["id"]) for example in examples])
            )
            return GFlowNetTrainIterationResult(
                metrics={
                    "iteration": float(iteration_index),
                    "objective_loss": 1.0,
                    "mean_stage_reward": 2.0,
                    "valid_fraction": 1.0,
                    "replay_size": 0.0,
                    "replay_total_action_tokens": 0.0,
                }
            )

    monkeypatch.setattr(
        "post_training.gflownet.trainer.AutoTokenizer",
        SimpleNamespace(from_pretrained=lambda *args, **kwargs: DummyTokenizer()),
    )
    monkeypatch.setattr(
        "post_training.gflownet.trainer.GFlowNetModel",
        SimpleNamespace(from_pretrained=lambda *args, **kwargs: DummyGFlowNetModel()),
    )
    monkeypatch.setattr(
        "post_training.gflownet.trainer.assert_checkpoint_tokenizer_matches_model",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        "post_training.gflownet.trainer.build_reward_config",
        lambda *args, **kwargs: SimpleNamespace(),
    )
    monkeypatch.setattr(
        "post_training.gflownet.trainer.choose_device",
        lambda *_args, **_kwargs: torch.device("cpu"),
    )
    monkeypatch.setattr(
        "post_training.gflownet.trainer.MultiMoleculeGFlowNetTrainer",
        DummyTrainer,
    )
    monkeypatch.setattr(
        "post_training.gflownet.trainer.MultiMoleculeDataset",
        SimpleNamespace(
            from_jsonl=lambda path: [
                {"id": "example-0"},
                {"id": "example-1"},
                {"id": "example-2"},
            ]
        ),
    )
    monkeypatch.setattr(
        "post_training.gflownet.trainer.prepare_gflownet_output_dir",
        lambda *_args, **_kwargs: output_dir,
    )
    monkeypatch.setattr(
        "post_training.gflownet.trainer.build_tracker",
        lambda *args, **kwargs: DummyTracker(),
    )

    config = resolve_gflownet_config_paths(
        {
            "seed": 123,
            "tracking": {"enabled": False},
            "model": {"checkpoint": DEFAULT_PPO_FALLBACK_CHECKPOINT, "use_lora": False},
            "data": {
                "train_file": "data/train_multimol.jsonl",
                "validation_file": "data/validation_multimol.jsonl",
                "test_file": "data/test_multimol.jsonl",
                "max_source_length": 512,
            },
            "training": {
                "output_dir": str(output_dir),
                "device": "cpu",
                "save_every_iterations": 10,
            },
            "gflownet": {
                "gflownet_iterations": 5,
                "batch_size": 2,
                "objective": "tb",
                "rollout": {"constrained_decoding": False},
            },
        },
        project_root=tmp_path,
    )

    summary = run_multi_molecule_gflownet(config)

    assert len(trainer_instances) == 1
    assert [iteration for iteration, _examples in trainer_instances[0].calls] == [
        1,
        2,
        3,
        4,
        5,
    ]
    assert [len(examples) for _iteration, examples in trainer_instances[0].calls] == [
        2,
        1,
        2,
        1,
        2,
    ]
    assert [record["epoch"] for record in summary["history"]] == [
        1.0,
        1.0,
        2.0,
        2.0,
        3.0,
    ]
    first_epoch_ids = [
        example_id
        for _iteration, examples in trainer_instances[0].calls[:2]
        for example_id in examples
    ]
    assert set(first_epoch_ids) == {"example-0", "example-1", "example-2"}
    assert summary["num_iterations"] == 5

    trainer_instances.clear()
    restart_config = resolve_gflownet_config_paths(
        {
            "seed": 123,
            "tracking": {"enabled": False},
            "model": {"checkpoint": DEFAULT_PPO_FALLBACK_CHECKPOINT, "use_lora": False},
            "data": {
                "train_file": "data/train_multimol.jsonl",
                "validation_file": "data/validation_multimol.jsonl",
                "test_file": "data/test_multimol.jsonl",
                "max_source_length": 512,
            },
            "training": {
                "output_dir": str(output_dir),
                "device": "cpu",
                "save_every_iterations": 10,
            },
            "gflownet": {
                "start_iteration": 1500,
                "gflownet_iterations": 2,
                "batch_size": 2,
                "objective": "tb",
                "rollout": {"constrained_decoding": False},
            },
        },
        project_root=tmp_path,
    )

    restart_summary = run_multi_molecule_gflownet(restart_config)

    assert [iteration for iteration, _examples in trainer_instances[0].calls] == [
        1501,
        1502,
    ]
    assert [record["iteration"] for record in restart_summary["history"]] == [
        1501.0,
        1502.0,
    ]
    assert restart_summary["start_iteration"] == 1500
    assert restart_summary["final_iteration"] == 1502
    assert restart_summary["num_iterations"] == 2


def test_run_multi_molecule_gflownet_uses_resolved_checkpoint_source_for_all_model_loads(
    monkeypatch,
    tmp_path: Path,
    capsys,
) -> None:
    load_calls: list[tuple[str, str]] = []
    output_dir = tmp_path / "outputs" / "multi_molecule_gflownet"

    class DummyTokenizer:
        def __init__(self) -> None:
            self.model_max_length = 0

    class DummyGFlowNetModel:
        def __init__(self) -> None:
            self.policy_model = object()

        def to(self, device) -> None:
            self.device = device

    class DummyTracker:
        def __init__(self) -> None:
            self.metric_calls: list[tuple[dict[str, object], int, str]] = []
            self.summary_calls: list[tuple[dict[str, object], str]] = []

        def log_config(self, payload) -> None:
            self.payload = payload

        def log_metrics(self, metrics, *, step: int, prefix: str) -> None:
            self.metric_calls.append((metrics, step, prefix))

        def log_summary(self, summary, *, prefix: str) -> None:
            self.summary_calls.append((summary, prefix))

        def finish(self, *, status: str) -> None:
            self.status = status

    class DummyTrainer:
        def __init__(self, **kwargs) -> None:
            self.kwargs = kwargs
            self.best_objective_loss = None
            self.best_checkpoint_iteration = None
            self.best_checkpoint_dir = None
            self.best_checkpoint_zip = None
            self.last_checkpoint_iteration = None
            self.last_checkpoint_dir = None
            self.last_checkpoint_zip = None

        def train_iteration(self, examples, *, iteration_index: int) -> GFlowNetTrainIterationResult:
            del examples
            self.best_objective_loss = 1.25
            self.best_checkpoint_iteration = iteration_index
            self.best_checkpoint_dir = str(output_dir / "checkpoints" / "best")
            self.best_checkpoint_zip = str(output_dir / "checkpoints" / "best.zip")
            self.last_checkpoint_iteration = iteration_index
            self.last_checkpoint_dir = str(output_dir / "checkpoints" / "last")
            self.last_checkpoint_zip = str(output_dir / "checkpoints" / "last.zip")
            return GFlowNetTrainIterationResult(
                metrics={
                    "iteration": float(iteration_index),
                    "objective_loss": 1.25,
                    "mean_stage_reward": 2.5,
                    "mean_training_stage_reward": 3.0,
                    "valid_fraction": 1.0,
                    "num_on_policy_trajectories": 4.0,
                    "num_optimization_trajectories": 8.0,
                    "num_target_prefix_trajectories": 2.0,
                    "num_target_teacher_trajectories": 2.0,
                    "num_target_guided_trajectories": 4.0,
                    "target_prefix_mean_stage_reward": 5.0,
                    "target_teacher_mean_stage_reward": 6.0,
                    "target_prefix_valid_fraction": 0.5,
                    "target_teacher_valid_fraction": 0.5,
                    "replay_size": 0.0,
                    "replay_total_action_tokens": 0.0,
                    "duplicate_count_on_policy": 1.0,
                    "num_novel_on_policy": 1.0,
                    "num_valid_on_policy_for_novelty": 4.0,
                    "novelty_fraction_on_policy": 0.25,
                    "mean_trajectory_length": 1.0,
                    "num_rollouts": 4.0,
                    "fraction_rollouts_trajectory_length_2_plus": 0.0,
                    "grad_norm": 0.5,
                    "termination_fraction_stop_token": 0.75,
                    "termination_fraction_max_stage_new_tokens": 0.25,
                    "iteration_duration_sec": 2.0,
                },
                diagnostic_metrics={
                    "iteration": float(iteration_index),
                    "grad_norm": 0.5,
                    "sampling_duration_sec": 0.25,
                    "mean_trajectory_length": 1.0,
                    "fraction_rollouts_trajectory_length_2_plus": 0.0,
                },
                trajectory_preview={
                    "iteration": iteration_index,
                    "records": [
                        {
                            "iteration": iteration_index,
                            "preview_slot": "best",
                            "rollout_id": "rollout-1",
                            "example_id": "example-1",
                            "total_reward": 2.5,
                            "stage_rewards": [2.5],
                            "raw_stage_text_sequence": ["<bom>[C][C][O]<eom>"],
                            "generated_selfies_sequence": ["[C][C][O]"],
                            "new_action_token_ids_sequence": [[1, 2]],
                            "new_action_text_sequence": [None],
                            "termination_reasons": ["stop_token"],
                            "valid_sequence": [True],
                            "duplicate_sequence": [False],
                        }
                    ],
                    "tracker_text": "preview-text",
                },
            )

    tracker = DummyTracker()

    monkeypatch.setattr(
        "post_training.gflownet.trainer.AutoTokenizer",
        SimpleNamespace(
            from_pretrained=lambda source, use_fast=True: (
                load_calls.append(("tokenizer", source)),
                DummyTokenizer(),
            )[1]
        ),
    )
    monkeypatch.setattr(
        "post_training.gflownet.trainer.GFlowNetModel",
        SimpleNamespace(
            from_pretrained=lambda source, **kwargs: (
                load_calls.append(("model", source)),
                DummyGFlowNetModel(),
            )[1]
        ),
    )
    monkeypatch.setattr(
        "post_training.gflownet.trainer.assert_checkpoint_tokenizer_matches_model",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        "post_training.gflownet.trainer.build_reward_config",
        lambda *args, **kwargs: SimpleNamespace(),
    )
    monkeypatch.setattr(
        "post_training.gflownet.trainer.choose_device",
        lambda *_args, **_kwargs: torch.device("cpu"),
    )
    monkeypatch.setattr(
        "post_training.gflownet.trainer.MultiMoleculeGFlowNetTrainer",
        DummyTrainer,
    )
    monkeypatch.setattr(
        "post_training.gflownet.trainer.MultiMoleculeDataset",
        SimpleNamespace(
            from_jsonl=lambda path: [
                {
                    "id": "example-1",
                    "prompt": "prompt",
                    "description": "description",
                    "target_selfies_list": ["[C][C][O]"],
                }
            ]
        ),
    )
    monkeypatch.setattr(
        "post_training.gflownet.trainer.build_tracker",
        lambda *args, **kwargs: tracker,
    )

    config = resolve_gflownet_config_paths(
        {
            "seed": 42,
            "tracking": {"enabled": False},
            "model": {
                "checkpoint": "outputs/multi_molecule_sft_mini/checkpoints/best",
                "use_lora": False,
            },
            "data": {
                "train_file": "data/train_multimol.jsonl",
                "validation_file": "data/validation_multimol.jsonl",
                "test_file": "data/test_multimol.jsonl",
                "max_source_length": 512,
            },
            "training": {
                "output_dir": str(output_dir),
                "device": "cpu",
                "save_every_iterations": 10,
            },
            "gflownet": {
                "gflownet_iterations": 1,
                "batch_size": 1,
                "objective": "tb",
                "diagnostic_log_every_iterations": 1,
                "trajectory_preview_every_iterations": 1,
                "rollout": {"constrained_decoding": False},
            },
        },
        project_root=tmp_path,
    )

    summary = run_multi_molecule_gflownet(config)
    captured = capsys.readouterr()

    assert summary["resolved_checkpoint_source"] == DEFAULT_PPO_FALLBACK_CHECKPOINT
    assert summary["best_objective_loss"] == 1.25
    assert summary["best_checkpoint_iteration"] == 1
    assert summary["best_checkpoint_dir"] == str(output_dir / "checkpoints" / "best")
    assert summary["best_checkpoint_zip"] == str(output_dir / "checkpoints" / "best.zip")
    assert summary["last_checkpoint_iteration"] == 1
    assert summary["last_checkpoint_dir"] == str(output_dir / "checkpoints" / "last")
    assert summary["last_checkpoint_zip"] == str(output_dir / "checkpoints" / "last.zip")
    assert "[gflownet][iteration 1] trajectory preview" in captured.out
    assert "preview-text" in captured.out
    assert load_calls == [
        ("tokenizer", DEFAULT_PPO_FALLBACK_CHECKPOINT),
        ("model", DEFAULT_PPO_FALLBACK_CHECKPOINT),
    ]
    headline_calls = [payload for payload, _, prefix in tracker.metric_calls if prefix == "gflownet"]
    diagnostic_calls = [
        payload for payload, _, prefix in tracker.metric_calls if prefix == "gflownet_diagnostics"
    ]
    report_calls = [
        payload for payload, _, prefix in tracker.metric_calls if prefix == "gflownet_report"
    ]
    report_appendix_calls = [
        payload
        for payload, _, prefix in tracker.metric_calls
        if prefix == "gflownet_report_appendix"
    ]
    categorized_calls = {
        prefix: payload
        for payload, _, prefix in tracker.metric_calls
        if prefix.startswith("gflownet_diagnostics_")
    }
    assert headline_calls == [
        {
            "iteration": 1.0,
            "objective_loss": 1.25,
            "mean_stage_reward": 2.5,
            "mean_training_stage_reward": 3.0,
            "valid_fraction": 1.0,
            "num_on_policy_trajectories": 4.0,
            "num_target_prefix_trajectories": 2.0,
            "num_target_teacher_trajectories": 2.0,
            "num_target_guided_trajectories": 4.0,
            "target_prefix_valid_fraction": 0.5,
            "target_teacher_valid_fraction": 0.5,
            "target_prefix_mean_stage_reward": 5.0,
            "target_teacher_mean_stage_reward": 6.0,
            "replay_size": 0.0,
            "replay_total_action_tokens": 0.0,
            "grad_norm": 0.5,
        }
    ]
    assert diagnostic_calls == [
        {
            "iteration": 1.0,
            "grad_norm": 0.5,
            "sampling_duration_sec": 0.25,
            "mean_trajectory_length": 1.0,
            "fraction_rollouts_trajectory_length_2_plus": 0.0,
        }
    ]
    assert report_calls == [
        {
            "loss_function_step": 1.25,
            "loss_function_running_sum": 1.25,
            "loss_function_running_mean": 1.25,
            "loss_function_weight_sum": 1.0,
            "average_on_policy_reward_step": 2.5,
            "average_on_policy_reward_running_sum": 10.0,
            "average_on_policy_reward_running_mean": 2.5,
            "average_on_policy_reward_weight_sum": 4.0,
            "average_teacher_reward_step": 6.0,
            "average_teacher_reward_running_sum": 12.0,
            "average_teacher_reward_running_mean": 6.0,
            "average_teacher_reward_weight_sum": 2.0,
            "average_valid_on_policy_step": 1.0,
            "average_valid_on_policy_running_sum": 4.0,
            "average_valid_on_policy_running_mean": 1.0,
            "average_valid_on_policy_weight_sum": 4.0,
            "average_valid_teacher_step": 0.5,
            "average_valid_teacher_running_sum": 1.0,
            "average_valid_teacher_running_mean": 0.5,
            "average_valid_teacher_weight_sum": 2.0,
            "average_trajectory_length_on_policy_step": 1.0,
            "average_trajectory_length_on_policy_running_sum": 4.0,
            "average_trajectory_length_on_policy_running_mean": 1.0,
            "average_trajectory_length_on_policy_weight_sum": 4.0,
            "novelty_fraction_on_policy_step": 0.25,
            "novelty_fraction_on_policy_running_sum": 1.0,
            "novelty_fraction_on_policy_running_mean": 0.25,
            "novelty_fraction_on_policy_weight_sum": 4.0,
        }
    ]
    assert report_appendix_calls == [
        {
            "average_prefix_reward_step": 5.0,
            "average_prefix_reward_running_sum": 10.0,
            "average_prefix_reward_running_mean": 5.0,
            "average_prefix_reward_weight_sum": 2.0,
            "average_valid_prefix_step": 0.5,
            "average_valid_prefix_running_sum": 1.0,
            "average_valid_prefix_running_mean": 0.5,
            "average_valid_prefix_weight_sum": 2.0,
            "average_training_reward_step": 3.0,
            "average_training_reward_running_sum": 24.0,
            "average_training_reward_running_mean": 3.0,
            "average_training_reward_weight_sum": 8.0,
            "duplicate_count_on_policy_step": 1.0,
            "duplicate_count_on_policy_running_sum": 1.0,
            "num_novel_on_policy_step": 1.0,
            "num_novel_on_policy_running_sum": 1.0,
            "num_novel_on_policy_running_mean": 1.0,
            "num_novel_on_policy_weight_sum": 1.0,
            "num_valid_on_policy_for_novelty_step": 4.0,
            "num_valid_on_policy_for_novelty_running_sum": 4.0,
            "num_valid_on_policy_for_novelty_running_mean": 4.0,
            "num_valid_on_policy_for_novelty_weight_sum": 1.0,
            "gradient_norm_step": 0.5,
            "gradient_norm_running_sum": 0.5,
            "gradient_norm_running_mean": 0.5,
            "gradient_norm_weight_sum": 1.0,
            "termination_fraction_stop_token_step": 0.75,
            "termination_fraction_stop_token_running_sum": 3.0,
            "termination_fraction_stop_token_running_mean": 0.75,
            "termination_fraction_stop_token_weight_sum": 4.0,
            "termination_fraction_max_stage_new_tokens_step": 0.25,
            "termination_fraction_max_stage_new_tokens_running_sum": 1.0,
            "termination_fraction_max_stage_new_tokens_running_mean": 0.25,
            "termination_fraction_max_stage_new_tokens_weight_sum": 4.0,
            "iteration_duration_sec_step": 2.0,
            "iteration_duration_sec_running_sum": 2.0,
            "iteration_duration_sec_running_mean": 2.0,
            "iteration_duration_sec_weight_sum": 1.0,
        }
    ]
    assert "duplicate_fraction_step" not in report_appendix_calls[0]
    assert categorized_calls == {
        "gflownet_diagnostics_stage_rollout": {
            "mean_trajectory_length": 1.0,
            "fraction_rollouts_trajectory_length_2_plus": 0.0,
        },
        "gflownet_diagnostics_sec_timer": {
            "sampling_duration_sec": 0.25,
        },
        "gflownet_diagnostics_numerics": {
            "grad_norm": 0.5,
        },
    }
    assert "mean_trajectory_length" not in headline_calls[0]
    assert tracker.summary_calls[0] == (
        {
            "latest_trajectory_preview": "preview-text",
            "latest_trajectory_preview_iteration": 1.0,
        },
        "gflownet",
    )
    assert tracker.summary_calls[1][1] == "gflownet"
    diagnostics_dir = output_dir / "diagnostics"
    iteration_diagnostics = diagnostics_dir / "iteration_diagnostics.jsonl"
    iteration_diagnostics_categorized = diagnostics_dir / "iteration_diagnostics_categorized.jsonl"
    report_metrics = diagnostics_dir / "gflownet_report_metrics.jsonl"
    trajectory_previews = diagnostics_dir / "trajectory_previews.jsonl"
    assert iteration_diagnostics.exists()
    assert iteration_diagnostics_categorized.exists()
    assert report_metrics.exists()
    assert trajectory_previews.exists()
    assert [json.loads(line) for line in iteration_diagnostics.read_text().splitlines()] == [
        {
            "iteration": 1.0,
            "grad_norm": 0.5,
            "sampling_duration_sec": 0.25,
            "mean_trajectory_length": 1.0,
            "fraction_rollouts_trajectory_length_2_plus": 0.0,
        }
    ]
    assert [json.loads(line) for line in iteration_diagnostics_categorized.read_text().splitlines()] == [
        {
            "iteration": 1.0,
            "categories": {
                "stage_rollout": {
                    "mean_trajectory_length": 1.0,
                    "fraction_rollouts_trajectory_length_2_plus": 0.0,
                },
                "sec_timer": {
                    "sampling_duration_sec": 0.25,
                },
                "numerics": {
                    "grad_norm": 0.5,
                },
            },
        }
    ]
    assert [json.loads(line) for line in report_metrics.read_text().splitlines()] == [
        {
            "iteration": 1.0,
            "main": report_calls[0],
            "appendix": report_appendix_calls[0],
        }
    ]
    assert [json.loads(line) for line in trajectory_previews.read_text().splitlines()] == [
        {
            "iteration": 1,
            "preview_slot": "best",
            "rollout_id": "rollout-1",
            "example_id": "example-1",
            "total_reward": 2.5,
            "stage_rewards": [2.5],
            "raw_stage_text_sequence": ["<bom>[C][C][O]<eom>"],
            "generated_selfies_sequence": ["[C][C][O]"],
            "new_action_token_ids_sequence": [[1, 2]],
            "new_action_text_sequence": [None],
            "termination_reasons": ["stop_token"],
            "valid_sequence": [True],
            "duplicate_sequence": [False],
        }
    ]


def test_run_multi_molecule_gflownet_builds_constraints_once_and_attaches_to_model(
    monkeypatch,
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "outputs" / "multi_molecule_gflownet"
    built_constraints = StageTokenConstraints(
        bom_token_id=1,
        eom_token_id=2,
        content_token_ids=(3, 4),
    )
    builder_calls: list[tuple[str, str]] = []

    class DummyTokenizer:
        def __init__(self) -> None:
            self.model_max_length = 0

    class DummyGFlowNetModel:
        def __init__(self) -> None:
            self.policy_model = object()
            self.constraints: list[StageTokenConstraints | None] = []

        def to(self, device) -> None:
            self.device = device

        def set_stage_token_constraints(self, constraints) -> None:
            self.constraints.append(constraints)

        def get_stage_token_constraints(self):
            return self.constraints[-1] if self.constraints else None

    class DummyTracker:
        def log_config(self, payload) -> None:
            self.payload = payload

        def log_metrics(self, metrics, *, step: int, prefix: str) -> None:
            del metrics, step, prefix

        def log_summary(self, summary, *, prefix: str) -> None:
            del summary, prefix

        def finish(self, *, status: str) -> None:
            self.status = status

    class DummyTrainer:
        model = None

        def __init__(self, **kwargs) -> None:
            type(self).model = kwargs["model"]

    model = DummyGFlowNetModel()

    monkeypatch.setattr(
        "post_training.gflownet.trainer.AutoTokenizer",
        SimpleNamespace(from_pretrained=lambda *args, **kwargs: DummyTokenizer()),
    )
    monkeypatch.setattr(
        "post_training.gflownet.trainer.GFlowNetModel",
        SimpleNamespace(from_pretrained=lambda *args, **kwargs: model),
    )
    monkeypatch.setattr(
        "post_training.gflownet.trainer.assert_checkpoint_tokenizer_matches_model",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        "post_training.gflownet.trainer.build_reward_config",
        lambda *args, **kwargs: SimpleNamespace(),
    )
    monkeypatch.setattr(
        "post_training.gflownet.trainer.choose_device",
        lambda *_args, **_kwargs: torch.device("cpu"),
    )
    monkeypatch.setattr(
        "post_training.gflownet.trainer.MultiMoleculeGFlowNetTrainer",
        DummyTrainer,
    )
    monkeypatch.setattr(
        "post_training.gflownet.trainer.MultiMoleculeDataset",
        SimpleNamespace(
            from_jsonl=lambda path: [
                {
                    "id": "example-1",
                    "prompt": "prompt",
                    "description": "description",
                    "target_selfies_list": ["[C]"],
                }
            ]
        ),
    )
    monkeypatch.setattr(
        "post_training.gflownet.trainer.prepare_gflownet_output_dir",
        lambda *_args, **_kwargs: output_dir,
    )
    monkeypatch.setattr(
        "post_training.gflownet.trainer.build_tracker",
        lambda *args, **kwargs: DummyTracker(),
    )
    monkeypatch.setattr(
        "post_training.gflownet.trainer.build_stage_token_constraints",
        lambda _tokenizer, _dataset, *, selfies_dict_path, separator_token: (
            builder_calls.append((selfies_dict_path, separator_token)),
            built_constraints,
        )[1],
    )

    config = resolve_gflownet_config_paths(
        {
            "tracking": {"enabled": False},
            "model": {"checkpoint": DEFAULT_PPO_FALLBACK_CHECKPOINT, "use_lora": False},
            "data": {
                "train_file": "data/train_multimol.jsonl",
                "validation_file": "data/validation_multimol.jsonl",
                "test_file": "data/test_multimol.jsonl",
                "max_source_length": 512,
            },
            "training": {
                "output_dir": str(output_dir),
                "device": "cpu",
                "save_every_iterations": 10,
            },
            "gflownet": {
                "gflownet_iterations": 0,
                "batch_size": 1,
                "objective": "tb",
                "rollout": {
                    "constrained_decoding": True,
                    "selfies_dict_path": "custom_selfies_dict.txt",
                },
            },
        },
        project_root=tmp_path,
    )

    run_multi_molecule_gflownet(config)

    assert builder_calls == [("custom_selfies_dict.txt", " ")]
    assert model.constraints == [built_constraints]
    assert DummyTrainer.model.get_stage_token_constraints() is built_constraints


def test_run_multi_molecule_gflownet_skips_constraint_initialization_when_disabled(
    monkeypatch,
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "outputs" / "multi_molecule_gflownet"

    class DummyTokenizer:
        def __init__(self) -> None:
            self.model_max_length = 0

    class DummyGFlowNetModel:
        def __init__(self) -> None:
            self.policy_model = object()
            self.constraints: list[StageTokenConstraints | None] = []

        def to(self, device) -> None:
            self.device = device

        def set_stage_token_constraints(self, constraints) -> None:
            self.constraints.append(constraints)

    class DummyTracker:
        def log_config(self, payload) -> None:
            self.payload = payload

        def log_metrics(self, metrics, *, step: int, prefix: str) -> None:
            del metrics, step, prefix

        def log_summary(self, summary, *, prefix: str) -> None:
            del summary, prefix

        def finish(self, *, status: str) -> None:
            self.status = status

    class DummyTrainer:
        def __init__(self, **kwargs) -> None:
            self.kwargs = kwargs

    model = DummyGFlowNetModel()

    monkeypatch.setattr(
        "post_training.gflownet.trainer.AutoTokenizer",
        SimpleNamespace(from_pretrained=lambda *args, **kwargs: DummyTokenizer()),
    )
    monkeypatch.setattr(
        "post_training.gflownet.trainer.GFlowNetModel",
        SimpleNamespace(from_pretrained=lambda *args, **kwargs: model),
    )
    monkeypatch.setattr(
        "post_training.gflownet.trainer.assert_checkpoint_tokenizer_matches_model",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        "post_training.gflownet.trainer.build_reward_config",
        lambda *args, **kwargs: SimpleNamespace(),
    )
    monkeypatch.setattr(
        "post_training.gflownet.trainer.choose_device",
        lambda *_args, **_kwargs: torch.device("cpu"),
    )
    monkeypatch.setattr(
        "post_training.gflownet.trainer.MultiMoleculeGFlowNetTrainer",
        DummyTrainer,
    )
    monkeypatch.setattr(
        "post_training.gflownet.trainer.MultiMoleculeDataset",
        SimpleNamespace(from_jsonl=lambda path: []),
    )
    monkeypatch.setattr(
        "post_training.gflownet.trainer.prepare_gflownet_output_dir",
        lambda *_args, **_kwargs: output_dir,
    )
    monkeypatch.setattr(
        "post_training.gflownet.trainer.build_tracker",
        lambda *args, **kwargs: DummyTracker(),
    )
    monkeypatch.setattr(
        "post_training.gflownet.trainer.build_stage_token_constraints",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("builder should not be called when constrained decoding is disabled")
        ),
    )

    config = resolve_gflownet_config_paths(
        {
            "tracking": {"enabled": False},
            "model": {"checkpoint": DEFAULT_PPO_FALLBACK_CHECKPOINT, "use_lora": False},
            "data": {
                "train_file": "data/train_multimol.jsonl",
                "validation_file": "data/validation_multimol.jsonl",
                "test_file": "data/test_multimol.jsonl",
                "max_source_length": 512,
            },
            "training": {
                "output_dir": str(output_dir),
                "device": "cpu",
                "save_every_iterations": 10,
            },
            "gflownet": {
                "gflownet_iterations": 0,
                "batch_size": 1,
                "objective": "tb",
                "rollout": {"constrained_decoding": False},
            },
        },
        project_root=tmp_path,
    )

    run_multi_molecule_gflownet(config)

    assert model.constraints == [None]
