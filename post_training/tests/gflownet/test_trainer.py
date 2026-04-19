from pathlib import Path
from types import SimpleNamespace

import torch

from src.constants import EOM_TOKEN

from post_training.gflownet.config import GFlowNetConfig, ReplayConfig
from post_training.gflownet.trajectory import SampledStageTrajectory, ScoredStageTrajectory
from post_training.gflownet.trainer import MultiMoleculeGFlowNetTrainer, run_multi_molecule_gflownet
from post_training.shared.config import DEFAULT_PPO_FALLBACK_CHECKPOINT, resolve_gflownet_config_paths


def _make_sampled_trajectory(
    *,
    rollout_id: str,
    terminal_reward: float,
    stage_index: int = 1,
    action_token_ids: tuple[int, ...] = (1, 2),
) -> SampledStageTrajectory:
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
        reward_breakdown={"amplified_reward": terminal_reward},
        prefix_rewards=tuple([1.0e-4] * len(action_token_ids) + [terminal_reward]),
        terminal_reward=terminal_reward,
        stop_token=EOM_TOKEN,
        termination_reason="stop_token",
        is_valid=True,
        is_duplicate=False,
    )


def test_train_iteration_mixes_on_policy_and_replay(monkeypatch) -> None:
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
            batch_size=2,
            objective="tb",
            save_every_iterations=99,
            replay=ReplayConfig(capacity=8, replay_batch_size=1, max_total_action_tokens=32),
        ),
        device=torch.device("cpu"),
    )

    on_policy = [
        _make_sampled_trajectory(rollout_id="fresh-1", terminal_reward=2.0),
        _make_sampled_trajectory(rollout_id="fresh-2", terminal_reward=3.0, stage_index=2),
    ]
    replay_item = _make_sampled_trajectory(rollout_id="replay-1", terminal_reward=1.5)
    trainer.replay_buffer.add(replay_item)

    monkeypatch.setattr(
        trainer,
        "collect_on_policy_trajectories",
        lambda _examples, iteration_index: on_policy,
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

    metrics = trainer.train_iteration([{"id": "unused"}], iteration_index=1)

    assert metrics["num_on_policy_trajectories"] == 2.0
    assert metrics["num_replay_trajectories"] == 1.0
    assert metrics["replay_size"] == 3.0
    assert metrics["replay_total_action_tokens"] == 6.0
    assert metrics["mean_stage_reward"] == 2.5
    assert metrics["mean_stage_index"] == 1.5
    assert "objective_loss" in metrics
    assert "grad_norm" in metrics


def test_run_multi_molecule_gflownet_uses_resolved_checkpoint_source_for_all_model_loads(
    monkeypatch,
    tmp_path: Path,
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

        def log_config(self, payload) -> None:
            self.payload = payload

        def log_metrics(self, metrics, *, step: int, prefix: str) -> None:
            self.metric_calls.append((metrics, step, prefix))

        def log_summary(self, summary, *, prefix: str) -> None:
            self.summary = (summary, prefix)

        def finish(self, *, status: str) -> None:
            self.status = status

    class DummyTrainer:
        def __init__(self, **kwargs) -> None:
            self.kwargs = kwargs

        def train_iteration(self, examples, *, iteration_index: int) -> dict[str, float]:
            del examples
            return {
                "iteration": float(iteration_index),
                "objective_loss": 1.25,
                "mean_stage_reward": 2.5,
                "valid_fraction": 1.0,
                "replay_size": 0.0,
                "replay_total_action_tokens": 0.0,
            }

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
            },
        },
        project_root=tmp_path,
    )

    summary = run_multi_molecule_gflownet(config)

    assert summary["resolved_checkpoint_source"] == DEFAULT_PPO_FALLBACK_CHECKPOINT
    assert load_calls == [
        ("tokenizer", DEFAULT_PPO_FALLBACK_CHECKPOINT),
        ("model", DEFAULT_PPO_FALLBACK_CHECKPOINT),
    ]
    assert tracker.metric_calls == [
        (
            {
                "iteration": 1.0,
                "objective_loss": 1.25,
                "mean_stage_reward": 2.5,
                "valid_fraction": 1.0,
                "replay_size": 0.0,
                "replay_total_action_tokens": 0.0,
            },
            1,
            "gflownet",
        )
    ]
