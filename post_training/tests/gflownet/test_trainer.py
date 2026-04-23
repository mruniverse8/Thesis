import json
from pathlib import Path
from types import SimpleNamespace
from zipfile import ZipFile

import pytest
import torch

from src.constants import EOM_TOKEN

from post_training.gflownet.config import GFlowNetConfig, ReplayConfig
from post_training.gflownet.diagnostics import GFlowNetTrainIterationResult
from post_training.gflownet.trajectory import SampledStageTrajectory, ScoredStageTrajectory
from post_training.gflownet.trainer import MultiMoleculeGFlowNetTrainer, run_multi_molecule_gflownet
from post_training.shared.decoding import StageTokenConstraints
from post_training.shared.config import DEFAULT_PPO_FALLBACK_CHECKPOINT, resolve_gflownet_config_paths


def _make_sampled_trajectory(
    *,
    rollout_id: str,
    terminal_reward: float,
    stage_index: int = 1,
    action_token_ids: tuple[int, ...] = (1, 2),
    target_selfies_list: tuple[str, ...] = ("[C][C][O]",),
    termination_reason: str = "stop_token",
    is_valid: bool = True,
) -> SampledStageTrajectory:
    return SampledStageTrajectory(
        rollout_id=rollout_id,
        example_id=f"example-{rollout_id}",
        prompt_text="prompt",
        description="description",
        target_selfies_list=target_selfies_list,
        stage_index=stage_index,
        decoder_prefix_text="",
        previous_sampled_selfies=(),
        stage_text="<bom>[C][C][O]<eom>",
        sampled_selfies="[C][C][O]",
        action_token_ids=action_token_ids,
        reward_breakdown={"amplified_reward": terminal_reward},
        prefix_rewards=tuple([1.0e-4] * len(action_token_ids) + [terminal_reward]),
        terminal_reward=terminal_reward,
        stop_token=EOM_TOKEN,
        termination_reason=termination_reason,
        is_valid=is_valid,
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
            batch_size=3,
            objective="tb",
            save_every_iterations=99,
            diagnostic_log_every_iterations=1,
            trajectory_preview_every_iterations=1,
            trajectory_preview_num_samples=3,
            replay=ReplayConfig(
                capacity=8,
                replay_fraction=0.2,
                max_total_action_tokens=32,
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

    result = trainer.train_iteration([{"id": "unused"}], iteration_index=1)
    metrics = result.metrics

    assert metrics["num_on_policy_trajectories"] == 4.0
    assert metrics["num_replay_trajectories"] == 1.0
    assert metrics["configured_replay_fraction"] == pytest.approx(0.2)
    assert metrics["replay_fraction"] == pytest.approx(0.2)
    assert metrics["replay_buffer_type"] == "priority"
    assert metrics["replay_top_reward_count"] == pytest.approx(1.0)
    assert metrics["replay_hard_positive_count"] == pytest.approx(0.0)
    assert metrics["replay_hard_negative_count"] == pytest.approx(0.0)
    assert metrics["replay_size"] == 5.0
    assert metrics["replay_total_action_tokens"] == 12.0
    assert metrics["rollout_append_probability"] == pytest.approx(0.30)
    assert metrics["mean_stage_reward"] == pytest.approx((2.0 + 3.0 + 1.0e-4 + 4.0) / 4.0)
    assert metrics["valid_fraction"] == pytest.approx(0.75)
    assert metrics["mean_num_actions"] == pytest.approx(2.5)
    assert metrics["max_num_actions"] == pytest.approx(4.0)
    assert metrics["mean_stage_index"] == pytest.approx(1.25)
    assert metrics["termination_fraction_stop_token"] == pytest.approx(0.75)
    assert metrics["termination_fraction_max_stage_new_tokens"] == pytest.approx(0.25)
    assert metrics["num_rollouts"] == pytest.approx(3.0)
    assert metrics["max_stage_index"] == pytest.approx(2.0)
    assert metrics["mean_planned_stage_count"] == pytest.approx(8.0)
    assert metrics["max_planned_stage_count"] == pytest.approx(8.0)
    assert metrics["mean_realized_stage_count"] == pytest.approx(4.0 / 3.0)
    assert metrics["max_realized_stage_count"] == pytest.approx(2.0)
    assert metrics["fraction_rollouts_planned_stage_2_plus"] == pytest.approx(1.0)
    assert metrics["fraction_rollouts_reaching_stage_2"] == pytest.approx(1.0 / 3.0)
    assert metrics["fraction_rollouts_reaching_planned_stage_count"] == pytest.approx(0.0)
    assert metrics["rollout_stage_count_1_fraction"] == pytest.approx(2.0 / 3.0)
    assert metrics["rollout_stage_count_2_fraction"] == pytest.approx(1.0 / 3.0)
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
    assert "action_tokens_per_sec" in metrics
    assert "mean_log_pf_token" in metrics
    assert result.diagnostic_metrics is not None
    assert result.categorized_diagnostic_metrics is not None
    assert result.categorized_diagnostic_metrics["sec_timer"]["sampling_duration_sec"] > 0.0
    assert result.categorized_diagnostic_metrics["stage_rollout"]["mean_realized_stage_count"] == pytest.approx(
        4.0 / 3.0
    )
    assert result.trajectory_preview is not None
    assert len(result.trajectory_preview["records"]) == 3
    assert {record["preview_slot"] for record in result.trajectory_preview["records"]} == {
        "best",
        "median",
        "worst",
    }


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

        def train_iteration(self, examples, *, iteration_index: int) -> GFlowNetTrainIterationResult:
            del examples
            self.best_objective_loss = 1.25
            self.best_checkpoint_iteration = iteration_index
            self.best_checkpoint_dir = str(output_dir / "checkpoints" / "best")
            self.best_checkpoint_zip = str(output_dir / "checkpoints" / "best.zip")
            return GFlowNetTrainIterationResult(
                metrics={
                    "iteration": float(iteration_index),
                    "objective_loss": 1.25,
                    "mean_stage_reward": 2.5,
                    "valid_fraction": 1.0,
                    "replay_size": 0.0,
                    "replay_total_action_tokens": 0.0,
                    "mean_realized_stage_count": 1.0,
                    "fraction_rollouts_reaching_stage_2": 0.0,
                },
                diagnostic_metrics={
                    "iteration": float(iteration_index),
                    "grad_norm": 0.5,
                    "sampling_duration_sec": 0.25,
                    "mean_realized_stage_count": 1.0,
                    "fraction_rollouts_reaching_stage_2": 0.0,
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
            "valid_fraction": 1.0,
            "replay_size": 0.0,
            "replay_total_action_tokens": 0.0,
        }
    ]
    assert diagnostic_calls == [
        {
            "iteration": 1.0,
            "grad_norm": 0.5,
            "sampling_duration_sec": 0.25,
            "mean_realized_stage_count": 1.0,
            "fraction_rollouts_reaching_stage_2": 0.0,
        }
    ]
    assert categorized_calls == {
        "gflownet_diagnostics_stage_rollout": {
            "mean_realized_stage_count": 1.0,
            "fraction_rollouts_reaching_stage_2": 0.0,
        },
        "gflownet_diagnostics_sec_timer": {
            "sampling_duration_sec": 0.25,
        },
        "gflownet_diagnostics_numerics": {
            "grad_norm": 0.5,
        },
    }
    assert "mean_realized_stage_count" not in headline_calls[0]
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
    trajectory_previews = diagnostics_dir / "trajectory_previews.jsonl"
    assert iteration_diagnostics.exists()
    assert iteration_diagnostics_categorized.exists()
    assert trajectory_previews.exists()
    assert [json.loads(line) for line in iteration_diagnostics.read_text().splitlines()] == [
        {
            "iteration": 1.0,
            "grad_norm": 0.5,
            "sampling_duration_sec": 0.25,
            "mean_realized_stage_count": 1.0,
            "fraction_rollouts_reaching_stage_2": 0.0,
        }
    ]
    assert [json.loads(line) for line in iteration_diagnostics_categorized.read_text().splitlines()] == [
        {
            "iteration": 1.0,
            "categories": {
                "stage_rollout": {
                    "mean_realized_stage_count": 1.0,
                    "fraction_rollouts_reaching_stage_2": 0.0,
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
