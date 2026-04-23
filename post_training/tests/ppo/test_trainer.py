import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from post_training.ppo.config import PPOConfig, StageTrajectory
from post_training.ppo.trainer import (
    PPOTrainIterationResult,
    MoleculeWisePPOTrainer,
    build_trajectory_preview_payload,
    run_molecule_stage_ppo,
    standardize_tensor,
)
from post_training.shared.decoding import StageTokenConstraints
from post_training.shared.config import (
    DEFAULT_PPO_FALLBACK_CHECKPOINT,
    resolve_ppo_checkpoint_source,
    resolve_ppo_config_paths,
)


def test_standardize_tensor_centers_values() -> None:
    standardized = standardize_tensor(torch.tensor([1.0, 2.0, 3.0]))

    assert standardized.mean().abs().item() < 1.0e-6


def test_ppo_config_from_dict_reads_rollout_section_and_ignores_old_separator_key() -> None:
    config = PPOConfig.from_dict(
        {
            "output_dir": "outputs/test",
            "target_modules": ["q", "v"],
            "rollout": {
                "max_stage_new_tokens": 16,
                "max_molecules_per_sequence": 4,
                "append_probability": 0.6,
                "molecule_separator_token": "<mol_sep>",
            },
        }
    )

    assert config.output_dir == "outputs/test"
    assert config.rollout.max_stage_new_tokens == 16
    assert config.rollout.max_molecules_per_sequence == 4
    assert config.rollout.append_probability == pytest.approx(0.6)
    assert config.rollout.stage_separator == " "


def test_ppo_config_from_dict_supports_diagnostic_logging_fields() -> None:
    default_config = PPOConfig.from_dict({})
    explicit_config = PPOConfig.from_dict(
        {
            "diagnostic_log_every_ppo_epochs": 3,
            "trajectory_preview_every_iterations": 7,
            "num_trajectory_samples_to_log": 5,
            "trajectory_preview_max_chars": 180,
        }
    )

    assert default_config.diagnostic_log_every_ppo_epochs == 1
    assert default_config.trajectory_preview_every_iterations == 25
    assert default_config.num_trajectory_samples_to_log == 3
    assert default_config.trajectory_preview_max_chars == 240
    assert explicit_config.diagnostic_log_every_ppo_epochs == 3
    assert explicit_config.trajectory_preview_every_iterations == 7
    assert explicit_config.num_trajectory_samples_to_log == 5
    assert explicit_config.trajectory_preview_max_chars == 180


def test_ppo_config_from_dict_clamps_epoch_diagnostic_logging_cadence() -> None:
    config = PPOConfig.from_dict({"diagnostic_log_every_ppo_epochs": 0})

    assert config.diagnostic_log_every_ppo_epochs == 1


def test_ppo_rollout_defaults_enable_constrained_decoding_and_tighter_sampling() -> None:
    config = PPOConfig.from_dict({})

    assert config.rollout.temperature == 0.8
    assert config.rollout.top_p == 0.95
    assert config.rollout.constrained_decoding is True
    assert config.rollout.append_probability == pytest.approx(0.30)
    assert config.rollout.selfies_dict_path == "molecules/dict/selfies_dict.txt"


def _make_reward_breakdown(
    *,
    amplified_reward: float,
    total_reward: float | None = None,
    match_reward: float = 0.0,
    diversity_reward: float = 0.0,
    is_duplicate: bool = False,
    is_valid: bool = True,
):
    return SimpleNamespace(
        amplified_reward=amplified_reward,
        total_reward=amplified_reward if total_reward is None else total_reward,
        match=SimpleNamespace(reward=match_reward),
        diversity=SimpleNamespace(reward=diversity_reward),
        is_duplicate=is_duplicate,
        candidate=SimpleNamespace(is_valid=is_valid, canonical_smiles="C"),
    )


def _make_stage_trajectory(
    *,
    rollout_id: str,
    example_id: str,
    stage_index: int,
    reward: float,
    old_value: float,
    old_logprob: float,
    reference_logprob: float,
    action_token_ids: tuple[int, ...],
    sampled_selfies: str | None,
    stage_text: str,
    termination_reason: str = "stop_token",
    is_valid: bool = True,
    is_duplicate: bool = False,
) -> StageTrajectory:
    return StageTrajectory(
        rollout_id=rollout_id,
        example_id=example_id,
        prompt_text="prompt",
        description="description",
        target_selfies_list=("target",),
        stage_index=stage_index,
        decoder_prefix_text="",
        stage_text=stage_text,
        sampled_selfies=sampled_selfies,
        stop_token="<eom>" if termination_reason == "stop_token" else None,
        termination_reason=termination_reason,
        action_token_ids=action_token_ids,
        action_logprob_sum_old=old_logprob,
        reference_logprob_sum=reference_logprob,
        value_old=old_value,
        reward_breakdown=_make_reward_breakdown(
            amplified_reward=reward,
            total_reward=reward,
            match_reward=reward / 2,
            diversity_reward=reward / 4,
            is_duplicate=is_duplicate,
            is_valid=is_valid,
        ),
        reward=reward,
        entropy_sum_old=0.0,
        is_valid=is_valid,
        is_duplicate=is_duplicate,
    )


def test_build_trajectory_preview_payload_selects_best_median_and_worst_rollouts() -> None:
    trajectories = [
        _make_stage_trajectory(
            rollout_id="rollout-1",
            example_id="shared-example",
            stage_index=1,
            reward=0.5,
            old_value=0.0,
            old_logprob=-1.0,
            reference_logprob=-1.1,
            action_token_ids=(1,),
            sampled_selfies="A",
            stage_text="A",
        ),
        _make_stage_trajectory(
            rollout_id="rollout-2",
            example_id="shared-example",
            stage_index=1,
            reward=1.0,
            old_value=0.0,
            old_logprob=-0.5,
            reference_logprob=-0.7,
            action_token_ids=(2,),
            sampled_selfies="B",
            stage_text="B",
        ),
        _make_stage_trajectory(
            rollout_id="rollout-2",
            example_id="shared-example",
            stage_index=2,
            reward=2.0,
            old_value=0.0,
            old_logprob=-0.4,
            reference_logprob=-0.6,
            action_token_ids=(4,),
            sampled_selfies="B2",
            stage_text="B2",
        ),
        _make_stage_trajectory(
            rollout_id="rollout-3",
            example_id="other-example",
            stage_index=1,
            reward=1.5,
            old_value=0.0,
            old_logprob=-0.8,
            reference_logprob=-0.9,
            action_token_ids=(3,),
            sampled_selfies="C",
            stage_text="C",
        ),
    ]

    preview = build_trajectory_preview_payload(
        trajectories,
        iteration_index=25,
        num_samples=3,
        max_chars=120,
    )

    assert preview is not None
    assert [record["preview_slot"] for record in preview["records"]] == ["best", "median", "worst"]
    assert [record["rollout_id"] for record in preview["records"]] == [
        "rollout-2",
        "rollout-3",
        "rollout-1",
    ]
    assert preview["records"][0]["num_stages"] == 2
    assert "num_stages=2" in preview["tracker_text"]
    assert "generated_selfies=B | B2" in preview["tracker_text"]
    assert "generated_selfies=C" in preview["tracker_text"]
    assert "generated_selfies=A" in preview["tracker_text"]


def _build_dummy_ppo_trainer(config: PPOConfig) -> MoleculeWisePPOTrainer:
    class DummyPolicyValueModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(1.0))
            self.policy_model = SimpleNamespace(config=SimpleNamespace(decoder_start_token_id=0))

        def to(self, device):
            return super().to(device)

        def save_checkpoint(self, *args, **kwargs) -> None:
            del args, kwargs
            return None

    policy_model = DummyPolicyValueModel()
    reference_model = torch.nn.Linear(1, 1)
    return MoleculeWisePPOTrainer(
        policy_model=policy_model,
        reference_model=reference_model,
        tokenizer=None,
        config=config,
        device=torch.device("cpu"),
    )


def _build_iteration_test_trajectories() -> list[StageTrajectory]:
    return [
        _make_stage_trajectory(
            rollout_id="rollout-a",
            example_id="example-a",
            stage_index=1,
            reward=1.0,
            old_value=0.2,
            old_logprob=-1.2,
            reference_logprob=-1.1,
            action_token_ids=(1, 2),
            sampled_selfies="AA",
            stage_text="AA",
        ),
        _make_stage_trajectory(
            rollout_id="rollout-b",
            example_id="example-b",
            stage_index=1,
            reward=2.0,
            old_value=0.4,
            old_logprob=-0.9,
            reference_logprob=-0.8,
            action_token_ids=(3,),
            sampled_selfies="BB",
            stage_text="BB",
            termination_reason="max_stage_new_tokens",
        ),
        _make_stage_trajectory(
            rollout_id="rollout-c",
            example_id="example-c",
            stage_index=1,
            reward=3.0,
            old_value=0.6,
            old_logprob=-0.7,
            reference_logprob=-0.6,
            action_token_ids=(),
            sampled_selfies=None,
            stage_text="",
            termination_reason="max_sequence_length",
            is_valid=False,
            is_duplicate=True,
        ),
    ]


def _patch_iteration_statistics(
    monkeypatch: pytest.MonkeyPatch,
    trainer: MoleculeWisePPOTrainer,
    trajectories: list[StageTrajectory],
) -> None:
    coefficients = {
        "rollout-a": (0.9, 0.2, 0.5),
        "rollout-b": (0.7, 0.3, 0.4),
        "rollout-c": (0.4, 0.1, 0.2),
    }

    monkeypatch.setattr(trainer, "collect_rollouts", lambda _examples: trajectories)

    def fake_compute_stage_statistics(trajectory: StageTrajectory):
        logprob_coef, entropy_coef, value_coef = coefficients[trajectory.rollout_id]
        base = trainer.policy_model.weight
        return (
            base * logprob_coef,
            base * entropy_coef,
            base * value_coef,
        )

    monkeypatch.setattr(
        trainer,
        "_compute_stage_statistics",
        fake_compute_stage_statistics,
    )


def test_train_iteration_returns_epoch_and_optimizer_diagnostics_and_preview(
    monkeypatch,
    capsys,
) -> None:
    config = PPOConfig(
        ppo_iterations=3,
        batch_size=3,
        mini_batch_size=2,
        ppo_epochs_per_batch=2,
        diagnostic_log_every_ppo_epochs=1,
        trajectory_preview_every_iterations=1,
        num_trajectory_samples_to_log=2,
        save_every_iterations=99,
    )
    trainer = _build_dummy_ppo_trainer(config)
    trajectories = _build_iteration_test_trajectories()
    _patch_iteration_statistics(monkeypatch, trainer, trajectories)

    result = trainer.train_iteration([{"id": "unused"}], iteration_index=1)
    captured = capsys.readouterr()

    assert {"reward_std", "mean_old_logprob", "termination_fraction_max_sequence_length"} <= set(
        result.metrics
    )
    assert abs(result.metrics["standardized_advantage_mean"]) < 1.0e-6
    assert result.metrics["max_action_token_count"] == 2.0
    assert result.metrics["empty_action_rate"] == 1.0 / 3.0
    assert result.diagnostic_metrics is not None
    assert result.diagnostic_metrics["num_stage_trajectories"] == 3.0
    assert result.diagnostic_metrics["mean_realized_stage_count"] == 1.0
    assert result.categorized_diagnostic_metrics is not None
    assert result.categorized_diagnostic_metrics["stage_rollout"]["num_stage_trajectories"] == 3.0
    assert result.categorized_diagnostic_metrics["stage_rollout"]["mean_realized_stage_count"] == 1.0
    assert result.categorized_diagnostic_metrics["reward_only"]["mean_total_reward"] == 2.0
    assert len(result.epoch_metrics) == 2
    assert [metrics["ppo_epoch_in_iteration"] for metrics in result.epoch_metrics] == [1, 2]
    assert all(metrics["ppo_iteration"] == 1 for metrics in result.epoch_metrics)
    assert all(metrics["num_stage_trajectories"] == 3.0 for metrics in result.epoch_metrics)
    assert result.epoch_metrics[0]["optimizer_steps_completed_in_iteration"] == 2
    assert result.epoch_metrics[1]["optimizer_steps_completed_in_iteration"] == 4
    assert result.categorized_epoch_metrics[0]["stage_rollout"]["num_rollouts"] == 3.0
    assert result.categorized_epoch_metrics[0]["optimizer"]["mean_policy_loss"] > 0.0
    assert len(result.optimizer_step_metrics) == 4
    assert [metrics["optimizer_step"] for metrics in result.optimizer_step_metrics] == [1, 2, 3, 4]
    assert [metrics["optimizer_step_in_iteration"] for metrics in result.optimizer_step_metrics] == [
        1,
        2,
        3,
        4,
    ]
    assert [metrics["ppo_epoch_in_iteration"] for metrics in result.optimizer_step_metrics] == [
        1,
        1,
        2,
        2,
    ]
    assert [metrics["mini_batch_size"] for metrics in result.optimizer_step_metrics] == [2, 1, 2, 1]
    assert all(metrics["all_finite"] is True for metrics in result.optimizer_step_metrics)
    assert result.categorized_optimizer_step_metrics[0]["optimizer"]["mini_batch_size"] == 2
    assert result.categorized_optimizer_step_metrics[0]["numerics"]["all_finite"] is True
    assert "clip_fraction" in result.optimizer_step_metrics[0]
    assert result.trajectory_preview is not None
    assert len(result.trajectory_preview["records"]) == 2
    assert all(record["num_stages"] == 1 for record in result.trajectory_preview["records"])
    assert "[ppo][iteration 1/3][epoch 1/2] optimizer steps this epoch=2 total_optimizer_step=2" in captured.out
    assert "[ppo][iteration 1/3][epoch 2/2] optimizer steps this epoch=2 total_optimizer_step=4" in captured.out


def test_train_iteration_emits_batch_step_diagnostics_for_every_optimizer_step(
    monkeypatch,
) -> None:
    config = PPOConfig(
        ppo_iterations=2,
        batch_size=3,
        mini_batch_size=2,
        ppo_epochs_per_batch=3,
        diagnostic_log_every_ppo_epochs=1,
        trajectory_preview_every_iterations=99,
        save_every_iterations=99,
    )
    trainer = _build_dummy_ppo_trainer(config)
    trajectories = _build_iteration_test_trajectories()
    _patch_iteration_statistics(monkeypatch, trainer, trajectories)

    result = trainer.train_iteration([{"id": "unused"}], iteration_index=1)

    assert [metrics["ppo_epoch_in_iteration"] for metrics in result.epoch_metrics] == [1, 2, 3]
    assert [metrics["optimizer_step"] for metrics in result.optimizer_step_metrics] == [1, 2, 3, 4, 5, 6]
    assert [metrics["ppo_epoch_in_iteration"] for metrics in result.optimizer_step_metrics] == [
        1,
        1,
        2,
        2,
        3,
        3,
    ]


def test_resolve_ppo_checkpoint_source_prefers_existing_local_checkpoint(tmp_path: Path) -> None:
    checkpoint_dir = tmp_path / "outputs" / "multi_molecule_sft_mini" / "checkpoints" / "best"
    checkpoint_dir.mkdir(parents=True)

    resolved = resolve_ppo_checkpoint_source(
        "outputs/multi_molecule_sft_mini/checkpoints/best",
        project_root=tmp_path,
    )

    assert resolved == str(checkpoint_dir.resolve())


def test_resolve_ppo_checkpoint_source_preserves_remote_model_id() -> None:
    resolved = resolve_ppo_checkpoint_source(DEFAULT_PPO_FALLBACK_CHECKPOINT)

    assert resolved == DEFAULT_PPO_FALLBACK_CHECKPOINT


def test_resolve_ppo_config_paths_falls_back_to_original_weights_for_missing_checkpoint(
    tmp_path: Path,
) -> None:
    config = resolve_ppo_config_paths(
        {
            "model": {"checkpoint": "outputs/multi_molecule_sft_mini/checkpoints/best"},
            "data": {"train_file": "data/train.jsonl"},
            "training": {"output_dir": "outputs/molecule_wise_ppo"},
        },
        project_root=tmp_path,
    )

    assert config["model"]["checkpoint"] == DEFAULT_PPO_FALLBACK_CHECKPOINT
    assert config["data"]["train_file"] == str((tmp_path / "data" / "train.jsonl").resolve())


def test_run_molecule_stage_ppo_uses_resolved_checkpoint_source_for_all_model_loads(
    monkeypatch,
    tmp_path: Path,
) -> None:
    load_calls: list[tuple[str, str]] = []
    output_dir = tmp_path / "outputs" / "molecule_wise_ppo"

    class DummyTokenizer:
        def __init__(self) -> None:
            self.model_max_length = 0

    class DummyPolicyValueModel:
        def __init__(self) -> None:
            self.policy_model = object()

        def to(self, device) -> None:
            self.device = device

    class DummyReferenceModel:
        def to(self, device) -> None:
            self.device = device

    class DummyTracker:
        def log_config(self, payload) -> None:
            self.payload = payload

        def log_metrics(self, metrics, *, step: int, prefix: str) -> None:
            self.metrics = (metrics, step, prefix)

        def log_summary(self, summary, *, prefix: str) -> None:
            self.summary = (summary, prefix)

        def finish(self, *, status: str) -> None:
            self.status = status

    class DummyTrainer:
        def __init__(self, **kwargs) -> None:
            self.kwargs = kwargs

    monkeypatch.setattr(
        "post_training.ppo.trainer.AutoTokenizer",
        SimpleNamespace(
            from_pretrained=lambda source, use_fast=True: (
                load_calls.append(("tokenizer", source)),
                DummyTokenizer(),
            )[1]
        ),
    )
    monkeypatch.setattr(
        "post_training.ppo.trainer.PolicyValueModel",
        SimpleNamespace(
            from_pretrained=lambda source, **kwargs: (
                load_calls.append(("policy", source)),
                DummyPolicyValueModel(),
            )[1]
        ),
    )
    monkeypatch.setattr(
        "post_training.ppo.trainer.load_reference_model",
        lambda source: (
            load_calls.append(("reference", source)),
            DummyReferenceModel(),
        )[1],
    )
    monkeypatch.setattr(
        "post_training.ppo.trainer.assert_checkpoint_tokenizer_matches_model",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        "post_training.ppo.trainer.build_reward_config",
        lambda *args, **kwargs: SimpleNamespace(),
    )
    monkeypatch.setattr(
        "post_training.ppo.trainer.choose_device",
        lambda *_args, **_kwargs: torch.device("cpu"),
    )
    monkeypatch.setattr(
        "post_training.ppo.trainer.MoleculeWisePPOTrainer",
        DummyTrainer,
    )
    monkeypatch.setattr(
        "post_training.ppo.trainer.MultiMoleculeDataset",
        SimpleNamespace(from_jsonl=lambda path: []),
    )
    monkeypatch.setattr(
        "post_training.ppo.trainer.prepare_ppo_output_dir",
        lambda *_args, **_kwargs: output_dir,
    )
    monkeypatch.setattr(
        "post_training.ppo.trainer.build_tracker",
        lambda *args, **kwargs: DummyTracker(),
    )

    config = resolve_ppo_config_paths(
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
                "output_dir": "outputs/molecule_wise_ppo",
                "device": "cpu",
                "save_every_iterations": 10,
            },
            "ppo": {
                "ppo_iterations": 0,
                "batch_size": 1,
                "mini_batch_size": 1,
                "ppo_epochs_per_batch": 1,
                "rollout": {"constrained_decoding": False},
            },
        },
        project_root=tmp_path,
    )

    summary = run_molecule_stage_ppo(config)

    assert summary["resolved_checkpoint_source"] == DEFAULT_PPO_FALLBACK_CHECKPOINT
    assert load_calls == [
        ("tokenizer", DEFAULT_PPO_FALLBACK_CHECKPOINT),
        ("policy", DEFAULT_PPO_FALLBACK_CHECKPOINT),
        ("reference", DEFAULT_PPO_FALLBACK_CHECKPOINT),
    ]


def test_run_molecule_stage_ppo_builds_constraints_once_and_attaches_to_policy_model(
    monkeypatch,
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "outputs" / "molecule_wise_ppo"
    built_constraints = StageTokenConstraints(
        bom_token_id=1,
        eom_token_id=2,
        content_token_ids=(3, 4),
    )
    builder_calls: list[tuple[str, str]] = []

    class DummyTokenizer:
        def __init__(self) -> None:
            self.model_max_length = 0

    class DummyPolicyValueModel:
        def __init__(self) -> None:
            self.policy_model = object()
            self.constraints: list[StageTokenConstraints | None] = []

        def to(self, device) -> None:
            self.device = device

        def set_stage_token_constraints(self, constraints) -> None:
            self.constraints.append(constraints)

        def get_stage_token_constraints(self):
            return self.constraints[-1] if self.constraints else None

    class DummyReferenceModel:
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
        policy_model = None

        def __init__(self, **kwargs) -> None:
            type(self).policy_model = kwargs["policy_model"]

    policy_model = DummyPolicyValueModel()

    monkeypatch.setattr(
        "post_training.ppo.trainer.AutoTokenizer",
        SimpleNamespace(from_pretrained=lambda *args, **kwargs: DummyTokenizer()),
    )
    monkeypatch.setattr(
        "post_training.ppo.trainer.PolicyValueModel",
        SimpleNamespace(from_pretrained=lambda *args, **kwargs: policy_model),
    )
    monkeypatch.setattr(
        "post_training.ppo.trainer.load_reference_model",
        lambda *args, **kwargs: DummyReferenceModel(),
    )
    monkeypatch.setattr(
        "post_training.ppo.trainer.assert_checkpoint_tokenizer_matches_model",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        "post_training.ppo.trainer.build_reward_config",
        lambda *args, **kwargs: SimpleNamespace(),
    )
    monkeypatch.setattr(
        "post_training.ppo.trainer.choose_device",
        lambda *_args, **_kwargs: torch.device("cpu"),
    )
    monkeypatch.setattr(
        "post_training.ppo.trainer.MoleculeWisePPOTrainer",
        DummyTrainer,
    )
    monkeypatch.setattr(
        "post_training.ppo.trainer.MultiMoleculeDataset",
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
        "post_training.ppo.trainer.prepare_ppo_output_dir",
        lambda *_args, **_kwargs: output_dir,
    )
    monkeypatch.setattr(
        "post_training.ppo.trainer.build_tracker",
        lambda *args, **kwargs: DummyTracker(),
    )
    monkeypatch.setattr(
        "post_training.ppo.trainer.build_stage_token_constraints",
        lambda _tokenizer, _dataset, *, selfies_dict_path, separator_token: (
            builder_calls.append((selfies_dict_path, separator_token)),
            built_constraints,
        )[1],
    )

    config = resolve_ppo_config_paths(
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
            "ppo": {
                "ppo_iterations": 0,
                "batch_size": 1,
                "mini_batch_size": 1,
                "ppo_epochs_per_batch": 1,
                "rollout": {
                    "constrained_decoding": True,
                    "selfies_dict_path": "custom_selfies_dict.txt",
                },
            },
        },
        project_root=tmp_path,
    )

    run_molecule_stage_ppo(config)

    assert builder_calls == [("custom_selfies_dict.txt", " ")]
    assert policy_model.constraints == [built_constraints]
    assert DummyTrainer.policy_model.get_stage_token_constraints() is built_constraints


def test_run_molecule_stage_ppo_skips_constraint_initialization_when_disabled(
    monkeypatch,
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "outputs" / "molecule_wise_ppo"

    class DummyTokenizer:
        def __init__(self) -> None:
            self.model_max_length = 0

    class DummyPolicyValueModel:
        def __init__(self) -> None:
            self.policy_model = object()
            self.constraints: list[StageTokenConstraints | None] = []

        def to(self, device) -> None:
            self.device = device

        def set_stage_token_constraints(self, constraints) -> None:
            self.constraints.append(constraints)

    class DummyReferenceModel:
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

    policy_model = DummyPolicyValueModel()

    monkeypatch.setattr(
        "post_training.ppo.trainer.AutoTokenizer",
        SimpleNamespace(from_pretrained=lambda *args, **kwargs: DummyTokenizer()),
    )
    monkeypatch.setattr(
        "post_training.ppo.trainer.PolicyValueModel",
        SimpleNamespace(from_pretrained=lambda *args, **kwargs: policy_model),
    )
    monkeypatch.setattr(
        "post_training.ppo.trainer.load_reference_model",
        lambda *args, **kwargs: DummyReferenceModel(),
    )
    monkeypatch.setattr(
        "post_training.ppo.trainer.assert_checkpoint_tokenizer_matches_model",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        "post_training.ppo.trainer.build_reward_config",
        lambda *args, **kwargs: SimpleNamespace(),
    )
    monkeypatch.setattr(
        "post_training.ppo.trainer.choose_device",
        lambda *_args, **_kwargs: torch.device("cpu"),
    )
    monkeypatch.setattr(
        "post_training.ppo.trainer.MoleculeWisePPOTrainer",
        DummyTrainer,
    )
    monkeypatch.setattr(
        "post_training.ppo.trainer.MultiMoleculeDataset",
        SimpleNamespace(from_jsonl=lambda path: []),
    )
    monkeypatch.setattr(
        "post_training.ppo.trainer.prepare_ppo_output_dir",
        lambda *_args, **_kwargs: output_dir,
    )
    monkeypatch.setattr(
        "post_training.ppo.trainer.build_tracker",
        lambda *args, **kwargs: DummyTracker(),
    )
    monkeypatch.setattr(
        "post_training.ppo.trainer.build_stage_token_constraints",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("builder should not be called when constrained decoding is disabled")
        ),
    )

    config = resolve_ppo_config_paths(
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
            "ppo": {
                "ppo_iterations": 0,
                "batch_size": 1,
                "mini_batch_size": 1,
                "ppo_epochs_per_batch": 1,
                "rollout": {"constrained_decoding": False},
            },
        },
        project_root=tmp_path,
    )

    run_molecule_stage_ppo(config)

    assert policy_model.constraints == [None]


def test_run_molecule_stage_ppo_logs_iteration_epoch_and_batch_step_diagnostics_and_preview(
    monkeypatch,
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "outputs" / "molecule_wise_ppo"

    class DummyTokenizer:
        def __init__(self) -> None:
            self.model_max_length = 0

    class DummyPolicyValueModel:
        def __init__(self) -> None:
            self.policy_model = object()

        def to(self, device) -> None:
            self.device = device

    class DummyReferenceModel:
        def to(self, device) -> None:
            self.device = device

    class DummyTracker:
        def __init__(self) -> None:
            self.metric_calls: list[tuple[dict[str, object], int, str]] = []
            self.summary_calls: list[tuple[dict[str, object], str]] = []

        def log_config(self, payload) -> None:
            self.config = payload

        def log_metrics(self, metrics, *, step: int, prefix: str) -> None:
            self.metric_calls.append((metrics, step, prefix))

        def log_summary(self, summary, *, prefix: str) -> None:
            self.summary_calls.append((summary, prefix))

        def finish(self, *, status: str) -> None:
            self.status = status

    class DummyTrainer:
        def __init__(self, **kwargs) -> None:
            self.kwargs = kwargs

        def train_iteration(self, examples, *, iteration_index: int) -> PPOTrainIterationResult:
            del examples
            return PPOTrainIterationResult(
                metrics={
                    "iteration": float(iteration_index),
                    "mean_reward": 1.5,
                    "mean_kl": 0.1,
                    "mean_entropy": 0.2,
                    "mean_policy_loss": 0.3,
                    "mean_value_loss": 0.4,
                    "num_stage_trajectories": 2.0,
                    "num_rollouts": 1.0,
                    "mean_realized_stage_count": 1.5,
                    "fraction_rollouts_reaching_stage_2": 0.5,
                },
                diagnostic_metrics={
                    "iteration": float(iteration_index),
                    "mean_reward": 1.5,
                    "mean_kl": 0.1,
                    "mean_entropy": 0.2,
                    "mean_policy_loss": 0.3,
                    "mean_value_loss": 0.4,
                    "num_stage_trajectories": 2.0,
                    "num_rollouts": 1.0,
                    "mean_realized_stage_count": 1.5,
                    "fraction_rollouts_reaching_stage_2": 0.5,
                },
                epoch_metrics=[
                    {
                        "ppo_iteration": iteration_index,
                        "ppo_epoch_in_iteration": 1,
                        "ppo_epochs_per_batch": 2,
                        "optimizer_steps_completed_in_iteration": 2,
                        "mean_policy_loss": 0.35,
                        "mean_value_loss": 0.45,
                        "mean_kl": 0.15,
                        "mean_entropy": 0.25,
                        "num_stage_trajectories": 2.0,
                        "num_rollouts": 1.0,
                        "stage1_num_trajectories": 1.0,
                        "stage2_num_trajectories": 1.0,
                        "stage3_num_trajectories": 0.0,
                        "grad_norm": 0.8,
                        "all_finite": True,
                    },
                    {
                        "ppo_iteration": iteration_index,
                        "ppo_epoch_in_iteration": 2,
                        "ppo_epochs_per_batch": 2,
                        "optimizer_steps_completed_in_iteration": 4,
                        "mean_policy_loss": 0.3,
                        "mean_value_loss": 0.4,
                        "mean_kl": 0.1,
                        "mean_entropy": 0.2,
                        "num_stage_trajectories": 2.0,
                        "num_rollouts": 1.0,
                        "stage1_num_trajectories": 1.0,
                        "stage2_num_trajectories": 1.0,
                        "stage3_num_trajectories": 0.0,
                        "grad_norm": 0.7,
                        "all_finite": True,
                    },
                ],
                optimizer_step_metrics=[
                    {
                        "optimizer_step": 25,
                        "optimizer_step_in_iteration": 4,
                        "ppo_iteration": iteration_index,
                        "ppo_epoch_in_iteration": 2,
                        "mini_batch_size": 4,
                        "policy_loss": 0.3,
                        "value_loss": 0.4,
                        "total_loss": 0.5,
                        "entropy_bonus": 0.2,
                        "approx_kl_mean": 0.1,
                        "ratio_mean": 1.0,
                        "ratio_std": 0.05,
                        "clip_fraction": 0.0,
                        "batch_advantage_mean": 0.0,
                        "batch_advantage_std": 1.0,
                        "batch_return_mean": 1.5,
                        "new_value_mean": 1.2,
                        "grad_norm": 0.9,
                        "all_finite": True,
                    }
                ],
                trajectory_preview={
                    "records": [
                        {
                            "iteration": iteration_index,
                            "preview_slot": "best",
                            "rollout_id": "sample-0000-example-1",
                            "example_id": "example-1",
                            "num_stages": 2,
                            "stage_indices": [1, 2],
                            "total_reward": 3.0,
                            "stage_rewards": [1.0, 2.0],
                            "generated_selfies_sequence": ["A", "B"],
                            "raw_stage_text_sequence": ["A", "B"],
                            "valid_sequence": [True, True],
                            "duplicate_sequence": [False, False],
                            "termination_reasons": ["stop_token", "stop_token"],
                        }
                    ],
                    "tracker_text": "preview text",
                },
            )

    tracker = DummyTracker()

    monkeypatch.setattr(
        "post_training.ppo.trainer.AutoTokenizer",
        SimpleNamespace(from_pretrained=lambda *_args, **_kwargs: DummyTokenizer()),
    )
    monkeypatch.setattr(
        "post_training.ppo.trainer.PolicyValueModel",
        SimpleNamespace(from_pretrained=lambda *_args, **_kwargs: DummyPolicyValueModel()),
    )
    monkeypatch.setattr(
        "post_training.ppo.trainer.load_reference_model",
        lambda _source: DummyReferenceModel(),
    )
    monkeypatch.setattr(
        "post_training.ppo.trainer.assert_checkpoint_tokenizer_matches_model",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        "post_training.ppo.trainer.build_reward_config",
        lambda *args, **kwargs: SimpleNamespace(),
    )
    monkeypatch.setattr(
        "post_training.ppo.trainer.choose_device",
        lambda *_args, **_kwargs: torch.device("cpu"),
    )
    monkeypatch.setattr(
        "post_training.ppo.trainer.MoleculeWisePPOTrainer",
        DummyTrainer,
    )
    monkeypatch.setattr(
        "post_training.ppo.trainer.MultiMoleculeDataset",
        SimpleNamespace(
            from_jsonl=lambda path: [
                {
                    "id": "example-1",
                    "prompt": "prompt",
                    "description": "description",
                    "target_selfies_list": ["A"],
                }
            ]
        ),
    )
    monkeypatch.setattr(
        "post_training.ppo.trainer.build_tracker",
        lambda *args, **kwargs: tracker,
    )

    config = resolve_ppo_config_paths(
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
            "ppo": {
                "ppo_iterations": 1,
                "batch_size": 1,
                "mini_batch_size": 1,
                "ppo_epochs_per_batch": 2,
                "rollout": {"constrained_decoding": False},
            },
        },
        project_root=tmp_path,
    )

    summary = run_molecule_stage_ppo(config)

    iteration_diagnostics_path = output_dir / "diagnostics" / "iteration_diagnostics.jsonl"
    iteration_diagnostics_categorized_path = (
        output_dir / "diagnostics" / "iteration_diagnostics_categorized.jsonl"
    )
    epoch_diagnostics_path = output_dir / "diagnostics" / "epoch_diagnostics.jsonl"
    epoch_diagnostics_categorized_path = (
        output_dir / "diagnostics" / "epoch_diagnostics_categorized.jsonl"
    )
    optimizer_diagnostics_path = output_dir / "diagnostics" / "optimizer_step_metrics.jsonl"
    optimizer_diagnostics_categorized_path = (
        output_dir / "diagnostics" / "optimizer_step_metrics_categorized.jsonl"
    )
    trajectory_previews_path = output_dir / "diagnostics" / "trajectory_previews.jsonl"
    iteration_diagnostics_records = [
        json.loads(line)
        for line in iteration_diagnostics_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    iteration_diagnostics_categorized_records = [
        json.loads(line)
        for line in iteration_diagnostics_categorized_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    epoch_diagnostics_records = [
        json.loads(line)
        for line in epoch_diagnostics_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    epoch_diagnostics_categorized_records = [
        json.loads(line)
        for line in epoch_diagnostics_categorized_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    optimizer_records = [
        json.loads(line)
        for line in optimizer_diagnostics_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    optimizer_categorized_records = [
        json.loads(line)
        for line in optimizer_diagnostics_categorized_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    trajectory_records = [
        json.loads(line)
        for line in trajectory_previews_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    assert summary["num_iterations"] == 1
    headline_calls = [payload for payload, _, prefix in tracker.metric_calls if prefix == "ppo"]
    diagnostic_calls = [
        (payload, step) for payload, step, prefix in tracker.metric_calls if prefix == "ppo_diagnostics"
    ]
    epoch_headline_calls = [
        (payload, step) for payload, step, prefix in tracker.metric_calls if prefix == "ppo_epoch"
    ]
    epoch_diagnostic_calls = [
        (payload, step)
        for payload, step, prefix in tracker.metric_calls
        if prefix == "ppo_epoch_diagnostics"
    ]
    optimizer_calls = [
        (payload, step) for payload, step, prefix in tracker.metric_calls if prefix == "ppo_optimizer"
    ]
    categorized_diagnostic_calls = {
        (prefix, step): payload
        for payload, step, prefix in tracker.metric_calls
        if prefix.startswith("ppo_diagnostics_")
    }
    categorized_epoch_calls = {
        (prefix, step): payload
        for payload, step, prefix in tracker.metric_calls
        if prefix.startswith("ppo_epoch_diagnostics_")
    }
    categorized_optimizer_calls = {
        (prefix, step): payload
        for payload, step, prefix in tracker.metric_calls
        if prefix.startswith("ppo_optimizer_")
    }
    assert headline_calls
    assert "mean_realized_stage_count" not in headline_calls[0]
    assert diagnostic_calls == [
        (
            {
                "iteration": 1.0,
                "mean_reward": 1.5,
                "mean_kl": 0.1,
                "mean_entropy": 0.2,
                "mean_policy_loss": 0.3,
                "mean_value_loss": 0.4,
                "num_stage_trajectories": 2.0,
                "num_rollouts": 1.0,
                "mean_realized_stage_count": 1.5,
                "fraction_rollouts_reaching_stage_2": 0.5,
            },
            1,
        ),
    ]
    assert epoch_headline_calls == [
        (
            {
                "ppo_iteration": 1,
                "ppo_epoch_in_iteration": 1,
                "ppo_epochs_per_batch": 2,
                "optimizer_steps_completed_in_iteration": 2,
                "num_stage_trajectories": 2.0,
                "num_rollouts": 1.0,
                "mean_policy_loss": 0.35,
                "mean_value_loss": 0.45,
                "mean_kl": 0.15,
                "mean_entropy": 0.25,
            },
            1,
        ),
        (
            {
                "ppo_iteration": 1,
                "ppo_epoch_in_iteration": 2,
                "ppo_epochs_per_batch": 2,
                "optimizer_steps_completed_in_iteration": 4,
                "num_stage_trajectories": 2.0,
                "num_rollouts": 1.0,
                "mean_policy_loss": 0.3,
                "mean_value_loss": 0.4,
                "mean_kl": 0.1,
                "mean_entropy": 0.2,
            },
            2,
        ),
    ]
    assert epoch_diagnostic_calls[0][1] == 1
    assert epoch_diagnostic_calls[1][1] == 2
    assert epoch_diagnostic_calls[0][0]["ppo_epoch_in_iteration"] == 1
    assert epoch_diagnostic_calls[1][0]["ppo_epoch_in_iteration"] == 2
    assert categorized_diagnostic_calls == {
        ("ppo_diagnostics_reward_only", 1): {
            "mean_reward": 1.5,
        },
        ("ppo_diagnostics_stage_rollout", 1): {
            "num_stage_trajectories": 2.0,
            "num_rollouts": 1.0,
            "mean_realized_stage_count": 1.5,
            "fraction_rollouts_reaching_stage_2": 0.5,
        },
        ("ppo_diagnostics_optimizer", 1): {
            "mean_kl": 0.1,
            "mean_entropy": 0.2,
            "mean_policy_loss": 0.3,
            "mean_value_loss": 0.4,
        },
    }
    assert categorized_epoch_calls == {
        ("ppo_epoch_diagnostics_stage_rollout", 1): {
            "num_stage_trajectories": 2.0,
            "num_rollouts": 1.0,
            "stage1_num_trajectories": 1.0,
            "stage2_num_trajectories": 1.0,
            "stage3_num_trajectories": 0.0,
        },
        ("ppo_epoch_diagnostics_optimizer", 1): {
            "mean_policy_loss": 0.35,
            "mean_value_loss": 0.45,
            "mean_kl": 0.15,
            "mean_entropy": 0.25,
        },
        ("ppo_epoch_diagnostics_numerics", 1): {
            "grad_norm": 0.8,
            "all_finite": True,
        },
        ("ppo_epoch_diagnostics_stage_rollout", 2): {
            "num_stage_trajectories": 2.0,
            "num_rollouts": 1.0,
            "stage1_num_trajectories": 1.0,
            "stage2_num_trajectories": 1.0,
            "stage3_num_trajectories": 0.0,
        },
        ("ppo_epoch_diagnostics_optimizer", 2): {
            "mean_policy_loss": 0.3,
            "mean_value_loss": 0.4,
            "mean_kl": 0.1,
            "mean_entropy": 0.2,
        },
        ("ppo_epoch_diagnostics_numerics", 2): {
            "grad_norm": 0.7,
            "all_finite": True,
        },
    }
    assert optimizer_calls == [
        (
            {
                "optimizer_step": 25,
                "optimizer_step_in_iteration": 4,
                "ppo_iteration": 1,
                "ppo_epoch_in_iteration": 2,
                "mini_batch_size": 4,
                "policy_loss": 0.3,
                "value_loss": 0.4,
                "total_loss": 0.5,
                "entropy_bonus": 0.2,
                "approx_kl_mean": 0.1,
                "ratio_mean": 1.0,
                "ratio_std": 0.05,
                "clip_fraction": 0.0,
                "batch_advantage_mean": 0.0,
                "batch_advantage_std": 1.0,
                "batch_return_mean": 1.5,
                "new_value_mean": 1.2,
                "grad_norm": 0.9,
                "all_finite": True,
            },
            25,
        )
    ]
    assert categorized_optimizer_calls == {
        ("ppo_optimizer_optimizer", 25): {
            "mini_batch_size": 4,
            "policy_loss": 0.3,
            "value_loss": 0.4,
            "total_loss": 0.5,
            "entropy_bonus": 0.2,
            "approx_kl_mean": 0.1,
            "ratio_mean": 1.0,
            "ratio_std": 0.05,
            "clip_fraction": 0.0,
            "batch_advantage_mean": 0.0,
            "batch_advantage_std": 1.0,
            "batch_return_mean": 1.5,
            "new_value_mean": 1.2,
        },
        ("ppo_optimizer_numerics", 25): {
            "grad_norm": 0.9,
            "all_finite": True,
        },
    }
    assert any(
        summary_payload.get("latest_trajectory_preview") == "preview text"
        for summary_payload, prefix in tracker.summary_calls
        if prefix == "ppo"
    )
    assert iteration_diagnostics_records == [
        {
            "iteration": 1.0,
            "mean_reward": 1.5,
            "mean_kl": 0.1,
            "mean_entropy": 0.2,
            "mean_policy_loss": 0.3,
            "mean_value_loss": 0.4,
            "num_stage_trajectories": 2.0,
            "num_rollouts": 1.0,
            "mean_realized_stage_count": 1.5,
            "fraction_rollouts_reaching_stage_2": 0.5,
        }
    ]
    assert iteration_diagnostics_categorized_records == [
        {
            "iteration": 1.0,
            "categories": {
                "reward_only": {"mean_reward": 1.5},
                "stage_rollout": {
                    "num_stage_trajectories": 2.0,
                    "num_rollouts": 1.0,
                    "mean_realized_stage_count": 1.5,
                    "fraction_rollouts_reaching_stage_2": 0.5,
                },
                "optimizer": {
                    "mean_kl": 0.1,
                    "mean_entropy": 0.2,
                    "mean_policy_loss": 0.3,
                    "mean_value_loss": 0.4,
                },
            },
        }
    ]
    assert epoch_diagnostics_records == [
        {
            "ppo_iteration": 1,
            "ppo_epoch_in_iteration": 1,
            "ppo_epochs_per_batch": 2,
            "optimizer_steps_completed_in_iteration": 2,
            "mean_policy_loss": 0.35,
            "mean_value_loss": 0.45,
            "mean_kl": 0.15,
            "mean_entropy": 0.25,
            "num_stage_trajectories": 2.0,
            "num_rollouts": 1.0,
            "stage1_num_trajectories": 1.0,
            "stage2_num_trajectories": 1.0,
            "stage3_num_trajectories": 0.0,
            "grad_norm": 0.8,
            "all_finite": True,
        },
        {
            "ppo_iteration": 1,
            "ppo_epoch_in_iteration": 2,
            "ppo_epochs_per_batch": 2,
            "optimizer_steps_completed_in_iteration": 4,
            "mean_policy_loss": 0.3,
            "mean_value_loss": 0.4,
            "mean_kl": 0.1,
            "mean_entropy": 0.2,
            "num_stage_trajectories": 2.0,
            "num_rollouts": 1.0,
            "stage1_num_trajectories": 1.0,
            "stage2_num_trajectories": 1.0,
            "stage3_num_trajectories": 0.0,
            "grad_norm": 0.7,
            "all_finite": True,
        },
    ]
    assert epoch_diagnostics_categorized_records == [
        {
            "ppo_iteration": 1,
            "ppo_epoch_in_iteration": 1,
            "ppo_epochs_per_batch": 2,
            "optimizer_steps_completed_in_iteration": 2,
            "categories": {
                "stage_rollout": {
                    "num_stage_trajectories": 2.0,
                    "num_rollouts": 1.0,
                    "stage1_num_trajectories": 1.0,
                    "stage2_num_trajectories": 1.0,
                    "stage3_num_trajectories": 0.0,
                },
                "optimizer": {
                    "mean_policy_loss": 0.35,
                    "mean_value_loss": 0.45,
                    "mean_kl": 0.15,
                    "mean_entropy": 0.25,
                },
                "numerics": {
                    "grad_norm": 0.8,
                    "all_finite": True,
                },
            },
        },
        {
            "ppo_iteration": 1,
            "ppo_epoch_in_iteration": 2,
            "ppo_epochs_per_batch": 2,
            "optimizer_steps_completed_in_iteration": 4,
            "categories": {
                "stage_rollout": {
                    "num_stage_trajectories": 2.0,
                    "num_rollouts": 1.0,
                    "stage1_num_trajectories": 1.0,
                    "stage2_num_trajectories": 1.0,
                    "stage3_num_trajectories": 0.0,
                },
                "optimizer": {
                    "mean_policy_loss": 0.3,
                    "mean_value_loss": 0.4,
                    "mean_kl": 0.1,
                    "mean_entropy": 0.2,
                },
                "numerics": {
                    "grad_norm": 0.7,
                    "all_finite": True,
                },
            },
        },
    ]
    assert optimizer_records[0]["optimizer_step"] == 25
    assert optimizer_categorized_records == [
        {
            "optimizer_step": 25,
            "optimizer_step_in_iteration": 4,
            "ppo_iteration": 1,
            "ppo_epoch_in_iteration": 2,
            "categories": {
                "optimizer": {
                    "mini_batch_size": 4,
                    "policy_loss": 0.3,
                    "value_loss": 0.4,
                    "total_loss": 0.5,
                    "entropy_bonus": 0.2,
                    "approx_kl_mean": 0.1,
                    "ratio_mean": 1.0,
                    "ratio_std": 0.05,
                    "clip_fraction": 0.0,
                    "batch_advantage_mean": 0.0,
                    "batch_advantage_std": 1.0,
                    "batch_return_mean": 1.5,
                    "new_value_mean": 1.2,
                },
                "numerics": {
                    "grad_norm": 0.9,
                    "all_finite": True,
                },
            },
        }
    ]
    assert trajectory_records[0]["example_id"] == "example-1"
    assert trajectory_records[0]["num_stages"] == 2
