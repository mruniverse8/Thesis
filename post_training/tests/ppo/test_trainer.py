import json
from pathlib import Path
from types import SimpleNamespace

import torch

from post_training.ppo.config import PPOConfig, StageTrajectory
from post_training.ppo.trainer import (
    PPOTrainIterationResult,
    MoleculeWisePPOTrainer,
    build_trajectory_preview_payload,
    run_molecule_stage_ppo,
    standardize_tensor,
)
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
                "molecule_separator_token": "<mol_sep>",
            },
        }
    )

    assert config.output_dir == "outputs/test"
    assert config.rollout.max_stage_new_tokens == 16
    assert config.rollout.max_molecules_per_sequence == 4
    assert config.rollout.stage_separator == " "


def test_ppo_config_from_dict_supports_diagnostic_logging_fields() -> None:
    default_config = PPOConfig.from_dict({})
    explicit_config = PPOConfig.from_dict(
        {
            "diagnostic_log_every_optimizer_steps": 12,
            "trajectory_preview_every_iterations": 7,
            "num_trajectory_samples_to_log": 5,
            "trajectory_preview_max_chars": 180,
        }
    )

    assert default_config.diagnostic_log_every_optimizer_steps == 25
    assert default_config.trajectory_preview_every_iterations == 25
    assert default_config.num_trajectory_samples_to_log == 3
    assert default_config.trajectory_preview_max_chars == 240
    assert explicit_config.diagnostic_log_every_optimizer_steps == 12
    assert explicit_config.trajectory_preview_every_iterations == 7
    assert explicit_config.num_trajectory_samples_to_log == 5
    assert explicit_config.trajectory_preview_max_chars == 180


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
            reward=3.0,
            old_value=0.0,
            old_logprob=-0.5,
            reference_logprob=-0.7,
            action_token_ids=(2,),
            sampled_selfies="B",
            stage_text="B",
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
    assert "generated_selfies=B" in preview["tracker_text"]
    assert "generated_selfies=C" in preview["tracker_text"]
    assert "generated_selfies=A" in preview["tracker_text"]


def test_train_iteration_returns_sparse_optimizer_diagnostics_and_preview(monkeypatch) -> None:
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
    config = PPOConfig(
        ppo_iterations=3,
        batch_size=3,
        mini_batch_size=2,
        ppo_epochs_per_batch=1,
        diagnostic_log_every_optimizer_steps=2,
        trajectory_preview_every_iterations=1,
        num_trajectory_samples_to_log=2,
        save_every_iterations=99,
    )
    trainer = MoleculeWisePPOTrainer(
        policy_model=policy_model,
        reference_model=reference_model,
        tokenizer=None,
        config=config,
        device=torch.device("cpu"),
    )

    trajectories = [
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

    result = trainer.train_iteration([{"id": "unused"}], iteration_index=1)

    assert {"reward_std", "mean_old_logprob", "termination_max_sequence_length_rate"} <= set(
        result.metrics
    )
    assert abs(result.metrics["standardized_advantage_mean"]) < 1.0e-6
    assert result.metrics["max_action_token_count"] == 2.0
    assert result.metrics["empty_action_rate"] == 1.0 / 3.0
    assert len(result.optimizer_step_metrics) == 1
    assert result.optimizer_step_metrics[0]["optimizer_step"] == 2
    assert result.optimizer_step_metrics[0]["mini_batch_size"] == 1
    assert result.optimizer_step_metrics[0]["all_finite"] is True
    assert "clip_fraction" in result.optimizer_step_metrics[0]
    assert result.trajectory_preview is not None
    assert len(result.trajectory_preview["records"]) == 2


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


def test_run_molecule_stage_ppo_logs_sparse_diagnostics_and_preview(
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
                },
                optimizer_step_metrics=[
                    {
                        "optimizer_step": 25,
                        "ppo_iteration": iteration_index,
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
                "ppo_epochs_per_batch": 1,
            },
        },
        project_root=tmp_path,
    )

    summary = run_molecule_stage_ppo(config)

    optimizer_diagnostics_path = output_dir / "diagnostics" / "optimizer_step_metrics.jsonl"
    trajectory_previews_path = output_dir / "diagnostics" / "trajectory_previews.jsonl"
    optimizer_records = [
        json.loads(line)
        for line in optimizer_diagnostics_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    trajectory_records = [
        json.loads(line)
        for line in trajectory_previews_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    assert summary["num_iterations"] == 1
    assert any(prefix == "ppo" for _, _, prefix in tracker.metric_calls)
    assert any(prefix == "ppo_step" for _, _, prefix in tracker.metric_calls)
    assert any(
        summary_payload.get("latest_trajectory_preview") == "preview text"
        for summary_payload, prefix in tracker.summary_calls
        if prefix == "ppo"
    )
    assert optimizer_records[0]["optimizer_step"] == 25
    assert trajectory_records[0]["example_id"] == "example-1"
