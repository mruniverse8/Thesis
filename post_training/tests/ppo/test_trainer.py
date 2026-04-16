from pathlib import Path
from types import SimpleNamespace

import torch

from post_training.ppo.config import PPOConfig
from post_training.ppo.trainer import run_molecule_stage_ppo, standardize_tensor
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
