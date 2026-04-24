from pathlib import Path

from post_training.gflownet.config import GFlowNetConfig, build_gflownet_config
from post_training.shared.config import (
    DEFAULT_PPO_FALLBACK_CHECKPOINT,
    resolve_gflownet_config_paths,
)
from src.io_utils import load_yaml


def test_gflownet_config_from_dict_supports_stage_rollout_and_bounded_replay() -> None:
    config = GFlowNetConfig.from_dict(
        {
            "objective": "db",
            "diagnostic_log_every_iterations": 12,
            "trajectory_preview_every_iterations": 8,
            "trajectory_preview_num_samples": 5,
            "trajectory_preview_max_chars": 320,
            "rollout": {
                "max_stage_new_tokens": 96,
                "max_molecules_per_sequence": 4,
                "append_probability": 0.6,
                "invalid_append_probability": 0.25,
            },
            "replay": {
                "enabled": True,
                "buffer_type": "uniform",
                "capacity": 128,
                "replay_fraction": 0.2,
                "max_total_action_tokens": 4096,
                "with_replacement": True,
                "top_reward_fraction": 0.4,
                "hard_positive_fraction": 0.3,
                "hard_negative_fraction": 0.3,
            },
        }
    )

    assert config.objective == "db"
    assert config.diagnostic_log_every_iterations == 12
    assert config.trajectory_preview_every_iterations == 8
    assert config.trajectory_preview_num_samples == 5
    assert config.trajectory_preview_max_chars == 320
    assert config.rollout.max_stage_new_tokens == 96
    assert config.rollout.max_molecules_per_sequence == 4
    assert config.rollout.append_probability == 0.6
    assert config.rollout.invalid_append_probability == 0.25
    assert config.rollout.decoding_strategy == "sample"
    assert config.rollout.num_beams == 1
    assert config.replay.enabled is True
    assert config.replay.buffer_type == "uniform"
    assert config.replay.capacity == 128
    assert config.replay.replay_fraction == 0.2
    assert config.replay.max_total_action_tokens == 4096
    assert config.replay.with_replacement is True
    assert config.replay.top_reward_fraction == 0.4
    assert config.replay.hard_positive_fraction == 0.3
    assert config.replay.hard_negative_fraction == 0.3


def test_gflownet_config_from_dict_keeps_legacy_replay_batch_size_when_fraction_is_absent() -> None:
    config = GFlowNetConfig.from_dict(
        {
            "replay": {
                "replay_batch_size": 16,
            }
        }
    )

    assert config.replay.replay_fraction is None
    assert config.replay.replay_batch_size == 16


def test_gflownet_config_from_dict_accepts_subtb() -> None:
    config = GFlowNetConfig.from_dict({"objective": "subtb"})

    assert config.objective == "subtb"


def test_gflownet_config_from_dict_accepts_legacy_max_new_tokens_field() -> None:
    config = GFlowNetConfig.from_dict({"rollout": {"max_new_tokens": 80}})

    assert config.rollout.max_stage_new_tokens == 80


def test_gflownet_config_from_dict_clamps_sparse_diagnostics_fields() -> None:
    config = GFlowNetConfig.from_dict(
        {
            "diagnostic_log_every_iterations": 0,
            "trajectory_preview_every_iterations": -1,
            "trajectory_preview_num_samples": 0,
            "trajectory_preview_max_chars": 8,
        }
    )

    assert config.diagnostic_log_every_iterations == 1
    assert config.trajectory_preview_every_iterations == 1
    assert config.trajectory_preview_num_samples == 1
    assert config.trajectory_preview_max_chars == 32


def test_gflownet_rollout_defaults_enable_constrained_decoding_and_tighter_sampling() -> None:
    config = GFlowNetConfig.from_dict({})

    assert config.rollout.temperature == 0.8
    assert config.rollout.top_p == 0.95
    assert config.rollout.decoding_strategy == "sample"
    assert config.rollout.num_beams == 1
    assert config.rollout.length_penalty == 1.0
    assert config.rollout.early_stopping is True
    assert config.rollout.constrained_decoding is True
    assert config.rollout.append_probability == 0.30
    assert config.rollout.invalid_append_probability == 0.0
    assert config.rollout.terminate_on_invalid_stage is True
    assert config.rollout.selfies_dict_path == "molecules/dict/selfies_dict.txt"
    assert config.replay.enabled is True
    assert config.replay.buffer_type == "priority"
    assert config.replay.replay_fraction == 0.25


def test_gflownet_config_normalizes_invalid_stage_termination_to_true() -> None:
    config = GFlowNetConfig.from_dict({"rollout": {"terminate_on_invalid_stage": False}})

    assert config.rollout.terminate_on_invalid_stage is True


def test_gflownet_config_from_dict_supports_beam_rollout_options() -> None:
    config = GFlowNetConfig.from_dict(
        {
            "rollout": {
                "decoding_strategy": "beam",
                "num_beams": 4,
                "length_penalty": 0.7,
                "early_stopping": False,
            }
        }
    )

    assert config.rollout.decoding_strategy == "beam"
    assert config.rollout.num_beams == 4
    assert config.rollout.length_penalty == 0.7
    assert config.rollout.early_stopping is False


def test_build_gflownet_config_uses_training_and_model_defaults() -> None:
    config = build_gflownet_config(
        {
            "training": {"output_dir": "outputs/gflownet"},
            "model": {"use_lora": False},
            "data": {"max_source_length": 384},
            "gflownet": {"batch_size": 32},
        }
    )

    assert config.output_dir == "outputs/gflownet"
    assert config.use_lora is False
    assert config.batch_size == 32
    assert config.rollout.max_source_length == 384
    assert config.replay.capacity == 256
    assert config.replay.max_total_action_tokens == 50_000


def test_multi_molecule_gflownet_mini_config_parses_for_tb_db_and_subtb() -> None:
    config_path = Path(__file__).resolve().parents[3] / "configs" / "multi_molecule_gflownet_mini.yaml"
    payload = load_yaml(config_path)

    tb_config = build_gflownet_config(payload)
    assert tb_config.objective == "tb"
    assert tb_config.rollout.temperature == 0.08
    assert tb_config.rollout.top_p == 0.25
    assert tb_config.diagnostic_log_every_iterations == 5
    assert tb_config.trajectory_preview_every_iterations == 5
    assert tb_config.trajectory_preview_num_samples == 3
    assert tb_config.trajectory_preview_max_chars == 480

    payload["gflownet"]["objective"] = "db"
    db_config = build_gflownet_config(payload)
    assert db_config.objective == "db"

    payload["gflownet"]["objective"] = "subtb"
    subtb_config = build_gflownet_config(payload)
    assert subtb_config.objective == "subtb"


def test_multi_molecule_gflownet_full_config_uses_conservative_rollout_defaults() -> None:
    config_path = Path(__file__).resolve().parents[3] / "configs" / "multi_molecule_gflownet.yaml"
    payload = load_yaml(config_path)

    config = build_gflownet_config(payload)

    assert config.rollout.temperature == 0.08
    assert config.rollout.top_p == 0.25


def test_resolve_gflownet_config_paths_falls_back_to_original_weights_for_missing_checkpoint(
    tmp_path: Path,
) -> None:
    config = resolve_gflownet_config_paths(
        {
            "model": {"checkpoint": "outputs/multi_molecule_sft_mini/checkpoints/best"},
            "data": {"train_file": "data/train.jsonl"},
            "training": {"output_dir": "outputs/multi_molecule_gflownet"},
        },
        project_root=tmp_path,
    )

    assert config["model"]["checkpoint"] == DEFAULT_PPO_FALLBACK_CHECKPOINT
    assert config["data"]["train_file"] == str((tmp_path / "data" / "train.jsonl").resolve())
