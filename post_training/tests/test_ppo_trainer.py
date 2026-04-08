import torch

from post_training.ppo_trainer import compute_clipped_policy_objective, standardize_tensor
from post_training.ppo_types import PPOConfig


def test_standardize_tensor_centers_values() -> None:
    standardized = standardize_tensor(torch.tensor([1.0, 2.0, 3.0]))

    assert standardized.mean().abs().item() < 1.0e-6


def test_compute_clipped_policy_objective_clamps_ratio() -> None:
    ratio = torch.tensor([1.5, 0.7])
    advantages = torch.tensor([1.0, -1.0])

    objective = compute_clipped_policy_objective(ratio, advantages, clip_range=0.2)

    assert torch.allclose(objective, torch.tensor([1.2, -0.8]))


def test_ppo_config_from_dict_reads_rollout_section() -> None:
    config = PPOConfig.from_dict(
        {
            "output_dir": "outputs/test",
            "target_modules": ["q", "v"],
            "rollout": {
                "max_stage_new_tokens": 16,
                "max_molecules_per_sequence": 4,
            },
        }
    )

    assert config.output_dir == "outputs/test"
    assert config.rollout.max_stage_new_tokens == 16
    assert config.rollout.max_molecules_per_sequence == 4
