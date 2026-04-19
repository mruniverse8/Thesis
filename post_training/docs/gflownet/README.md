# Multi-Molecule GFlowNet

This page is the compact reading guide for the stage-local GFlowNet stage.

## Reward Contract

The GFlowNet terminal reward is built from the current stage chemistry scoring:

```text
state = prompt + previous_valid_molecules
stage_breakdown = score_stage_reward(sampled_stage, targets, previous_valid_molecules, config)
R_terminal = max(amplified_reward(stage_breakdown), ε)
```

Invalid stage terminals receive the positive floor `ε` so the log-space objectives remain defined.

## Main Files To Read

1. [post_training/gflownet/trainer.py](../../gflownet/trainer.py)
2. [post_training/gflownet/config.py](../../gflownet/config.py)
3. [post_training/gflownet/model.py](../../gflownet/model.py)
4. [post_training/gflownet/rollout.py](../../gflownet/rollout.py)
5. [post_training/gflownet/rewarding.py](../../gflownet/rewarding.py)
6. [post_training/gflownet/losses.py](../../gflownet/losses.py)

## Entry Flow

```text
trainer.main
  -> resolve_gflownet_config_paths
  -> build_gflownet_config
  -> build_reward_config
  -> GFlowNetModel
  -> sample_stage_trajectories_for_example
  -> score_stage_terminal_reward
  -> TB / DB update
```

## Run It

- Python script: [scripts/train_multi_molecule_gflownet.py](../../../scripts/train_multi_molecule_gflownet.py)
- Wrapper script: [post_training/scripts/train_multi_molecule_gflownet.sh](../../scripts/train_multi_molecule_gflownet.sh)
- Default config: [configs/multi_molecule_gflownet.yaml](../../../configs/multi_molecule_gflownet.yaml)
- Mini Colab config: [configs/multi_molecule_gflownet_mini.yaml](../../../configs/multi_molecule_gflownet_mini.yaml)
- Tests: [post_training/scripts/test_post_training.sh](../../scripts/test_post_training.sh)

`peft` is required when `use_lora: true`.

The config accepts a top-level `tracking:` block for basic `wandb` or `comet` logging.

`subtb` is intentionally disabled pending a learned state-flow redesign.
