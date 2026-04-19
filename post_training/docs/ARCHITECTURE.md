# Post-Training Architecture Map

This is the shortest reading path for the rewritten `post_training` package.

## Main Entry Points

1. [post_training/sft_multi/trainer.py](../sft_multi/trainer.py)
2. [post_training/ppo/trainer.py](../ppo/trainer.py)
3. [post_training/gflownet/trainer.py](../gflownet/trainer.py)
4. [post_training/scripts/test_post_training.sh](../scripts/test_post_training.sh)

## Shared Layer

- Config helpers: [post_training/shared/config.py](../shared/config.py)
- Canonical grouped dataset type: [post_training/shared/dataset_types.py](../shared/dataset_types.py)
- Staged sequence helpers: [post_training/shared/sequence.py](../shared/sequence.py)
- Experiment tracking: [post_training/logging/tracker.py](../logging/tracker.py)

## SFT Flow

```text
post_training.sft_multi.trainer.main
  -> resolve_multi_molecule_sft_config_paths
  -> run_multi_molecule_sft
  -> MultiMoleculeDataset / MultiMoleculeCollator
  -> serialize_staged_target
  -> T5 supervised loop
  -> checkpoints
```

Main files:

- [post_training/sft_multi/prompting.py](../sft_multi/prompting.py)
- [post_training/sft_multi/dataset.py](../sft_multi/dataset.py)
- [post_training/sft_multi/collator.py](../sft_multi/collator.py)
- [post_training/sft_multi/trainer.py](../sft_multi/trainer.py)

## PPO Flow

```text
post_training.ppo.trainer.main
  -> resolve_ppo_config_paths
  -> build_ppo_config / build_reward_config
  -> PolicyValueModel / load_reference_model
  -> sample_rollout_for_example
  -> score_stage_reward
  -> PPO update
  -> checkpoints
```

Main files:

- [post_training/ppo/config.py](../ppo/config.py)
- [post_training/ppo/model.py](../ppo/model.py)
- [post_training/ppo/rewarding.py](../ppo/rewarding.py)
- [post_training/ppo/rollout.py](../ppo/rollout.py)
- [post_training/ppo/trainer.py](../ppo/trainer.py)

## GFlowNet Flow

```text
post_training.gflownet.trainer.main
  -> resolve_gflownet_config_paths
  -> build_gflownet_config / build_reward_config
  -> GFlowNetModel
  -> sample_stage_trajectories_for_example
  -> score_stage_terminal_reward
  -> TB / DB update
  -> checkpoints
```

Main files:

- [post_training/gflownet/config.py](../gflownet/config.py)
- [post_training/gflownet/model.py](../gflownet/model.py)
- [post_training/gflownet/rewarding.py](../gflownet/rewarding.py)
- [post_training/gflownet/rollout.py](../gflownet/rollout.py)
- [post_training/gflownet/losses.py](../gflownet/losses.py)
- [post_training/gflownet/trainer.py](../gflownet/trainer.py)
