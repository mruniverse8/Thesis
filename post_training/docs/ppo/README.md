# Molecule-Stage PPO

This page is the compact reading guide for the reinforcement-learning stage.

## Reward Contract

```text
r_match(m_k, p_desc) =
max_{m_target in M_D(p_desc)} Dice(m_target, m_k)^alpha
```

```text
r_div(m_k, {m_i}_{i=1}^{k-1}) =
1 - max_{m in {m_i}_{i=1}^{k-1}} Tanimoto(m_k, m)^beta
```

```text
r_total = w_match * r_match + w_div * r_div
r_ppo = amplification * r_total
```

The post-training package wraps the existing chemistry implementation instead of reimplementing it:

- [post_training/ppo/rewarding.py](../../ppo/rewarding.py)
- [reward_utils/rewards.py](../../../reward_utils/rewards.py)

## Main Files To Read

1. [post_training/ppo/trainer.py](../../ppo/trainer.py)
2. [post_training/ppo/config.py](../../ppo/config.py)
3. [post_training/ppo/model.py](../../ppo/model.py)
4. [post_training/ppo/rollout.py](../../ppo/rollout.py)
5. [post_training/ppo/rewarding.py](../../ppo/rewarding.py)
6. [post_training/tests/ppo/test_rewarding.py](../../tests/ppo/test_rewarding.py)

## Entry Flow

```text
trainer.main
  -> resolve_ppo_config_paths
  -> build_ppo_config
  -> build_reward_config
  -> PolicyValueModel / load_reference_model
  -> sample_rollout_for_example
  -> score_stage_reward
  -> PPO update
```

## Run It

- Wrapper script: [post_training/scripts/train_molecule_wise_ppo.sh](../../scripts/train_molecule_wise_ppo.sh)
- Default config: [configs/molecule_wise_ppo.yaml](../../../configs/molecule_wise_ppo.yaml)
- Tests: [post_training/scripts/test_post_training.sh](../../scripts/test_post_training.sh)

`peft` is required when `use_lora: true`.
