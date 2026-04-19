# Post-Training

`post_training` is now organized around the real training stages instead of one flat module list.

## Start Here

- [Docs index](./docs/README.md)
- [Overall architecture](./docs/ARCHITECTURE.md)
- [SFT guide](./docs/sft/README.md)
- [PPO guide](./docs/ppo/README.md)
- [GFlowNet guide](./docs/gflownet/README.md)
- [Run scripts](./scripts/README.md)

## Package Layout

- [post_training/shared](./shared)
- [post_training/logging](./logging)
- [post_training/sft_multi](./sft_multi)
- [post_training/ppo](./ppo)
- [post_training/gflownet](./gflownet)

## Canonical Target Format

Staged multi-molecule SFT target:

```text
<bom>{m1}<eom> <bom>{m2}<eom> ... <bom>{mK}<eom>
```

PPO still generates one molecule per stage.

## Quick Run

```bash
bash post_training/scripts/train_multi_molecule_sft.sh
bash post_training/scripts/train_molecule_wise_ppo.sh
bash post_training/scripts/train_multi_molecule_gflownet.sh
bash post_training/scripts/test_post_training.sh
```

The wrapper scripts default to `CONDA_ENV=thesis_biot5_sft`.

## Tracking

Minimal remote tracking is available for SFT, PPO, and GFlowNet through the top-level `tracking:` config block. Supported backends are `wandb` and `comet`, and local JSON outputs remain unchanged.
