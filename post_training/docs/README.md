# Post-Training Docs

Use this folder as the compact map for the rewritten `post_training` package.

## Read Order

1. [Overall architecture](./ARCHITECTURE.md)
2. [Staged multi-molecule SFT](./sft/README.md)
3. [Molecule-stage PPO](./ppo/README.md)
4. [Tracking helpers](../logging/README.md)
5. [Run scripts](../scripts/README.md)

## Entry Points

- Internal SFT entrypoint: [post_training/sft_multi/trainer.py](../sft_multi/trainer.py)
- Internal PPO entrypoint: [post_training/ppo/trainer.py](../ppo/trainer.py)
- Repo compatibility scripts: [scripts/train_multi_molecule_sft.py](../../scripts/train_multi_molecule_sft.py) and [scripts/train_molecule_wise_ppo.py](../../scripts/train_molecule_wise_ppo.py)
