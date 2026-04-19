# Post-Training Scripts

These wrappers keep the working directory at the repo root and run everything inside the post-training conda environment.

## Defaults

- Default env: `thesis_biot5_sft`
- Override env: `CONDA_ENV=your_env_name`
- Optional tracking auth: `WANDB_API_KEY=...` or `COMET_API_KEY=...`

## Scripts

- [../../scripts/train_multi_molecule_sft.py](../../scripts/train_multi_molecule_sft.py): compatibility entrypoint for multi-molecule SFT
- [../../scripts/train_molecule_wise_ppo.py](../../scripts/train_molecule_wise_ppo.py): compatibility entrypoint for molecule-wise PPO
- [../../scripts/train_multi_molecule_gflownet.py](../../scripts/train_multi_molecule_gflownet.py): compatibility entrypoint for multi-molecule GFlowNet
- [train_multi_molecule_sft.sh](./train_multi_molecule_sft.sh): run `python -m post_training.sft_multi.trainer`
- [train_molecule_wise_ppo.sh](./train_molecule_wise_ppo.sh): run `python -m post_training.ppo.trainer`
- [train_multi_molecule_gflownet.sh](./train_multi_molecule_gflownet.sh): run `python -m post_training.gflownet.trainer`
- [test_post_training.sh](./test_post_training.sh): run `pytest` on `post_training/tests`

## Examples

```bash
bash post_training/scripts/train_multi_molecule_sft.sh
bash post_training/scripts/train_multi_molecule_sft.sh --config configs/multi_molecule_sft.yaml

WANDB_API_KEY=... bash post_training/scripts/train_multi_molecule_sft.sh --config configs/multi_molecule_sft.yaml

bash post_training/scripts/train_molecule_wise_ppo.sh
bash post_training/scripts/train_molecule_wise_ppo.sh --config configs/molecule_wise_ppo.yaml

COMET_API_KEY=... bash post_training/scripts/train_molecule_wise_ppo.sh --config configs/molecule_wise_ppo.yaml

bash post_training/scripts/train_multi_molecule_gflownet.sh
bash post_training/scripts/train_multi_molecule_gflownet.sh --config configs/multi_molecule_gflownet.yaml

bash post_training/scripts/test_post_training.sh
CONDA_ENV=thesis_biot5_sft bash post_training/scripts/test_post_training.sh post_training/tests/ppo/test_trainer.py -q
```
