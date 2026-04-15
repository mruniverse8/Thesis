# Colab Bootstrap

This folder contains the Colab-side support helpers for the thesis training pipeline.

It now also includes thin Colab notebooks that keep training launch notebook-local instead of hiding it behind the bootstrap script:

- `colab/01_train_sft.ipynb`
- `colab/02_train_multi_molecule_sft.ipynb`

The intended Colab pattern is:

1. Clone the repo into `/content/Thesis`.
2. `cd /content/Thesis`
3. Run one init command that installs dependencies and prepares the managed dataset when needed.
4. Run the selected training script directly with the repo config.

Examples:

```bash
python scripts/init_colab.py --stage sft
python scripts/train_sft.py --config configs/sft_chebi20.yaml

python scripts/init_colab.py --stage multi_sft
python scripts/train_multi_molecule_sft.py --config configs/multi_molecule_sft_mini.yaml

python scripts/init_colab.py --stage ppo
python scripts/train_molecule_wise_ppo.py --config configs/molecule_wise_ppo_mini.yaml
```

Notebook intent:

- the notebooks handle repo bootstrap, editable parameters, one bootstrap call, one direct training call, and quick output checks
- `scripts/init_colab.py` stays responsible for install/dataset preparation only
- training still runs directly through `scripts/train_*.py`

Defaults:

- `sft` uses `configs/sft_chebi20.yaml`
- `multi_sft` uses `configs/multi_molecule_sft_mini.yaml`
- `ppo` uses `configs/molecule_wise_ppo_mini.yaml`

The post-training mini dataset bootstrap uses:

- `scripts/download_train_dataset.py`
- default Google Drive file id `1fwRIHrcq0nGA1OJCCWqdxvGgcbW2oKY3`
- extracted dataset root `data/mini_post_training/`

Useful overrides:

```bash
python scripts/init_colab.py --stage multi_sft --dataset-mode always
python scripts/train_multi_molecule_sft.py --config configs/multi_molecule_sft.yaml
python scripts/init_colab.py --stage ppo --config configs/molecule_wise_ppo.yaml --dataset-mode never
python scripts/train_molecule_wise_ppo.py --config configs/molecule_wise_ppo.yaml
python scripts/download_train_dataset.py --skip-existing
```
