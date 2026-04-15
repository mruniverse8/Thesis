# Colab Bootstrap

This folder contains the Colab-side support helpers for the thesis training pipeline.

It now also includes thin Colab notebooks that wrap the bootstrap script instead of repeating setup logic:

- `colab/01_train_sft.ipynb`
- `colab/02_train_multi_molecule_sft.ipynb`

The intended Colab pattern is:

1. Clone the repo into `/content/Thesis`.
2. `cd /content/Thesis`
3. Run one init command that installs dependencies, prepares the managed dataset when needed, and launches the selected stage.

Examples:

```bash
python scripts/init_colab.py --stage sft
python scripts/init_colab.py --stage multi_sft
python scripts/init_colab.py --stage ppo
```

Notebook intent:

- the notebooks handle only repo bootstrap, editable parameters, one `init_colab.py` call, and quick output checks
- clone/install/dataset preparation/training stay inside `scripts/init_colab.py`

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
python scripts/init_colab.py --stage ppo --config configs/molecule_wise_ppo.yaml --dataset-mode never
python scripts/download_train_dataset.py --skip-existing
```
