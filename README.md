# Thesis

This repo contains a thesis-local pipeline for BioT5+ data collection, supervised fine-tuning, and post-training on:

`molecule description -> molecule SELFIES`

The active repo surface is focused on dataset preparation, training, post-training, and reward analysis.

## References

### Datasets

- `language-plus-molecules/LPM-24_train`: https://huggingface.co/datasets/language-plus-molecules/LPM-24_train
- `language-plus-molecules/LPM-24_eval-molgen`: https://huggingface.co/datasets/language-plus-molecules/LPM-24_eval-molgen
- `liupf/ChEBI-20-MM`: https://huggingface.co/datasets/liupf/ChEBI-20-MM

### Model

- `QizhiPei/biot5-plus-base`: https://huggingface.co/QizhiPei/biot5-plus-base
- `QizhiPei/biot5-plus-base-chebi20`: https://huggingface.co/QizhiPei/biot5-plus-base-chebi20
- Upstream BioT5+ code: https://github.com/QizhiPei/BioT5/tree/main/biot5_plus

## Repo Layout

- `configs/sft_chebi20.yaml`: training config
- `scripts/download_chebi20.py`: download and preprocess ChEBI-20-MM into local JSONL splits
- `scripts/download_train_dataset.py`: download and extract the mini post-training dataset bundle for Colab/Kaggle runs
- `configs/collect_biot5_chebi20.yaml`: BioT5+ data-collection config for ChEBI-20 train descriptions
- `scripts/collect_biot5_chebi20.py`: collect grouped training molecules with the BioT5+ ChEBI-20 checkpoint and filtering
- `scripts/merge_biot5_collection_parts.py`: merge partitioned BioT5 collection runs into one final grouped bundle
- `scripts/download_lpm24.py`: download and preprocess LPM-24 into grouped multi-molecule JSONL files
- `scripts/train_sft.py`: run description-to-SELFIES SFT
- `scripts/init_colab.py`: bootstrap a Colab runtime and launch one training stage
- `scripts/init_kaggle.py`: bootstrap a Kaggle runtime and launch one training stage
- `colab/`: Colab-specific support helpers and run notes
- `kaggle/`: Kaggle-specific support helpers and run notes
- `post_training/`: architecture specs for multi-molecule SFT and molecule-wise PPO
- `scripts/train_multi_molecule_sft.py`: run description-to-multi-SELFIES SFT
- `scripts/train_molecule_wise_ppo.py`: run molecule-wise PPO from a multi-molecule SFT checkpoint
- `molecules/`: canonical molecule and chemistry package
- `reward_utils/`: compatibility surface for older reward imports
- `src/`: prompt formatting, dataset handling, tokenizer setup, and training helpers
- `data/README.md`: dataset-specific notes
- `data_collection/README.md`: active data-collection runbook and architecture notes
- `legacy/data_collection/`: archived older experiments kept only for reference
- `legacy/legacy_eval/`: archived evaluation prototype kept only for reference

## Quick Start

Install dependencies:

```bash
cd /home/abril/Documents/HSE/Thesis/Thesis
conda activate thesis_biot5_sft
pip install -r requirements.txt
conda install -n thesis_biot5_sft -c conda-forge rdkit
```

Prepare the dataset:

```bash
python scripts/download_chebi20.py --output-dir data/chebi20
python scripts/download_train_dataset.py --skip-existing
```

Collect grouped ChEBI-20 training molecules with the BioT5+ ChEBI-20 checkpoint:

```bash
python scripts/collect_biot5_chebi20.py --config configs/collect_biot5_chebi20.yaml
```

Optional: download and preprocess grouped LPM-24 files:

```bash
python scripts/download_lpm24.py --output-dir data/lpm24
```

Train:

```bash
python scripts/train_sft.py --config configs/sft_chebi20.yaml
```

Train the post-training multi-molecule SFT stage:

```bash
python scripts/train_multi_molecule_sft.py --config configs/multi_molecule_sft.yaml
python scripts/init_colab.py --stage multi_sft
```

Train molecule-wise PPO from the multi-molecule SFT checkpoint:

```bash
python scripts/train_molecule_wise_ppo.py --config configs/molecule_wise_ppo.yaml
python scripts/init_kaggle.py --stage ppo
```

## Pipeline Notes

- Data collection uses the BioT5+ ChEBI-20 checkpoint, its checkpoint-native tokenizer, diverse beam search, SELFIES-first wrapper-token cleanup, RDKit validation, similarity acceptance, canonical-SMILES deduplication, and optional contiguous run partitioning for long collection jobs.
- The ChEBI collection step derives grouped multi-molecule training data at `data/post_training/processed/train_multimol.jsonl`.
- Training uses a custom PyTorch loop around `T5ForConditionalGeneration`.
- Prompts follow the BioT5+ text-to-molecule format.
- Targets are written as `<bom>{SELFIES}<eom>`.
- Molecule parsing, SELFIES utilities, fingerprints, similarity, and reward scoring now live under `molecules/`.
- The previous evaluation prototype has been archived under `legacy/legacy_eval/` while that work is being redesigned.
## Current Scope

- Implemented: ChEBI-20 train-split collection with the BioT5+ ChEBI-20 checkpoint and filtering
- Implemented: ChEBI-20-MM text-to-molecule SFT
- Implemented: LPM-24 download and grouped preprocessing
- Implemented: `reward_utils/` workspace for Appendix B.2 `rmatch` / `rdiv` reproduction
- Implemented: `molecules/` canonical package for chemistry, SELFIES, and reward logic
- Implemented: `post_training/` architecture specs for multi-molecule SFT and molecule-wise PPO
- Archived: legacy data-collection experiments under `legacy/data_collection/`, including the old contrastive-search collection bundle
- Archived: evaluation prototype under `legacy/legacy_eval/`
- Deferred: full BioT5+ collection on LPM-24
- Deferred: implementation of the new post-training SFT and PPO code paths
