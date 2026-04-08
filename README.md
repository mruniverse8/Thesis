# Thesis

This repo contains a thesis-local pipeline for BioT5+ data collection, supervised fine-tuning, evaluation, and post-training on:

`molecule description -> molecule SELFIES`

The active repo surface is focused on dataset preparation, training, evaluation, and reward analysis.

## References

### Datasets

- `language-plus-molecules/LPM-24_train`: https://huggingface.co/datasets/language-plus-molecules/LPM-24_train
- `language-plus-molecules/LPM-24_eval-molgen`: https://huggingface.co/datasets/language-plus-molecules/LPM-24_eval-molgen
- `liupf/ChEBI-20-MM`: https://huggingface.co/datasets/liupf/ChEBI-20-MM

### Model

- `QizhiPei/biot5-plus-base`: https://huggingface.co/QizhiPei/biot5-plus-base
- Upstream BioT5+ code: https://github.com/QizhiPei/BioT5/tree/main/biot5_plus

## Repo Layout

- `configs/sft_chebi20.yaml`: training config
- `scripts/download_chebi20.py`: download and preprocess ChEBI-20-MM into local JSONL splits
- `configs/collect_biot5_chebi20.yaml`: BioT5+ data-collection config for ChEBI-20 train descriptions
- `scripts/collect_biot5_chebi20.py`: collect grouped training molecules with base BioT5+ and filtering
- `scripts/download_lpm24.py`: download and preprocess LPM-24 into grouped multi-molecule JSONL files
- `scripts/train_sft.py`: run description-to-SELFIES SFT
- `scripts/eval_sft.py`: run generation evaluation on a saved checkpoint
- `post_training/`: architecture specs for multi-molecule SFT and molecule-wise PPO
- `scripts/train_multi_molecule_sft.py`: run description-to-multi-SELFIES SFT
- `scripts/eval_multi_molecule_sft.py`: evaluate a multi-molecule SFT checkpoint
- `scripts/train_molecule_wise_ppo.py`: run molecule-wise PPO from a multi-molecule SFT checkpoint
- `reward_utils/`: RDKit-backed validation, similarity, and reward experiments
- `src/`: prompt formatting, dataset handling, tokenizer setup, training loop, and evaluation helpers
- `data/README.md`: dataset-specific notes
- `data_collection/README.md`: active data-collection runbook and architecture notes
- `legacy/data_collection/`: archived older experiments kept only for reference

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
```

Collect grouped ChEBI-20 training molecules with the base BioT5+ model:

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
```

Train molecule-wise PPO from the multi-molecule SFT checkpoint:

```bash
python scripts/train_molecule_wise_ppo.py --config configs/molecule_wise_ppo.yaml
```

Evaluate the best checkpoint on the validation split:

```bash
python scripts/eval_sft.py \
  --config configs/sft_chebi20.yaml \
  --checkpoint outputs/chebi20_sft/checkpoints/best \
  --split validation
```

## Pipeline Notes

- Data collection uses the base BioT5+ checkpoint, contrastive search, RDKit validation, similarity acceptance, and canonical-SMILES deduplication.
- The ChEBI collection step derives grouped multi-molecule training data at `data/post_training/processed/train_multimol.jsonl`.
- Training uses a custom PyTorch loop around `T5ForConditionalGeneration`.
- Prompts follow the BioT5+ text-to-molecule format.
- Targets are written as `<bom>{SELFIES}<eom>`.
- Evaluation uses a decoder tokenizer derived from base T5 so generated SELFIES token IDs stay aligned.
## Current Scope

- Implemented: ChEBI-20 train-split collection with base BioT5+ and filtering
- Implemented: ChEBI-20-MM text-to-molecule SFT
- Implemented: LPM-24 download and grouped preprocessing
- Implemented: `reward_utils/` workspace for Appendix B.2 `rmatch` / `rdiv` reproduction
- Implemented: `post_training/` architecture specs for multi-molecule SFT and molecule-wise PPO
- Archived: legacy data-collection experiments under `legacy/data_collection/`
- Deferred: full BioT5+ collection on LPM-24
- Deferred: implementation of the new post-training SFT and PPO code paths
