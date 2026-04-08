# Thesis

This repo contains a thesis-local pipeline for supervised fine-tuning BioT5+ on:

`molecule description -> molecule SELFIES`

The active repo surface is focused on dataset preparation, training, evaluation, and reward analysis.

## References

### Datasets

- `language-plus-molecules/LPM-24_train`: https://huggingface.co/datasets/language-plus-molecules/LPM-24_train
- `liupf/ChEBI-20-MM`: https://huggingface.co/datasets/liupf/ChEBI-20-MM

### Model

- `QizhiPei/biot5-plus-base`: https://huggingface.co/QizhiPei/biot5-plus-base
- Upstream BioT5+ code: https://github.com/QizhiPei/BioT5/tree/main/biot5_plus

## Repo Layout

- `configs/sft_chebi20.yaml`: training config
- `scripts/download_chebi20.py`: download and preprocess ChEBI-20-MM into local JSONL splits
- `scripts/train_sft.py`: run description-to-SELFIES SFT
- `scripts/eval_sft.py`: run generation evaluation on a saved checkpoint
- `post_training/`: architecture specs for multi-molecule SFT and molecule-wise PPO
- `scripts/train_multi_molecule_sft.py`: run description-to-multi-SELFIES SFT
- `scripts/eval_multi_molecule_sft.py`: evaluate a multi-molecule SFT checkpoint
- `scripts/train_molecule_wise_ppo.py`: run molecule-wise PPO from a multi-molecule SFT checkpoint
- `reward_utils/`: RDKit-backed validation, similarity, and reward experiments
- `src/`: prompt formatting, dataset handling, tokenizer setup, training loop, and evaluation helpers
- `data/README.md`: dataset-specific notes
- `legacy/data_collection/`: archived data-collection experiments kept only for reference

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

- Training uses a custom PyTorch loop around `T5ForConditionalGeneration`.
- Prompts follow the BioT5+ text-to-molecule format.
- Targets are written as `<bom>{SELFIES}<eom>`.
- Evaluation uses a decoder tokenizer derived from base T5 so generated SELFIES token IDs stay aligned.
## Current Scope

- Implemented: ChEBI-20-MM text-to-molecule SFT
- Implemented: `reward_utils/` workspace for Appendix B.2 `rmatch` / `rdiv` reproduction
- Implemented: `post_training/` architecture specs for multi-molecule SFT and molecule-wise PPO
- Archived: legacy data-collection experiments under `legacy/data_collection/`
- Deferred: LPM-24 integration
- Deferred: implementation of the new post-training SFT and PPO code paths
