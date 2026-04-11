# Local Notebooks

This folder contains local-only review notebooks for inspecting the thesis pipeline without the Kaggle wrappers.

Current notebooks:

- `00_biot5_collection_local_review.ipynb`
  Runs a small active-pipeline BioT5 collection review on `1-2` ChEBI descriptions using the same `QizhiPei/biot5-plus-base-chebi20` checkpoint-native tokenizer path as the active dataset collection code, compares the maintained diverse-beam presets, decodes SELFIES first, and prints timing plus molecule-validity and similarity summaries.

- `01_biot5_diverse_beam_local_review.ipynb`
  Runs a native BioT5 text2mol local review with `QizhiPei/biot5-plus-base-chebi20`, reuses the shared active generator contract, compares greedy and diverse beam generation on the same ChEBI descriptions, decodes SELFIES first, and prints timing plus molecule-validity and similarity summaries.

Run these notebooks from the thesis environment:

```bash
cd /home/abril/Documents/HSE/Thesis/Thesis
conda activate thesis_biot5_sft
jupyter lab
```
