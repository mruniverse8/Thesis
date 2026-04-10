# Testing Guide

This file collects the shortest commands to validate the thesis BioT5+ SFT pipeline.

## Environment

```bash
conda activate thesis_biot5_sft
cd /home/abril/Documents/HSE/Thesis/Thesis
```

## Quick Smoke Test

Run the lightweight smoke test first. It checks the most important pieces of the pipeline without downloading a model or training:

```bash
python scripts/smoke_test_pipeline.py
```

It runs 7 checks:

1. Prompt formatting for `description -> SELFIES`
2. `<bom>...<eom>` target wrapping and unwrapping
3. SELFIES token filtering from noisy text
4. Clean SELFIES decoding to SMILES
5. Noisy SELFIES repair and decoding
6. Record preprocessing when the dataset already contains SELFIES
7. `SMILES -> SELFIES` fallback plus dataset JSONL round-trip

## Pytest Suite

Run the repo tests:

```bash
python -m pytest tests -q
```

## Full Pipeline

Download and preprocess ChEBI-20-MM:

```bash
python scripts/download_chebi20.py --output-dir data/chebi20
```

Train:

```bash
python scripts/train_sft.py --config configs/sft_chebi20.yaml
```

The previous evaluation commands have been archived under `legacy/legacy_eval/scripts/` while evaluation is being redesigned.

## Recommended Order

```bash
conda activate thesis_biot5_sft
cd /home/abril/Documents/HSE/Thesis/Thesis
python scripts/smoke_test_pipeline.py
python -m pytest tests -q
python scripts/download_chebi20.py --output-dir data/chebi20
python scripts/train_sft.py --config configs/sft_chebi20.yaml
```
