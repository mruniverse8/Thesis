# `molecules/`

Canonical home for molecule- and chemistry-specific code in this repo.

## Entry Points

- `molecules/parsing.py`
  Molecule parsing, canonicalization, RDKit checks, duplicate detection.
- `molecules/selfies.py`
  SELFIES normalization, wrapping/unwrapping, noisy-output repair, decode helpers.
- `molecules/fingerprints.py`
  Morgan fingerprint generation and molecule-record coercion.
- `molecules/similarity.py`
  Dice and Tanimoto similarity helpers.
- `molecules/rewards/scoring.py`
  `rmatch`, `rdiv`, total reward, and sequence scoring.
- `molecules/datasets/single.py`
  Single-molecule dataset preprocessing record builder.
- `molecules/datasets/multi.py`
  Multi-molecule dataset preprocessing record builder.
- `molecules/collection/filtering.py`
  BioT5 data-collection filtering and reference-group preparation.

## Codeflow

### Single-molecule SFT preprocessing

1. Raw ChEBI record enters `molecules.datasets.single.build_processed_record(...)`.
2. Description is normalized.
3. SELFIES is read or derived from SMILES.
4. SELFIES is validated before writing JSONL.

### BioT5 collection filtering

1. `data_collection/biot5_collection.py` builds prompts and generates raw candidates.
2. `molecules.collection.filtering.prepare_reference_groups(...)` prepares per-description reference molecules.
3. `molecules.collection.filtering.assess_candidate(...)` parses candidate SELFIES, validates chemistry, fingerprints it, and scores Dice similarity.
4. Accepted unique molecules are written into grouped outputs for post-training.

### PPO reward scoring

1. Post-training rollout code samples one SELFIES candidate at a stage.
2. `molecules.rewards.scoring.compute_total_reward(...)` dispatches between `reward_var1`, the default `reward_var2` match-plus-validity-bonus reward, and `reward_var3` which combines match, diversity, and the validity bonus.
3. Reward breakdown is attached to the stage trajectory used by PPO.

## Running

Use the thesis environment from the repo root:

```bash
cd /home/abril/Documents/HSE/Thesis/Thesis
conda activate thesis_biot5_sft
pip install -r requirements.txt
conda install -n thesis_biot5_sft -c conda-forge rdkit
```

Smoke-check the molecule-facing active paths:

```bash
python scripts/smoke_test_pipeline.py
python -m pytest tests/test_selfies_utils.py tests/test_dataset_processing.py tests/test_reward_utils.py -q
python -m pytest tests/test_biot5_collection.py tests/test_lpm24_download.py -q
python -m pytest post_training/tests/test_sft_dataset.py post_training/tests/test_reward_adapter.py -q
```

Main pipeline entrypoints:

```bash
python scripts/download_chebi20.py --output-dir data/chebi20
python scripts/collect_biot5_chebi20.py --config configs/collect_biot5_chebi20.yaml
python scripts/train_sft.py --config configs/sft_chebi20.yaml
python scripts/download_lpm24.py --output-dir data/lpm24
python scripts/train_multi_molecule_sft.py --config configs/multi_molecule_sft.yaml
python scripts/train_molecule_wise_ppo.py --config configs/molecule_wise_ppo.yaml
```
