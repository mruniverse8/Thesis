# Data Collection

This folder contains the active BioT5 ChEBI-20 data-collection pipeline. The active path now uses diverse beam search and a SELFIES-first extraction step that explicitly handles BioT5 wrapper tokens such as `<bom>`, `<eom>`, `<pad>`, and `</s>`.

## Run The Active Collection

From the repo root:

```bash
cd /home/abril/Documents/HSE/Thesis/Thesis
conda activate thesis_biot5_sft
pip install -r requirements.txt
conda install -n thesis_biot5_sft -c conda-forge rdkit
python scripts/download_chebi20.py --output-dir data/chebi20
python scripts/collect_biot5_chebi20.py --config configs/collect_biot5_chebi20.yaml
```

Current defaults in `configs/collect_biot5_chebi20.yaml`:

- model checkpoint: `QizhiPei/biot5-plus-base-chebi20`
- tokenizer source: checkpoint-native `T5Tokenizer.from_pretrained(model_name_or_path)`
- dataset: `data/chebi20/processed/train.jsonl`
- target generations per description: `30`
- decoding: diverse beam search
- default beam preset: `num_beams=30`, `num_return_sequences=30`, `num_beam_groups=6`, `diversity_penalty=0.5`
- acceptance threshold: Dice similarity `0.7`
- retained grouped molecules per description: `8`

## Active Outputs

Running the script writes:

- `data_collection/outputs/chebi20_biot5_train/resolved_config.yaml`
- `data_collection/outputs/chebi20_biot5_train/inputs.jsonl`
- `data_collection/outputs/chebi20_biot5_train/raw_candidates.jsonl`
- `data_collection/outputs/chebi20_biot5_train/candidate_assessments.jsonl`
- `data_collection/outputs/chebi20_biot5_train/accepted_grouped.jsonl`
- `data_collection/outputs/chebi20_biot5_train/summary.json`
- `data/post_training/processed/train_multimol.jsonl`

The active pipeline still writes the dataset artifacts with `write_jsonl(...)` in `collect_biot5_training_data(...)`; only the generation strategy and SELFIES extraction behavior changed.

## SELFIES Extraction Contract

BioT5 generates SELFIES tokens wrapped in seq2seq tokens. The active parser does this before chemistry validation:

1. decode model outputs without dropping special tokens
2. strip `<pad>`, `</s>`, and other generic wrappers
3. treat `<bom>` and `<eom>` as molecule-boundary markers
4. try direct SELFIES decoding first
5. fall back to `filter_selfies(...)` only when noisy text still contains a recoverable bracketed SELFIES span

This metadata is recorded in staged artifacts:

- `raw_candidates.jsonl`: cleaned SELFIES, filtered fallback span, selected SELFIES, fallback flag
- `candidate_assessments.jsonl`: cleaned/selected SELFIES, decoded SMILES, fallback flag, decode error, similarity, and rejection reason

## Code Flow

Read the active code in this order:

1. `scripts/collect_biot5_chebi20.py`
   Loads YAML, resolves paths, and runs `collect_biot5_training_data(...)`.
2. `data_collection/config_utils.py`
   Resolves repo-relative paths in the collection config.
3. `data_collection/biot5_generation.py`
   Builds diverse-beam `generate(...)` kwargs, loads the BioT5 ChEBI-20 checkpoint plus its native tokenizer, and handles remote `group-beam-search` fallback when required.
4. `data_collection/biot5_collection.py`
   Orchestrates dataset selection, prompt construction, candidate generation, artifact writing, and final grouped dataset export.
5. `molecules/selfies.py`
   Owns the shared BioT5 SELFIES cleanup, wrapper stripping, fallback extraction, and decode helpers.
6. `molecules/collection/filtering.py`
   Validates chemistry, computes Dice similarity, and assigns rejection reasons.

## Legacy Contrastive Flow

The previous contrastive-search implementation is preserved under:

`legacy/data_collection/contrastive_collection/`

That bundle includes the archived contrastive generator, legacy config, legacy CLI, and a README that explains the old code flow.

## Tests

Run the active collection tests with:

```bash
conda run -n thesis_biot5_sft pytest -q \
  tests/test_selfies_utils.py \
  tests/test_biot5_collection.py \
  tests/test_lpm24_download.py
```
