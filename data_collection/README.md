# Data Collection

This folder contains the active data-collection code for the thesis repo.

The current implementation targets the Section 4.1 style pipeline:

1. load ChEBI-20 training descriptions
2. generate molecules with the base BioT5+ model
3. filter invalid, duplicate, and unaccepted generations
4. derive grouped multi-molecule JSONL files for post-training

`LPM-24` is also supported for download and preprocessing into grouped records, but full BioT5+ collection for `LPM-24` is still a separate next step.

## Environment

From the repo root:

```bash
cd /home/abril/Documents/HSE/Thesis/Thesis
conda activate thesis_biot5_sft
pip install -r requirements.txt
conda install -n thesis_biot5_sft -c conda-forge rdkit
```

The collection script uses the base BioT5+ checkpoint from Hugging Face and the BioT5 SELFIES vocabulary file. The repo now vendors that vocab here:

```bash
molecules/dict/selfies_dict.txt
```

The default configs already point at that in-repo path, so no extra local dependency is required.

## Run ChEBI-20 Collection

Prepare the ChEBI-20 dataset first:

```bash
python scripts/download_chebi20.py --output-dir data/chebi20
```

Then run BioT5+ collection:

```bash
python scripts/collect_biot5_chebi20.py --config configs/collect_biot5_chebi20.yaml
```

Current defaults in `configs/collect_biot5_chebi20.yaml`:

- base model: `QizhiPei/biot5-plus-base`
- dataset: `data/chebi20/processed/train.jsonl`
- target generations per description: `100`
- decoding: contrastive search with `penalty_alpha: 0.5`
- current `top_k`: `4`
- acceptance threshold: Dice similarity `0.7`
- retained grouped molecules per description: `8`

With newer `transformers` releases, contrastive search is loaded through the
`transformers-community/contrastive-search` custom generator. The active
collection code retries generation with `trust_remote_code=True` automatically,
so the Kaggle notebooks do not need a separate workaround.

Useful debug knobs in the same config:

- `runtime.max_descriptions`: limit collection to the first `N` unique descriptions
- `runtime.description_offset`: skip the first `N` unique descriptions
- `generation.target_molecules_per_description`: reduce generation count for a fast smoke test

For a quick local debug run, lower `runtime.max_descriptions` and `generation.target_molecules_per_description` before launching the script.

## Run LPM-24 Download And Preprocess

The repo now includes a downloader/preprocessor for the grouped `LPM-24` files:

```bash
python scripts/download_lpm24.py --output-dir data/lpm24
```

This writes grouped multi-molecule JSONL files under `data/lpm24/processed/`, including the paper-style first-1000 evaluation subset.

The default evaluation dataset is:

```bash
language-plus-molecules/LPM-24_eval-molgen
```

That Hugging Face dataset exposes a `train` split, but in this repo it is treated as the evaluation/test source for molecule generation.

## Output Files

Running `scripts/collect_biot5_chebi20.py` writes these artifacts:

- `data_collection/outputs/chebi20_biot5_train/resolved_config.yaml`
- `data_collection/outputs/chebi20_biot5_train/inputs.jsonl`
- `data_collection/outputs/chebi20_biot5_train/raw_candidates.jsonl`
- `data_collection/outputs/chebi20_biot5_train/candidate_assessments.jsonl`
- `data_collection/outputs/chebi20_biot5_train/accepted_grouped.jsonl`
- `data_collection/outputs/chebi20_biot5_train/summary.json`
- `data/post_training/processed/train_multimol.jsonl`

File meaning:

- `inputs.jsonl`: unique descriptions selected for this run and the prompt text sent to BioT5+
- `raw_candidates.jsonl`: raw decoder outputs before chemistry validation
- `candidate_assessments.jsonl`: one record per candidate with parsed SELFIES, canonical SMILES, best Dice match, and rejection reason
- `accepted_grouped.jsonl`: all accepted unique molecules retained per description
- `train_multimol.jsonl`: final grouped dataset consumed by the post-training multi-molecule SFT code

## Compact Architecture

Read the code in this order if you want to understand the pipeline quickly.

- `scripts/collect_biot5_chebi20.py`
  Thin CLI entrypoint. Loads YAML, resolves paths, then calls `collect_biot5_training_data(...)`.
- `data_collection/config_utils.py::resolve_biot5_collection_config_paths`
  Resolves config paths like the dataset path, staging directory, derived output file, and BioT5 vocab file.
- `data_collection/biot5_collection.py::build_contrastive_generation_kwargs`
  Creates the `transformers.generate(...)` arguments for contrastive search.
- `data_collection/biot5_collection.py::BioT5ContrastiveGenerator`
  Loads the base BioT5+ model and tokenizers, then generates one candidate at a time while blocking exact repeats.
- `data_collection/biot5_collection.py::_select_collection_records`
  Selects unique descriptions in dataset order before applying offset and limit.
- `data_collection/molecule_utils.py::prepare_reference_groups`
  Groups reference molecules by description and precomputes fingerprints used by acceptance filtering.
- `data_collection/molecule_utils.py::assess_candidate`
  Core filtering step. Parses SELFIES, validates chemistry with RDKit, canonicalizes SMILES, computes best Dice similarity, and returns the rejection reason or acceptance result.
- `data_collection/biot5_collection.py::collect_biot5_training_data`
  Main orchestration entrypoint. Builds prompts, calls the generator, deduplicates accepted molecules by canonical SMILES, writes staged artifacts, and produces `train_multimol.jsonl`.
- `data_collection/lpm24.py::build_lpm24_grouped_records`
  Groups raw `LPM-24` rows by description before converting them into the multi-molecule training schema.
- `data_collection/lpm24.py::download_and_preprocess_lpm24`
  Downloads the Hugging Face datasets and writes grouped `train`, `test`, and `test_eval_first_1000` JSONL files.

Shared functions used by the pipeline:

- `src/prompting.py::build_text2mol_prompt`
  Builds the BioT5+ description-to-molecule prompt.
- `src/selfies_utils.py::normalize_generated_selfies`
  Light cleanup on raw model output before parsing.
- `src/selfies_utils.py::parse_generated_selfies`
  Extracts and decodes the SELFIES span from generated text.
- `data_collection/molecule_utils.py::CollectionMetricConfig`
  Holds the similarity and fingerprint settings used by filtering.
- `molecules/datasets/multi.py::build_multi_molecule_processed_record`
  Converts grouped description targets into the post-training schema.

The canonical molecule and chemistry code now lives under `../molecules/`. The `data_collection/` package consumes those helpers but is no longer the source of truth for chemistry primitives.

## Filtering Logic

Each generated candidate goes through this path:

1. normalize the raw text
2. extract SELFIES from the output
3. decode and validate the molecule with RDKit
4. compute Dice similarity against the reference molecules for the same description
5. reject duplicates by canonical SMILES inside that description
6. keep accepted unique molecules and group them into the final training file

Current rejection labels:

- `empty_output`
- `invalid_selfies`
- `invalid_molecule`
- `unaccepted_molecule`
- `duplicate_smiles`

## Tests

Run the data-collection tests with:

```bash
conda run -n thesis_biot5_sft pytest -q tests/test_biot5_collection.py tests/test_lpm24_download.py
```
