# Kaggle Notebooks

This folder contains Kaggle-ready notebooks for the thesis training pipeline.

The notebooks are designed to be uploaded directly to Kaggle and run top-to-bottom with:

- internet enabled
- GPU enabled for training and BioT5 collection/review

The new low-noise bootstrap path is:

- clone the repo into `/kaggle/working/Thesis`
- optionally run `python scripts/init_kaggle.py --stage <sft|multi_sft|ppo>`
- let the script install requirements and prepare the managed dataset when needed
- run the training script directly with the selected config

The existing notebooks already follow the direct training-script pattern; the init script is only a bootstrap helper.

Each notebook:

- clones this repo with HTTPS into `/kaggle/working/Thesis`
- installs the Python dependencies needed by the repo
- checks for `torch`, `rdkit`, `transformers`, `datasets`, and `peft`
- imports shared notebook helpers from `kaggle/thesis_kaggle_support.py` through the repo-root compatibility module `thesis_kaggle_support.py`
- writes a Kaggle-local temporary YAML config under `kaggle/generated_configs/`
- runs the existing repo CLI entrypoint instead of duplicating training logic
- exports a stage artifact bundle under `/kaggle/working/thesis_artifacts/<stage_name>/`

Exception:
- `04_biot5_diverse_beam_review.ipynb` is a diagnostic notebook, so it keeps the review loop notebook-local instead of calling a repo CLI.
- `04_biot5_diverse_beam_review.ipynb` still uses the same active `BioT5DiverseBeamGenerator` contract as collection, so the model, tokenizer, and grouped-beam fallback behavior stay aligned.
- `06_biot5_mini_dataset_review.ipynb` is a small-scale collection notebook on purpose: it reuses the active BioT5 collection path from `00_collect_chebi_biot5.ipynb`, but limits the run to 128 descriptions for debugging.

## Notebook Order

1. `00_collect_chebi_biot5.ipynb`
   - downloads and preprocesses ChEBI-20
   - runs one partition of the active BioT5 grouped collection pipeline with the `QizhiPei/biot5-plus-base-chebi20` checkpoint and diverse beam search
   - writes part-specific collection outputs and part-specific `train_multimol` JSONL
   - exports `thesis_artifacts/collect_chebi_biot5_part_<n>/`
   - writes `thesis_artifacts/collect_chebi_biot5_part_<n>.zip` so each part can be downloaded and re-uploaded later
2. `01_train_sft.ipynb`
   - runs standard single-molecule SFT on ChEBI-20
   - stays compatible with the upstream diverse-beam collection stage artifact chain
   - exports `thesis_artifacts/train_sft/`
3. `02_train_multi_molecule_sft.ipynb`
   - runs staged multi-molecule SFT
   - derives Kaggle-local grouped train/validation/test splits from the grouped train file produced by the merged collection stage
   - exports `thesis_artifacts/train_multi_molecule_sft/`
4. `03_train_molecule_wise_ppo.ipynb`
   - runs molecule-stage PPO from the multi-molecule SFT checkpoint
   - continues the downstream pipeline after diverse-beam collection and multi-molecule SFT
   - exports `thesis_artifacts/train_molecule_wise_ppo/`
5. `04_biot5_diverse_beam_review.ipynb`
   - downloads and preprocesses ChEBI-20 if needed
   - runs a native BioT5 text2mol review on selected ChEBI descriptions with the same checkpoint-native generator contract used by active collection
   - compares greedy and diverse beam generation with SELFIES-first decoding
   - mirrors the maintained local review style used by the repo notebooks
   - prints timing, SELFIES validity, SMILES validity, and similarity summaries
   - writes `raw_generations.jsonl`, `review_rows.jsonl`, and `summary_rows.json`
   - exports `thesis_artifacts/review_biot5_diverse_beam/`
6. `05_merge_biot5_collection_parts.ipynb`
   - copies the partitioned collection stage artifacts into Kaggle-local paths
   - expects an attached dataset containing extracted part folders under `thesis_artifacts/collect_chebi_biot5_part_<n>/`
   - validates that all attached parts come from the same collection config and source dataset
   - merges staging JSONL files plus the final grouped `train_multimol.jsonl`
   - exports `thesis_artifacts/merge_biot5_collection_parts/`
7. `06_biot5_mini_dataset_review.ipynb`
   - downloads and preprocesses ChEBI-20 if needed
   - runs the active BioT5 grouped collection pipeline on 128 train descriptions with 100 samples per description
   - writes a grouped mini `train_multimol` file in the same post-training format produced by the collection pipeline
   - derives grouped train/validation/test split files from that mini grouped train file
   - writes `config_snapshot.json` under `outputs/kaggle/mini_post_training/`
   - exports `thesis_artifacts/mini_post_training/` and writes `thesis_artifacts/mini-post-training.zip`

## Zipped Part Tutorial

1. Run `00_collect_chebi_biot5.ipynb` five times with `PART_INDEX = 1, 2, 3, 4, 5`.
2. Download each `collect_chebi_biot5_part_<n>.zip` from `/kaggle/working/thesis_artifacts/`.
3. Create one local folder named `thesis_artifacts`.
4. Extract every part zip into that same folder so it contains `thesis_artifacts/collect_chebi_biot5_part_1/` through `thesis_artifacts/collect_chebi_biot5_part_5/`.
5. Upload that parent `thesis_artifacts/` folder as one Kaggle dataset.
6. Attach that dataset to `05_merge_biot5_collection_parts.ipynb`.
7. Run `05_merge_biot5_collection_parts.ipynb`; it will resolve `thesis_artifacts/<stage_name>/...` automatically.

## Important Notes

- The repo remote in this workspace is SSH-based, but Kaggle notebooks use HTTPS cloning:
  - `https://github.com/mruniverse8/Thesis.git`
- The notebooks are standalone with fallback:
  - they prefer upstream artifacts from `/kaggle/input/...`
  - they fall back to local files in `/kaggle/working/Thesis/...`
- The collection notebook overrides the active generation config in a diverse-beam-compatible way, so `target_molecules_per_description` stays aligned with `num_return_sequences` while keeping the checkpoint-native BioT5 tokenizer/model path intact.
- The collection notebook is intended to run five separate part jobs by default; for the smoothest manual handoff, upload one dataset whose root contains `thesis_artifacts/collect_chebi_biot5_part_<n>/` for all five parts.
- The mini notebook is self-contained: it does not depend on `05_merge_biot5_collection_parts.ipynb` or any attached grouped artifact dataset.
- `06_biot5_mini_dataset_review.ipynb` collects 128 descriptions and samples 100 candidates per description before filtering, deduplication, and grouped post-training export.
- `scripts/init_kaggle.py --stage multi_sft` and `scripts/init_kaggle.py --stage ppo` default to the mini configs and call `scripts/download_train_dataset.py` automatically when `data/mini_post_training/` is missing.
- The post-training configs in the repo expect grouped validation and test files that are not produced by the ChEBI collection step. The multi-molecule SFT and PPO notebooks derive Kaggle-local split files from the collected grouped train file, and `06_biot5_mini_dataset_review.ipynb` now packages a small version of those grouped splits directly for debugging.
- The Kaggle configs intentionally reduce batch sizes and iteration counts compared with the local defaults so they are more realistic on smaller Kaggle GPUs.
- Notebook outputs are written under the cloned repo inside `/kaggle/working/Thesis/outputs/kaggle/`.
- To hand outputs from one notebook to the next in fresh Kaggle sessions, publish the relevant stage folder from `/kaggle/working/thesis_artifacts/` as a Kaggle dataset and attach it to the downstream notebook.
