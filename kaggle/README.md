# Kaggle Notebooks

This folder contains Kaggle-ready notebooks for the thesis training pipeline.

The notebooks are designed to be uploaded directly to Kaggle and run top-to-bottom with:

- internet enabled
- GPU enabled for training and BioT5 collection

Each notebook:

- clones this repo with HTTPS into `/kaggle/working/Thesis`
- installs the Python dependencies needed by the repo
- checks for `torch`, `rdkit`, `transformers`, `datasets`, and `peft`
- uses the committed BioT5 SELFIES vocabulary file from `molecules/dict/selfies_dict.txt`
- writes a Kaggle-local temporary YAML config under `kaggle/generated_configs/`
- runs the existing repo CLI entrypoint instead of duplicating training logic
- exports a stage artifact bundle under `/kaggle/working/thesis_artifacts/<stage_name>/`

## Notebook Order

1. `00_collect_chebi_biot5.ipynb`
   - downloads and preprocesses ChEBI-20
   - runs the BioT5 grouped collection pipeline
   - writes `data/post_training/processed/train_multimol.jsonl`
   - exports `thesis_artifacts/collect_chebi_biot5/`
2. `01_train_sft.ipynb`
   - runs standard single-molecule SFT on ChEBI-20
   - exports `thesis_artifacts/train_sft/`
3. `02_train_multi_molecule_sft.ipynb`
   - runs staged multi-molecule SFT
   - derives Kaggle-local grouped train/validation/test splits from the collected grouped train file
   - exports `thesis_artifacts/train_multi_molecule_sft/`
4. `03_train_molecule_wise_ppo.ipynb`
   - runs molecule-stage PPO from the multi-molecule SFT checkpoint
   - exports `thesis_artifacts/train_molecule_wise_ppo/`

## Important Notes

- The repo remote in this workspace is SSH-based, but Kaggle notebooks use HTTPS cloning:
  - `https://github.com/mruniverse8/Thesis.git`
- The notebooks are standalone with fallback:
  - they prefer upstream artifacts from `/kaggle/input/...`
  - they fall back to local files in `/kaggle/working/Thesis/...`
- The post-training configs in the repo expect grouped validation and test files that are not produced by the ChEBI collection step. The multi-molecule SFT and PPO notebooks therefore derive Kaggle-local split files from the collected grouped train file.
- The Kaggle configs intentionally reduce batch sizes and iteration counts compared with the local defaults so they are more realistic on smaller Kaggle GPUs.
- Notebook outputs are written under the cloned repo inside `/kaggle/working/Thesis/outputs/kaggle/`.
- To hand outputs from one notebook to the next in fresh Kaggle sessions, publish the relevant stage folder from `/kaggle/working/thesis_artifacts/` as a Kaggle dataset and attach it to the downstream notebook.
