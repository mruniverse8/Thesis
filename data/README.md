# Datasets

This folder documents the dataset setup for the active BioT5+ data-collection and SFT pipeline in this repo.

The old data-collection experiments were archived under `../legacy/data_collection/` and are no longer part of the active repo workflow.

## Active Datasets

### ChEBI-20-MM

The main collection and SFT pipeline targets `liupf/ChEBI-20-MM` from Hugging Face:

- Source: https://huggingface.co/datasets/liupf/ChEBI-20-MM
- Relevant fields used by this pipeline: `CID`, `description`, `SMILES`, `SELFIES`
- Task direction: `description -> SELFIES`

The preprocessing script prefers the dataset's existing `SELFIES` column and only falls back to `SMILES -> SELFIES` conversion if the `SELFIES` field is missing.

### LPM-24

The repo now also includes a downloader/preprocessor for:

- `language-plus-molecules/LPM-24_train`
- `language-plus-molecules/LPM-24_eval-molgen`

This step groups multiple molecules under the same description and writes multi-molecule JSONL files. The evaluation dataset is the `molgen` set from Hugging Face, and it currently exposes a `train` split even though it is used as the evaluation/test set. Full BioT5+ collection on `LPM-24` is still a later step.

## Setup

From the thesis repo root:

```bash
pip install -r requirements.txt
python scripts/download_chebi20.py --output-dir data/chebi20
python scripts/download_lpm24.py --output-dir data/lpm24
```

To run the BioT5+ ChEBI collection step after downloading ChEBI:

```bash
python scripts/collect_biot5_chebi20.py --config configs/collect_biot5_chebi20.yaml
```

## Outputs

The download script writes:

- `data/chebi20/raw/download_metadata.json`
- `data/chebi20/processed/train.jsonl`
- `data/chebi20/processed/validation.jsonl`
- `data/chebi20/processed/test.jsonl`

Each processed record has the schema:

```json
{
  "id": "129626631",
  "description": "The molecule is an epoxy(hydroxy)icosatrienoate ...",
  "selfies": "[C][C][C][C][C][C@@H1][O]...",
  "source_smiles": "CCCCC[C@@H]1O[C@@H]1/C=C/C(O)..."
}
```

The collection step then writes:

- `data_collection/outputs/chebi20_biot5_train/inputs.jsonl`
- `data_collection/outputs/chebi20_biot5_train/raw_candidates.jsonl`
- `data_collection/outputs/chebi20_biot5_train/candidate_assessments.jsonl`
- `data_collection/outputs/chebi20_biot5_train/accepted_grouped.jsonl`
- `data_collection/outputs/chebi20_biot5_train/summary.json`
- `data/post_training/processed/train_multimol.jsonl`

For `LPM-24`, the preprocessing step writes:

- `data/lpm24/raw/download_metadata.json`
- `data/lpm24/processed/train_multimol.jsonl`
- `data/lpm24/processed/test_multimol.jsonl`
- `data/lpm24/processed/test_eval_first_1000_multimol.jsonl`

## Notes

- Processed dataset files are ignored by `.gitignore`; regenerate them locally.
- The training config expects the processed files under `data/chebi20/processed/`.
- The grouped ChEBI collection output used by post-training is `data/post_training/processed/train_multimol.jsonl`.
- For code-level details on the collection flow, see `../data_collection/README.md`.
