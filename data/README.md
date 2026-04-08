# Datasets

This folder documents the dataset setup for the simplified BioT5+ SFT pipeline in this repo.

The old data-collection experiments were archived under `../legacy/data_collection/` and are no longer part of the active repo workflow.

## Current Dataset

The first working pipeline targets `liupf/ChEBI-20-MM` from Hugging Face:

- Source: https://huggingface.co/datasets/liupf/ChEBI-20-MM
- Relevant fields used by this pipeline: `CID`, `description`, `SMILES`, `SELFIES`
- Task direction: `description -> SELFIES`

The preprocessing script prefers the dataset's existing `SELFIES` column and only falls back to `SMILES -> SELFIES` conversion if the `SELFIES` field is missing.

## Setup

From the thesis repo root:

```bash
pip install -r requirements.txt
python scripts/download_chebi20.py --output-dir data/chebi20
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

## Notes

- Processed dataset files are ignored by `.gitignore`; regenerate them locally.
- The training config expects the processed files under `data/chebi20/processed/`.
- `language-plus-molecules/LPM-24_train` is not wired into the pipeline yet. Treat that as the next dataset integration step after the ChEBI-20 baseline is stable.
