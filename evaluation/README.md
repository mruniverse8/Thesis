# Evaluation Metrics

This folder contains the molecule-level evaluation layer for the thesis SFT pipeline.

The existing checkpoint evaluation in [`src/evaluation.py`](../src/evaluation.py) already generated:

- exact match
- valid SELFIES rate
- repair rate

This addition extends that flow with chemistry-aware metrics:

- `Accepted & Unique`
- `NCircles`
- `IntDiv`

## Files Added

### New `evaluation/` package

- [`evaluation/__init__.py`](./__init__.py)
  Re-exports the public evaluation helpers used by the rest of the repo.
- [`evaluation/config.py`](./config.py)
  Holds the metric defaults:
  - Morgan radius `2`
  - Morgan bits `2048`
  - acceptance Dice threshold `0.7`
  - `NCircles` Tanimoto threshold `0.6`
- [`evaluation/grouping.py`](./grouping.py)
  Loads the evaluated split, groups reference molecules by shared `description`, and resolves which reference set each prediction should be compared against.
- [`evaluation/metrics.py`](./metrics.py)
  Core metric logic:
  - candidate parsing and acceptance checks
  - unique accepted molecule collapse by canonical SMILES
  - exact `NCircles`
  - `IntDiv`
- [`evaluation/reporting.py`](./reporting.py)
  Builds the final split-level summary and the per-description report.
- [`evaluation/README.md`](./README.md)
  This guide.

### Existing files updated to use the new package

- [`src/evaluation.py`](../src/evaluation.py)
  Still runs generation, but now also runs molecule-level evaluation on the saved predictions.
- [`scripts/eval_sft.py`](../scripts/eval_sft.py)
  CLI entrypoint for checkpoint evaluation. It now exposes flags for the molecule metrics and writes the new report files.

## General Pipeline

The current evaluation pipeline is now:

1. Load a trained checkpoint.
2. Generate one molecule prediction per example in the requested split.
3. Save the raw prediction dump to JSONL.
4. Compute generation metrics:
   - exact match
   - valid SELFIES rate
   - repaired SELFIES rate
5. Load the evaluated split again and group the reference molecules by `description`.
6. For each prediction:
   - parse and canonicalize the generated molecule
   - compare it against all reference molecules for the same description
   - mark it as accepted if Dice similarity is `> 0.7`
7. Collapse accepted molecules by canonical SMILES.
8. Compute:
   - `Accepted & Unique`
   - `NCircles`
   - `IntDiv`
9. Write:
   - split-level molecule summary
   - per-description JSONL report

## Metric Definitions

### Accepted & Unique

A generated molecule is accepted if:

- it is chemically valid after parsing and canonicalization
- its Dice similarity to at least one target-reference molecule for the same description is strictly greater than `0.7`

The metric then counts unique accepted molecules by canonical SMILES.

Implementation notes:

- reference molecules are grouped by shared `description` in the evaluated split
- uniqueness is computed after RDKit canonicalization, so equivalent SMILES and SELFIES collapse to one molecule

### NCircles

`NCircles_h` is computed over the accepted unique molecules.

- build a compatibility rule using Tanimoto similarity on Morgan fingerprints
- two molecules are compatible only when their Tanimoto similarity is below `h`
- `NCircles_h` is the size of the largest subset in which every pair is compatible

The v1 implementation uses:

- `h = 0.6`
- an exact maximum-clique search on the compatibility graph

### IntDiv

`IntDiv` is the mean pairwise structural distance across the accepted unique molecules:

`IntDiv = average(1 - TanimotoSimilarity(m_i, m_j))`

If fewer than two accepted unique molecules exist, the metric returns `0.0`.

## Existing Code Reuse

This package intentionally reuses the chemistry utilities that already exist in `reward_utils/`:

- [`reward_utils/validation.py`](../reward_utils/validation.py): parse SMILES/SELFIES, canonicalize, validate
- [`reward_utils/fingerprints.py`](../reward_utils/fingerprints.py): build Morgan fingerprints
- [`reward_utils/similarity.py`](../reward_utils/similarity.py): Dice and Tanimoto primitives
- [`reward_utils/rewards.py`](../reward_utils/rewards.py): existing reward logic using the same chemistry building blocks

This avoids duplicating parsing or fingerprint rules between training-time rewards and evaluation-time metrics.

## How To Run

### Standard checkpoint evaluation

Run the current checkpoint evaluation script:

```bash
conda run -n thesis_biot5_sft python scripts/eval_sft.py \
  --config configs/sft_chebi20.yaml \
  --checkpoint outputs/chebi20_sft/checkpoints/best \
  --split validation
```

### Override metric thresholds

```bash
conda run -n thesis_biot5_sft python scripts/eval_sft.py \
  --config configs/sft_chebi20.yaml \
  --checkpoint outputs/chebi20_sft/checkpoints/best \
  --split validation \
  --acceptance-dice-threshold 0.7 \
  --ncircles-tanimoto-threshold 0.6
```

### Write explicit output files

```bash
conda run -n thesis_biot5_sft python scripts/eval_sft.py \
  --config configs/sft_chebi20.yaml \
  --checkpoint outputs/chebi20_sft/checkpoints/best \
  --split validation \
  --prediction-file outputs/chebi20_sft/validation_predictions.jsonl \
  --molecule-metrics-file outputs/chebi20_sft/validation_molecule_metrics.json \
  --group-metrics-file outputs/chebi20_sft/validation_molecule_metrics_by_group.jsonl
```

### Skip molecule metrics

If you only want the original generation metrics:

```bash
conda run -n thesis_biot5_sft python scripts/eval_sft.py \
  --config configs/sft_chebi20.yaml \
  --checkpoint outputs/chebi20_sft/checkpoints/best \
  --split validation \
  --skip-molecule-metrics
```

## Output Files

### Prediction dump

Default path:

- `outputs/.../<split>_predictions.jsonl`

Each record contains:

- example id
- description
- target SELFIES and target SMILES
- generated text
- normalized generated SELFIES
- decoded generated SMILES
- exact match
- validity flag
- repair flag

### Generation metrics

Default path:

- `outputs/.../<split>_generation_metrics.json`

Contains:

- `num_examples`
- `exact_match`
- `valid_selfies_rate`
- `repaired_selfies_rate`

### Molecule summary

Default path:

- `outputs/.../<split>_molecule_metrics.json`

Contains:

- `num_generated`
- `num_groups`
- `num_valid`
- `num_accepted`
- `num_unique_accepted`
- `accepted_unique_count`
- `accepted_unique_smiles`
- `accepted_predictions`
- `ncircles`
- `intdiv`
- threshold settings
- fingerprint settings

### Per-description report

Default path:

- `outputs/.../<split>_molecule_metrics_by_group.jsonl`

Each JSON line contains:

- `description`
- `num_predictions`
- `num_references`
- `reference_smiles`
- `num_valid`
- `num_accepted`
- `num_unique_accepted`
- `accepted_unique_smiles`
- `accepted_predictions`
- `ncircles`
- `intdiv`

## Notes

- The grouping rule is description-based, not row-based. A prediction is compared against all reference molecules that share the same description in the evaluated split.
- The current ChEBI-20 processed splits mostly look like one target per description, but the code is written to support multi-reference descriptions later.
- `NCircles` and `IntDiv` are computed on accepted unique molecules, not on all predictions.
