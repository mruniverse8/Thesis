# Segmented GFlowNet Evaluation Merge Scripts

This folder keeps merge utilities separate from training/evaluation scripts.

## Merge Segmented Beam Evaluation

Use `merge_segmented_gflownet_eval.py` after running the segmented full-test
notebook. The script does not load the model. It recomputes metrics from saved
generation JSONL rows and target molecules.

Expected input layout:

```text
DIR_RUN/
  test_iter_000000_to_000500/
    manifest.json
    generations.jsonl
    metrics_segment.json
  test_iter_000500_to_001000/
    manifest.json
    generations.jsonl
    metrics_segment.json
```

Run:

```bash
python scripts_merge/merge_segmented_gflownet_eval.py \
  --dir-run /path/to/DIR_RUN \
  --output-dir /path/to/outputs_DIR_RUN \
  --expected-test-examples 10000
```

Add `--strict-full-coverage` when the script should fail if any dataset index
from `0` to `expected-test-examples - 1` is missing.

Important outputs:

```text
outputs_DIR_RUN/
  generations_merged.jsonl
  metrics_merged.json
  candidate_assessments.jsonl
  segment_summary.csv
  merge_manifest.json
  plots/
    segment_valid_fraction.png
    segment_accepted_unique_count.png
    cumulative_rows_by_segment.png
    cumulative_candidates_by_segment.png
    merged_max_dice_histogram.png
    merged_candidate_validity_acceptance.png
```

Final thesis/report values should come from `metrics_merged.json`, not from
adding values across `metrics_segment.json` files. Accepted-unique, novelty,
and internal diversity are global metrics and must be recomputed after merging.
