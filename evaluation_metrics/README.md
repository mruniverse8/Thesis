# Evaluation Metrics

This folder contains post-training evaluation metrics for generated molecules.
The implementation uses the same RDKit parsing, canonical SMILES, and Morgan
fingerprint defaults as the training reward code.

## Formalism

Let `G = {g_i}` be generated molecules and let `T_j = {t_jm}` be the target
molecules for the prompt/example that produced `g_j`. Invalid generated
molecules are not accepted.

Accepted:

```text
accept(g_j) = valid(g_j) and max_m Dice(fp(g_j), fp(t_jm)) > d
```

with default `d = 0.7`. `Accepted & Unique` is the number of distinct canonical
SMILES among accepted molecules:

```text
AU = | unique_canonical_smiles({g_j : accept(g_j)}) |
```

NCircles at threshold `h` is the largest accepted subset whose members are all
mutually dissimilar:

```text
NCircles_h(A) =
  max |S|
  subject to S subset A_unique
  and for every x != y in S: Tanimoto(fp(x), fp(y)) < h
```

This is a maximum independent-set problem on the graph whose edges connect
molecules with Tanimoto similarity at least `h`. The implementation solves the
equivalent maximum-clique problem exactly up to
`n_circles_exact_max_molecules` accepted molecules, then falls back to a
deterministic greedy approximation for larger accepted sets.

NCircles is disabled by default because exact maximum-clique search is
exponential. Enable it explicitly with `EvaluationMetricConfig(compute_n_circles=True)`
or the CLI `--compute-n-circles` flag.

Internal diversity is the average pairwise distance between accepted unique
molecules:

```text
IntDiv(A) =
  0, if |A_unique| < 2
  1 - (2 / (n * (n - 1))) * sum_{i < j} Tanimoto(fp(a_i), fp(a_j)), otherwise
```

## 06_v2 After-Training Plan

1. Generate evaluation rollouts from the final or best GFlowNet checkpoint on
   the held-out LPM24 test examples.
2. For each prompt/example, keep the generated stage SELFIES and that example's
   `target_selfies_list`.
3. Build `GenerationGroup(group_id, candidates, targets)` objects.
4. Run `evaluate_generation_groups(...)` and write `result.to_dict()` to
   `OUTPUT_DIR / "evaluation_metrics.json"`.
5. Track at least:
   `accepted_unique_count`, `internal_diversity`, `num_valid_candidates`, and
   `num_candidates`. Track `n_circles` and `n_circles_exact` only when NCircles
   is explicitly enabled.

Minimal notebook-side shape:

```python
from evaluation_metrics import (
    EvaluationMetricConfig,
    GenerationGroup,
    MoleculeInput,
    evaluate_generation_groups,
)

groups = [
    GenerationGroup(
        group_id=example["id"],
        candidates=tuple(MoleculeInput(text=s, representation="selfies") for s in generated_selfies),
        targets=tuple(MoleculeInput(text=t, representation="selfies") for t in example["target_selfies_list"]),
    )
    for example, generated_selfies in evaluated_outputs
]

result = evaluate_generation_groups(
    groups,
    config=EvaluationMetricConfig(
        acceptance_dice_threshold=0.7,
        compute_n_circles=False,
        n_circles_tanimoto_threshold=0.6,
    ),
)
metrics = result.to_dict(include_assessments=True)
```

For a saved JSONL of generated molecules:

```bash
python -m evaluation_metrics.cli \
  --generated-jsonl "$OUTPUT_DIR/eval_generations.jsonl" \
  --targets-jsonl "$TEST_EVAL_JSONL" \
  --output "$OUTPUT_DIR/evaluation_metrics.json" \
  --include-assessments
```

Expected generated JSONL fields are intentionally permissive. The CLI looks for
candidate fields such as `generated_selfies_list`, `sampled_selfies_list`,
`candidates`, `generated_selfies`, `sampled_selfies`, or `selected_selfies`;
target fields such as `target_selfies_list` can be on the same row or in
`--targets-jsonl` keyed by `id`/`example_id`.
