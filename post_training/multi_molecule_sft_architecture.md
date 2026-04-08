# Multi-Molecule SFT Architecture

## Summary

This document defines the supervised fine-tuning stage that comes after the current single-molecule pretraining-style SFT.

The existing training path in `src/` learns:

- description -> one SELFIES string

The new post-training SFT must learn:

- `pdesc + div` -> `M1:K`

where `M1:K` is a serialized list of molecule SELFIES strings for the same description. This stage is the initialization point for PPO and replaces the current single-target assumption in `src/datasets.py`, `src/prompting.py`, `src/training.py`, and `src/evaluation.py` for post-training experiments.

The implementation should stay in the current repo style:

- custom PyTorch training loop
- `T5ForConditionalGeneration` / BioT5+ backbone
- SELFIES-based targets
- repo-local preprocessing and evaluation utilities

## Training Objective

Train a seq2seq policy `pi_sft(M1:K | pdesc + div)` with teacher-forced cross-entropy over the full serialized target sequence.

This stage is not PPO. It is the supervised initialization checkpoint for the later RL stage. The current `scripts/train_sft.py` remains the baseline single-molecule trainer; the multi-molecule SFT is a separate post-training path.

## Data Contract

Each preprocessed example must follow this schema:

```python
MultiMoleculeExample = {
    "id": str,
    "description": str,
    "target_selfies_list": list[str],
    "target_smiles_list": list[str] | None,
}
```

Required rules:

1. `description` is normalized with the same whitespace policy used by `src.prompting.normalize_free_text`.
2. Every entry in `target_selfies_list` is normalized with the same whitespace policy used by `src.prompting.normalize_selfies_text`.
3. Every target SELFIES string must decode successfully before the example is written.
4. Duplicate molecules are removed using canonical SMILES, keeping the first valid occurrence.
5. Invalid molecules are dropped during preprocessing, not during training.
6. The stored list order is the canonical training order.
7. If the upstream source does not define an order, preprocessing must sort the deduplicated list by canonical SMILES before writing JSONL.
8. Examples with zero valid molecules are rejected.

Default assumptions:

- the external dataset already provides multiple accepted molecules per description
- the list order in that dataset is meaningful unless explicitly missing
- SELFIES is the only decoder-side representation used in training

## Prompt and Target Format

### Prompt

The prompt must follow the current BioT5+ style but switch from single-molecule completion to diverse set generation.

Canonical prompt template:

```text
Definition: You are given a molecule description in English. Your job is to generate a diverse set of molecule SELFIES that fit the description.

Now complete the following example -
Input: {normalized_description}
Output:
```

Public helper:

```python
def build_diverse_text2mol_prompt(description: str) -> str: ...
```

### Sequence serialization

Each target is serialized as one decoder sequence:

```text
<bom>{m1}<mol_sep>{m2}<mol_sep>...{mk}<eom>
```

Rules:

1. `<mol_sep>` is a required additional special token.
2. `<bom>` and `<eom>` keep their current semantics.
3. Molecule separators appear only between molecules, never after the final molecule.
4. Empty molecules are not allowed.
5. The final stop token is always `<eom>`.

Public helpers:

```python
def serialize_molecule_sequence(selfies_list: list[str]) -> str: ...
def parse_molecule_sequence(text: str) -> list[str]: ...
```

Parsing rules:

1. Remove `<bom>` and `<eom>` before splitting.
2. Split only on `<mol_sep>`.
3. Normalize each segment with `normalize_selfies_text`.
4. Drop empty segments created by malformed output.
5. Return the surviving list in order.

Example:

```text
<bom>[C][C][O]<mol_sep>[C][C][C]<mol_sep>[O][=C][O]<eom>
```

## Planned Implementation Units

The post-training SFT implementation should be split into the following repo-local modules:

- `post_training/sft_prompting.py`
  - `build_diverse_text2mol_prompt`
- `post_training/sequence_format.py`
  - `<mol_sep>` constant
  - serialization and parsing helpers
- `post_training/sft_dataset.py`
  - JSONL loader
  - validation and preprocessing helpers
  - `MultiMoleculeDataset`
  - `MultiMoleculeCollator`
- `post_training/sft_training.py`
  - model setup
  - dataloaders
  - train/eval loop
  - checkpoint save/load helpers
- `post_training/sft_evaluation.py`
  - generation-time parsing
  - sequence metrics

The existing `src/tokenizer_utils.py` functions should be reused where possible, with one change: the tokenizer preparation step must add `<mol_sep>` together with `<bom>` and `<eom>`.

## Dataset and Collator Behavior

### Dataset output

Each dataset item should expose:

```python
{
    "id": str,
    "description": str,
    "target_selfies_list": list[str],
    "prompt": str,
    "target_text": str,
}
```

`target_text` is always the serialized output from `serialize_molecule_sequence`.

### Collator

The collator must behave like the current `TextToSelfiesCollator`, with sequence-aware metadata added for evaluation and RL handoff.

Expected collator output:

```python
{
    "input_ids": ...,
    "attention_mask": ...,
    "labels": ...,
    "example_ids": list[str],
    "prompts": list[str],
    "descriptions": list[str],
    "target_texts": list[str],
    "target_selfies_lists": list[list[str]],
}
```

Tokenizer/truncation rules:

1. Source tokenization uses `max_source_length`.
2. Target tokenization uses `max_target_length`.
3. Label padding tokens are replaced with `-100`.
4. Truncation is allowed only at tokenization time; preprocessing must not silently shorten molecule lists.
5. `max_target_length` must be sized for the worst-case serialized sequence, not the average single molecule.

## Config Contract

The multi-molecule SFT config should mirror `configs/sft_chebi20.yaml` but add sequence-aware fields.

Minimum required keys:

```yaml
seed: 42

model:
  name: QizhiPei/biot5-plus-base
  tokenizer_name: QizhiPei/biot5-plus-base
  base_tokenizer_name: google/t5-v1_1-base
  selfies_vocab_path: ../Other-projects/BioT5/dict/selfies_dict.txt
  molecule_separator_token: <mol_sep>

data:
  train_file: data/<dataset>/processed/train_multimol.jsonl
  validation_file: data/<dataset>/processed/validation_multimol.jsonl
  test_file: data/<dataset>/processed/test_multimol.jsonl
  max_source_length: 512
  max_target_length: 768
  max_molecules_per_sequence: 8
  num_workers: 2

training:
  output_dir: outputs/<dataset>_multimol_sft
  device: auto
  mixed_precision: auto
  per_device_train_batch_size: 2
  per_device_eval_batch_size: 4
  gradient_accumulation_steps: 8
  num_epochs: 3
  learning_rate: 2.0e-5
  weight_decay: 0.01
  warmup_ratio: 0.03
  max_grad_norm: 1.0
  log_every: 10
  save_every_epochs: 1
  num_beams: 1
  generation_max_new_tokens: 768
```

Defaults chosen for implementation:

- `molecule_separator_token = "<mol_sep>"`
- `max_molecules_per_sequence = 8`
- `max_target_length = 768`
- `generation_max_new_tokens = 768`

If the training dataset contains more than `max_molecules_per_sequence` valid targets for an example, preprocessing must keep the first `max_molecules_per_sequence` after deduplication.

## Training Flow

1. Load the tokenizer with the current SELFIES vocabulary expansion logic.
2. Add `<mol_sep>` as an additional special token.
3. Resize the BioT5+ embeddings if the tokenizer grew.
4. Load `MultiMoleculeDataset` for train and validation splits.
5. Train with the same teacher-forced seq2seq loss shape already used in `src/training.py`.
6. Save checkpoints in the same style as the current trainer:
   - model weights
   - training tokenizer
   - decoder tokenizer
   - resolved config
   - metrics JSON
7. Mark the best checkpoint by validation loss.

No RL signals are used in this stage.

## Evaluation Contract

Generation-time evaluation must operate on parsed molecule lists rather than raw strings only.

For each prediction:

1. Generate one serialized sequence.
2. Parse it with `parse_molecule_sequence`.
3. Validate every predicted molecule via SELFIES decode and canonical SMILES conversion.
4. Build both sequence-level and molecule-level metrics.

Required metrics:

- `num_examples`
- `exact_sequence_match`
- `exact_set_match`
- `all_molecules_valid_rate`
- `mean_predicted_molecule_count`
- `mean_unique_predicted_molecule_count`
- `duplicate_prediction_rate`
- `position_exact_match_rate`

Metric definitions:

- `exact_sequence_match`: exact equality of ordered normalized SELFIES lists
- `exact_set_match`: equality of canonical SMILES sets, ignoring order
- `all_molecules_valid_rate`: fraction of examples where every predicted molecule is valid
- `duplicate_prediction_rate`: fraction of examples with at least one duplicate predicted molecule after canonicalization
- `position_exact_match_rate`: mean fraction of positions where predicted and target molecules match exactly in order

## RL Handoff Contract

The PPO stage must consume the same sequence format and data schema.

The multi-molecule SFT checkpoint is the policy initialization checkpoint for PPO. The PPO code should assume:

- identical tokenizer
- identical prompt builder
- identical `<bom>`, `<mol_sep>`, `<eom>` serialization
- description-conditioned target molecule lists available in the training examples

This prevents SFT and PPO from learning different output grammars.

## Test Plan

The implementation of this architecture must include tests for:

1. prompt formatting for `build_diverse_text2mol_prompt`
2. round-trip serialization and parsing with `<mol_sep>`
3. preprocessing removal of invalid targets
4. deduplication by canonical SMILES while preserving the first valid occurrence
5. deterministic ordering when the upstream source is unordered
6. dataset item construction with correct `target_text`
7. collator label masking and metadata retention
8. generation parsing for:
   - valid sequences
   - missing `<eom>`
   - repeated `<mol_sep>`
   - empty segments
   - malformed SELFIES
9. evaluation metrics for exact sequence match, exact set match, and duplicate detection

## Assumptions

- The current `scripts/train_sft.py` is left unchanged and remains the single-molecule baseline.
- The new multi-molecule SFT is implemented under `post_training/` rather than by overloading the current `src/` path.
- The external dataset already exists or will be exported in the schema defined above.
- The decoder target is always SELFIES, never SMILES.
- Sequence formatting is strict and separator-based; raw concatenation without a separator is intentionally unsupported.
