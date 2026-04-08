# Reward Utils Test Guide

This note explains what to test in `reward_utils`, why those tests matter, and how to run them in the `thesis_biot5_sft` conda environment.

## Purpose

The goal of this test phase is to verify three things before PPO integration:

- molecule parsing behaves correctly for different representations
- similarity and reward calculations are stable across SMILES and SELFIES inputs
- invalid or duplicate molecules do not silently corrupt the reward signal

This is still a focused chemistry/unit test set, but it now covers all 10 high-value ideas listed below.

## 10 Candidate Ideas

These were the main test ideas considered for this stage:

1. Parse valid SMILES into canonical SMILES.
2. Parse valid SELFIES into canonical SMILES.
3. Auto-detect SELFIES even when it contains spaces.
4. Fail cleanly when the wrong representation is forced.
5. Confirm the same molecule in SMILES and SELFIES becomes the same canonical molecule.
6. Confirm Dice and Tanimoto are `1.0` for the same molecule across different representations.
7. Check `rmatch` when candidate and targets use different representations.
8. Check `rdiv` when previous molecules use different representations.
9. Check sequence scoring with mixed valid, duplicate, and invalid candidates.
10. Check batch-scale reproducibility on a small curated prompt-target set.

## Implemented Coverage

All 10 ideas are now covered by concrete tests.

### 1. Parse valid SMILES into canonical SMILES

Current test:

- `test_parse_smiles_candidate_to_canonical_smiles`

### 2. Parse valid SELFIES into canonical SMILES

Current test:

- `test_parse_selfies_candidate_to_canonical_smiles`

### 3. Auto-detect SELFIES even when it contains spaces

Current test:

- `test_auto_parser_accepts_spaced_selfies`

### 4. Fail cleanly when the wrong representation is forced

Current test:

- `test_forcing_selfies_representation_on_plain_smiles_fails`

### 5. Confirm the same molecule in SMILES and SELFIES becomes the same canonical molecule

Current test:

- `test_smiles_and_selfies_for_same_molecule_are_duplicates`

### 6. Confirm Dice and Tanimoto are `1.0` for the same molecule across different representations

Current tests:

- `test_self_similarity_is_one_for_dice_and_tanimoto`
- `test_similarity_is_one_across_smiles_and_selfies_for_same_molecule`

### 7. Check `rmatch` when candidate and targets use different representations

Current tests:

- `test_total_reward_supports_cross_representation_inputs`
- `test_rmatch_supports_cross_representation_inputs`

### 8. Check `rdiv` when previous molecules use different representations

Current tests:

- `test_total_reward_supports_cross_representation_inputs`
- `test_rdiv_supports_cross_representation_inputs`

### 9. Check sequence scoring with mixed valid, duplicate, and invalid candidates

Current tests:

- `test_sequence_scoring_tracks_first_step_and_duplicates`
- `test_sequence_scoring_handles_mixed_representations_and_invalid_inputs`

### 10. Check batch-scale reproducibility on a small curated prompt-target set

Current test:

- `test_curated_sequence_scoring_is_reproducible`

## Still Out of Scope

These are useful later, but they are beyond the current 10-idea completion target:

- large curated regression sets across many prompts
- performance benchmarking for fingerprint generation
- property-based fuzz testing for malformed SELFIES
- sensitivity sweeps over `alpha`, `beta`, or fingerprint size
- PPO-loop tests that consume reward outputs during training

## Commands

Run only the reward-utils tests:

```bash
conda run -n thesis_biot5_sft pytest -q tests/test_reward_utils.py
```

Run a single focused test:

```bash
conda run -n thesis_biot5_sft pytest -q tests/test_reward_utils.py -k "cross_representation"
```

Run with verbose names:

```bash
conda run -n thesis_biot5_sft pytest tests/test_reward_utils.py -vv
```

## Files to Read While Testing

- [validation.py](./validation.py)
- [similarity.py](./similarity.py)
- [rewards.py](./rewards.py)
- [test_reward_utils.py](../tests/test_reward_utils.py)

## Expected Outcomes

When the full 10-idea suite passes, you can be confident that:

- parsing works for SMILES, SELFIES, and auto-detected SELFIES
- cross-representation equivalence is preserved after canonicalization
- reward functions can compare SELFIES candidates against SMILES targets and previous molecules
- duplicates reduce diversity reward as expected
- invalid molecules collapse to zero reward instead of producing undefined behavior
- repeated scoring of the same curated set is deterministic
