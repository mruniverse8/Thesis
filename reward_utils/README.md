# `reward_utils`

This package is now a compatibility layer for older imports.

The canonical molecule and chemistry implementation lives under:

- `../molecules/`
- `../molecules/README.md`

This workspace isolates the reward logic needed to reproduce the reinforcement-learning stage from:

- arXiv `2410.03138v2`

The reward module now supports three configurable variants:

- `reward_var1` keeps the `v2` Appendix B.2 design:
  `rmatch(m_k, p_desc) = max_{m in M_D(p_desc)} D(m, m_k)^alpha`
  `rdiv(m_k, {m_i}_{i=1}^{k-1}) = 1 - max_{m in {m_i}_{i=1}^{k-1}} T(m_k, m)^beta`
- `reward_var2` is the default:
  `total_reward = rmatch + plus_valid` for valid candidates
  `total_reward = rmatch` for invalid candidates
- `reward_var3` combines both:
  `total_reward = match_weight * rmatch + diversity_weight * rdiv + plus_valid` for valid candidates
  `total_reward = match_weight * rmatch + diversity_weight * rdiv` for invalid candidates

where:

- `D` is Dice similarity
- `T` is Tanimoto similarity
- both are computed on Morgan fingerprints via RDKit

## Tooling

Use the thesis conda environment and install RDKit there:

```bash
conda install -n thesis_biot5_sft -c conda-forge rdkit
```

The current project already depends on:

- `selfies`
- `pytest`

## File Layout

Canonical runtime ownership:

- `molecules/parsing.py`: SELFIES/SMILES parsing, canonicalization, duplicate checks
- `molecules/selfies.py`: SELFIES normalization, wrapping, repair, and decode helpers
- `molecules/fingerprints.py`: Morgan fingerprint generation
- `molecules/similarity.py`: Dice and Tanimoto similarity helpers
- `molecules/rewards/scoring.py`: `rmatch`, `rdiv`, total reward, and sequence scoring

Compatibility wrappers retained here:

- `defaults.py`: paper-grounded reward defaults and PPO hyperparameter references
- `validation.py`: SELFIES/SMILES parsing, canonicalization, duplicate checks
- `fingerprints.py`: Morgan fingerprint generation
- `similarity.py`: Dice and Tanimoto similarity helpers
- `rewards.py`: `rmatch`, `rdiv`, total reward, and sequence scoring
- `reward_pipeline_demo.ipynb`: standalone walkthrough of the reward pipeline with executable examples
- `plan.md`: reproduction spec for this workspace

## Paper-Grounded Defaults

- fingerprint radius: `2`
- fingerprint bits: `2048`
- `alpha = 0.5`
- `plus_valid = 0.8`
- default reward variant: `reward_var2`
- `beta = 1.0`
- ChEBI-20 override: `beta = 2.0`
- reward amplification: `8.0`

PPO-side reference hyperparameters captured here for later integration:

- PPO iterations: `200`
- mini-batch size: `8`
- batch size: `128`
- learning rate: `5e-5`
- KL penalty: `0.01`
- max sequence length: `2560`
- LoRA rank: `16`
- LoRA alpha: `32`

## Notes

- `score_candidate_sequence(...)` is the intended bridge into future PPO code.
- Invalid molecules yield zero reward rather than raising.
- `reward_var1` remains available for ablations that need the legacy match-plus-diversity formula.
- `reward_var3` is available when you want diversity pressure without dropping the validity bonus.
- Duplicate molecules are still surfaced explicitly in the reward breakdown even when the active reward variant does not penalize them.
- This workspace does not yet modify `src/training.py`; it is intentionally isolated until PPO integration begins.
