# `reward_utils`

This workspace isolates the reward logic needed to reproduce the reinforcement-learning stage from:

- arXiv `2410.03138v2`

The implementation target follows the `v2` reward design in Appendix B.2:

- `rmatch(m_k, p_desc) = max_{m in M_D(p_desc)} D(m, m_k)^alpha`
- `rdiv(m_k, {m_i}_{i=1}^{k-1}) = 1 - max_{m in {m_i}_{i=1}^{k-1}} T(m_k, m)^beta`

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
- Duplicate molecules are surfaced explicitly in the reward breakdown and naturally collapse `rdiv` toward `0` through Tanimoto similarity.
- This workspace does not yet modify `src/training.py`; it is intentionally isolated until PPO integration begins.
