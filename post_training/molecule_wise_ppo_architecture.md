# Molecule-Wise PPO Architecture

## Summary

This document defines the reinforcement-learning stage that follows the new multi-molecule SFT checkpoint.

The implementation target is a repo-local PPO trainer for BioT5+ that follows the molecule-wise training pattern from Algorithm 2:

1. sample `m1`
2. update with PPO on `r(m1)`
3. sample `m2` conditioned on `m1`
4. update with PPO on `r(m2)`
5. continue until `mK`

The architecture is intentionally custom and stays aligned with the current codebase:

- no TRL dependency
- no framework pivot away from `T5ForConditionalGeneration`
- reward logic delegated to `reward_utils`
- training loop written in repo-local PyTorch

## Objective

Optimize a policy `pi_rl` initialized from the multi-molecule SFT checkpoint so that each generated molecule improves:

- description match
- within-sequence diversity

The scalar reward used by PPO for stage `k` is:

```text
r_k = amplified_reward(m_k, targets_for_description, m_1 ... m_{k-1})
```

where the source of truth is the current `reward_utils.compute_total_reward(...)` behavior and the PPO trainer uses `RewardBreakdown.amplified_reward` as the scalar optimization target.

## Starting Point and Dependencies

### Reused components

- tokenizer expansion from `src/tokenizer_utils.py`
- SELFIES parsing and normalization patterns already present in `src/selfies_utils.py`
- reward semantics from `reward_utils/rewards.py`
- project-wide device, mixed-precision, and checkpoint style from `src/training.py`

### New implementation units

The PPO path should be organized into the following modules:

- `post_training/ppo_types.py`
  - typed rollout and batch records
- `post_training/ppo_sequence.py`
  - stage-wise prefix construction
  - stop-token handling
  - sequence serialization/parsing reuse
- `post_training/reward_adapter.py`
  - bridge from generated SELFIES to `RewardBreakdown`
- `post_training/policy_model.py`
  - policy model wrapper
  - value head wrapper
  - reference model loader
- `post_training/rollout.py`
  - description sampling
  - molecule-wise generation
  - rollout record construction
- `post_training/ppo_trainer.py`
  - PPO update logic
  - minibatch loop
  - checkpointing
- `post_training/ppo_evaluation.py`
  - reward, validity, uniqueness, and diversity metrics

Required dependency additions for the later implementation:

- `peft` for LoRA adapters

Not included in v1:

- TRL
- distributed training framework changes
- off-policy replay

## Policy Representation

### Base policy

Use the multi-molecule SFT checkpoint as the initial policy.

Backbone:

- `T5ForConditionalGeneration`

Tokenizer:

- the exact tokenizer saved with the multi-molecule SFT checkpoint
- includes `<bom>`, `<eom>`, and `<mol_sep>`

### Trainable parameters

Train only:

- LoRA adapters attached to the policy model attention projections
- a scalar value head

Freeze:

- base SFT backbone weights
- reference policy weights

Default LoRA setup:

```yaml
lora_rank: 16
lora_alpha: 32
lora_dropout: 0.05
target_modules: ["q", "v"]
```

### Value function

The critic predicts one scalar value per molecule-generation stage, not one scalar per token.

Critic input:

- encoder input: the `pdesc + div` prompt
- decoder prefix: the serialized history up to, but not including, the current molecule

Critic output:

- a scalar `V_k` computed from the final decoder hidden state of the prefix before sampling stage `k`

This makes the PPO action unit equal to one molecule string plus its terminating token.

## State, Action, and Transition Design

### Encoder state

The encoder input stays constant for the full rollout:

```text
build_diverse_text2mol_prompt(description)
```

### Decoder prefix

The decoder prefix changes by stage.

Stage 1 prefix:

```text
<bom>
```

Stage 2 prefix after generating `m1`:

```text
<bom>{m1}<mol_sep>
```

Stage 3 prefix after generating `m1, m2`:

```text
<bom>{m1}<mol_sep>{m2}<mol_sep>
```

Final sequence after the last molecule:

```text
<bom>{m1}<mol_sep>...{mk}<eom>
```

### Action definition

A PPO action at stage `k` is the complete sampled molecule token sequence for `m_k`, plus its terminating control token:

- `<mol_sep>` if another molecule will follow
- `<eom>` if the rollout terminates

The rollout sampler must generate one stage at a time and stop stage decoding as soon as it emits either stop token.

### Termination rules

A rollout terminates when any of the following happens:

1. the model emits `<eom>`
2. the sampled number of molecules reaches `max_molecules_per_sequence`
3. the stage generation length reaches `max_stage_new_tokens`
4. the model emits an empty or unparseable molecule and `terminate_on_invalid_stage` is enabled

Defaults:

```yaml
max_molecules_per_sequence: 8
max_stage_new_tokens: 128
terminate_on_invalid_stage: true
```

If `terminate_on_invalid_stage` is true and a stage is invalid, that stage still receives reward `0.0` and is included in PPO, but no later stages are sampled for that rollout.

## Reward Integration

### Reward source

Use `reward_utils.compute_total_reward(...)` exactly as the scoring function for each stage.

For stage `k`:

- `candidate` = generated `m_k`
- `targets` = the full description-specific target molecule list from the training example
- `previous_candidates` = generated molecules from stages `1 ... k-1`

Representation choices:

- `candidate_representation = "selfies"`
- `target_representation = "selfies"` by default
- `previous_representation = "selfies"`

### PPO reward scalar

Use:

```text
reward_for_ppo = RewardBreakdown.amplified_reward
```

Log but do not optimize directly on:

- `match.reward`
- `diversity.reward`
- `total_reward`
- duplicate flag
- validity metadata

### Stage semantics

Stage 1 naturally gets `rdiv = 0.0` because there are no previous candidates.

Later stages get lower reward when:

- the candidate duplicates a previous molecule
- the candidate is highly similar to a previous molecule
- the candidate is invalid

## Rollout Record Contract

Every sampled stage must be stored as one independent PPO training record.

```python
StageTrajectory = {
    "example_id": str,
    "description": str,
    "target_selfies_list": list[str],
    "stage_index": int,
    "decoder_prefix_text": str,
    "sampled_selfies": str | None,
    "stop_token": str,
    "action_token_ids": list[int],
    "action_logprob_sum_old": float,
    "reference_logprob_sum": float,
    "value_old": float,
    "reward_breakdown": RewardBreakdown,
    "reward": float,
    "is_valid": bool,
    "is_duplicate": bool,
}
```

Rules:

1. `stage_index` is 1-based.
2. `action_token_ids` includes the stage stop token.
3. `action_logprob_sum_old` is the summed policy log-probability over the stage action tokens.
4. `reference_logprob_sum` is the summed SFT reference log-probability over the same tokens.
5. `value_old` is the scalar critic value for the prefix state before the action.

## PPO Update Rule

### Advantage

This trainer treats each stage as a one-step decision problem.

Advantage:

```text
A_k = reward_k - value_old_k
```

Return:

```text
R_k = reward_k
```

Advantages are standardized across the collected PPO batch before optimization.

No multi-step GAE is used in v1 because the action unit is already the whole molecule for one stage.

### Policy ratio

For each stage trajectory:

```text
ratio_k = exp(logprob_new_sum_k - action_logprob_sum_old_k)
```

Use standard clipped PPO:

```yaml
clip_range: 0.2
value_loss_coef: 0.5
entropy_coef: 0.0
max_grad_norm: 1.0
ppo_epochs_per_batch: 4
```

### KL penalty

KL is measured against the frozen SFT reference model over the same action tokens used for PPO.

Per-stage KL term:

```text
kl_k = logprob_new_sum_k - reference_logprob_sum_k
```

Penalized reward objective:

```text
policy_objective_k = clipped_ppo_objective_k - kl_penalty * kl_k
```

Default:

```yaml
kl_penalty: 0.01
```

This keeps the RL policy anchored to the SFT initialization while still allowing reward-seeking updates.

### Value loss

Value loss is mean-squared error between:

- current value prediction
- scalar return `R_k`

## Batch Construction

Each PPO iteration should do the following:

1. Sample `batch_size` descriptions from the RL training split.
2. For each description, run one full staged rollout up to termination.
3. Flatten all stage trajectories from the batch into one PPO dataset.
4. Shuffle stage trajectories.
5. Run `ppo_epochs_per_batch` epochs of minibatch updates with `mini_batch_size`.
6. Save iteration metrics and periodic checkpoints.

Default PPO config:

```yaml
ppo_iterations: 200
batch_size: 128
mini_batch_size: 8
learning_rate: 5.0e-5
kl_penalty: 0.01
max_sequence_length: 2560
max_molecules_per_sequence: 8
max_stage_new_tokens: 128
clip_range: 0.2
value_loss_coef: 0.5
entropy_coef: 0.0
lora_rank: 16
lora_alpha: 32
lora_dropout: 0.05
```

`max_sequence_length` is the absolute cap for encoder-plus-decoder tokens during rollout. If a rollout would exceed it, terminate the example immediately with `<eom>` and keep the stages already collected.

## Checkpointing and Artifacts

Each PPO checkpoint directory should contain:

- policy backbone with LoRA adapters
- value head weights
- tokenizer
- resolved PPO config
- training metrics JSON
- rollout summary JSONL for the saved iteration

Required logged metrics per PPO iteration:

- mean `amplified_reward`
- mean `match.reward`
- mean `diversity.reward`
- valid molecule rate
- duplicate molecule rate
- mean molecules per rollout
- mean unique molecules per rollout
- mean KL
- mean value loss
- mean policy loss

## Evaluation Contract

Evaluation must compare the multi-molecule SFT policy and the PPO policy using the same prompt set and generation settings.

Required evaluation outputs:

- generated serialized sequence
- parsed molecule list
- canonical SMILES list
- per-stage reward breakdowns
- sequence summary metrics

Required evaluation metrics:

- mean amplified reward
- mean total reward
- valid molecule rate
- unique molecule rate
- duplicate rate
- mean number of generated molecules
- best-match similarity against target sets

## Test Plan

The implementation of this architecture must include tests for:

1. stage-wise prefix construction
2. correct stopping on `<mol_sep>` and `<eom>`
3. one-stage-at-a-time rollout sampling
4. reward adapter correctness for:
   - first stage
   - duplicate later stage
   - novel later stage
   - invalid stage
5. correct storage of `action_logprob_sum_old`, `reference_logprob_sum`, and `value_old`
6. PPO ratio and clipped loss calculations
7. KL penalty calculation against the reference model
8. value loss target construction
9. rollout termination at:
   - invalid stage
   - max molecules
   - max stage tokens
   - max sequence length
10. checkpoint save/load consistency for adapters and value head

## Assumptions

- PPO starts only from the new multi-molecule SFT checkpoint, not from the current single-molecule baseline.
- The RL training data uses the same `MultiMoleculeExample` schema defined in `post_training/multi_molecule_sft_architecture.md`.
- The reward implementation in `reward_utils` remains the single source of truth for `rmatch`, `rdiv`, duplicate handling, and reward amplification.
- The v1 PPO trainer is single-process and repo-local.
- The PPO action unit is one complete molecule stage, not one token.
