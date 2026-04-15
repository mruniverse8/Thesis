# Staged Multi-Molecule SFT

This page is the compact reading guide for the supervised post-training stage.

## Target Format

```text
<bom>{m1}<eom> <bom>{m2}<eom> ... <bom>{mK}<eom>
```

Each molecule is wrapped independently, and staged molecules are separated by one space.

## Main Files To Read

1. [post_training/sft_multi/trainer.py](../../sft_multi/trainer.py)
2. [post_training/shared/sequence.py](../../shared/sequence.py)
3. [post_training/sft_multi/dataset.py](../../sft_multi/dataset.py)
4. [post_training/sft_multi/collator.py](../../sft_multi/collator.py)
5. [post_training/sft_multi/prompting.py](../../sft_multi/prompting.py)
6. [post_training/tests/sft_multi/test_dataset.py](../../tests/sft_multi/test_dataset.py)

## Entry Flow

```text
trainer.main
  -> resolve_multi_molecule_sft_config_paths
  -> run_multi_molecule_sft
  -> prepare_sft_tokenizers
  -> MultiMoleculeDataset
  -> MultiMoleculeCollator
  -> serialize_staged_target
  -> supervised training loop
```

## Run It

- Wrapper script: [post_training/scripts/train_multi_molecule_sft.sh](../../scripts/train_multi_molecule_sft.sh)
- Default config: [configs/multi_molecule_sft.yaml](../../../configs/multi_molecule_sft.yaml)
- Tests: [post_training/scripts/test_post_training.sh](../../scripts/test_post_training.sh)

The config accepts a top-level `tracking:` block for basic `wandb` or `comet` logging.
