# Post-Training Tracking

This folder contains the minimal experiment tracking layer used by `post_training`.

Supported backends:

- `wandb`
- `comet`

The runtime code lives in [tracker.py](./tracker.py). The example config snippets below are intentionally small and only enable basic scalar logging.

## Environment

Set one of these before launching training:

- `WANDB_API_KEY=...`
- `COMET_API_KEY=...`

Tracking is fail-fast when enabled. If the backend package is missing or authentication is not available, the run raises instead of silently disabling remote logging.

## What Gets Logged

- resolved run config
- output directory and stage name
- SFT epoch metrics
- PPO iteration metrics
- short final summary

Local files such as `history.json` and `run_summary.json` are still written exactly as before.
