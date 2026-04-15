#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

cd "$PROJECT_ROOT"

exec bash post_training/scripts/train_molecule_wise_ppo.sh --config configs/molecule_wise_ppo_mini.yaml "$@"
