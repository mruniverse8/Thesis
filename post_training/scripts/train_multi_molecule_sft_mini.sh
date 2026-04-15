#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

cd "$PROJECT_ROOT"

exec bash post_training/scripts/train_multi_molecule_sft.sh --config configs/multi_molecule_sft_mini.yaml "$@"
