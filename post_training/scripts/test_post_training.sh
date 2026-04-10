#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CONDA_ENV="${CONDA_ENV:-thesis_biot5_sft}"

cd "$PROJECT_ROOT"

if [ "$#" -eq 0 ]; then
  set -- post_training/tests -q
fi

exec conda run --no-capture-output -n "$CONDA_ENV" python -m pytest "$@"
