#!/usr/bin/env bash
set -euo pipefail
if [[ $# -lt 4 || $# -gt 6 ]]; then
    echo "Usage: bash scripts/test_isic_setting1.sh FOLD CHECKPOINT DATA_ROOT ISIC_BASE_PATH [VARIANT] [SEED]" >&2
    exit 2
fi
FOLD="$1"
CHECKPOINT="$2"
DATA_ROOT="$3"
BASE_PATH="$4"
VARIANT="${5:-full}"
SEED_VALUE="${6:-2025}"
PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"
"${PYTHON:-python}" scripts/run_ablation.py test --dataset isic \
    --variant "$VARIANT" --fold "$FOLD" --checkpoint "$CHECKPOINT" \
    --data-root "$DATA_ROOT" --isic-base-path "$BASE_PATH" \
    --workers "${NUM_WORKERS:-4}" --seed "$SEED_VALUE"
