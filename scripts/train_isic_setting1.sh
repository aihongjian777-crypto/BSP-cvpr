#!/usr/bin/env bash
set -euo pipefail
if [[ $# -lt 3 ]]; then
    echo "Usage: bash scripts/train_isic_setting1.sh DATA_ROOT PRETRAINED_WEIGHTS ISIC_BASE_PATH [FOLD ...]" >&2
    exit 2
fi
DATA_ROOT="$1"
WEIGHTS="$2"
BASE_PATH="$3"
shift 3
if (($#)); then
    FOLDS=("$@")
else
    FOLDS=(1)
fi
PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"
for FOLD in "${FOLDS[@]}"; do
    "${PYTHON:-python}" scripts/run_ablation.py train --dataset isic \
        --variant "${ABLATION_VARIANT:-full}" --fold "$FOLD" \
        --data-root "$DATA_ROOT" --pretrained-weights "$WEIGHTS" --isic-base-path "$BASE_PATH" \
        --steps "${N_STEPS:-50000}" --workers "${NUM_WORKERS:-4}" --seed "${SEED:-2025}"
done
