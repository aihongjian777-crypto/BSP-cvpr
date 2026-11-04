#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"
if [[ $# -ne 2 ]]; then
    echo "Usage: bash scripts/test_isic_setting1.sh FOLD CHECKPOINT" >&2
    exit 2
fi
FOLD="$1"
CHECKPOINT="$2"
case "$FOLD" in [1-5]) ;; *) echo "Fold must be 1..5" >&2; exit 2 ;; esac
if [[ ! -f "$CHECKPOINT" ]]; then
    echo "Checkpoint not found: $CHECKPOINT" >&2
    exit 2
fi
PYTHON_BIN="${PYTHON:-python}"
"$PYTHON_BIN" -u test.py with isic_setting1 \
    mode=test eval_fold="$FOLD" "reload_model_path=$CHECKPOINT" \
    num_workers="${NUM_WORKERS:-4}" \
    "path.isic.data_dir=${ISIC_DATA_ROOT:-$PROJECT_DIR/data/ISIC2018}" \
    "isic_setting_1_base_path=${ISIC_BASE_PATH:-$PROJECT_DIR/data/isic/ISIC_setting_1}" \
    "path.log_dir=$PROJECT_DIR/results/ISIC2018"
