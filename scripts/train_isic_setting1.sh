#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"
PYTHON_BIN="${PYTHON:-python}"
ISIC_DATA_ROOT="${ISIC_DATA_ROOT:-$PROJECT_DIR/data/ISIC2018}"
ISIC_BASE_PATH="${ISIC_BASE_PATH:-$PROJECT_DIR/data/isic/ISIC_setting_1}"
WEIGHTS="${BSP_PRETRAINED_WEIGHTS:-$PROJECT_DIR/checkpoints/deeplabv3_resnet101_coco-586e9e4e.pth}"
export BSP_PRETRAINED_WEIGHTS="$WEIGHTS"

if (($#)); then
    FOLDS=("$@")
else
    FOLDS=(1)
fi
for FOLD in "${FOLDS[@]}"; do
    case "$FOLD" in [1-5]) ;; *) echo "Fold must be 1..5: $FOLD" >&2; exit 2 ;; esac
done

"$PYTHON_BIN" data/isic/prepare_setting1_dataset.py \
    --data-root "$ISIC_DATA_ROOT" --base-path "$ISIC_BASE_PATH" \
    --check-only --require-superpixels
if [[ ! -f "$WEIGHTS" ]]; then
    echo "Missing pretrained encoder weights: $WEIGHTS" >&2
    echo "Place the DeepLabV3-ResNet101 checkpoint there, or set BSP_PRETRAINED_WEIGHTS." >&2
    exit 2
fi

for FOLD in "${FOLDS[@]}"; do
    "$PYTHON_BIN" -u train.py with isic_setting1 \
        mode=train eval_fold="$FOLD" \
        num_workers="${NUM_WORKERS:-4}" \
        n_steps="${N_STEPS:-50000}" \
        "path.isic.data_dir=$ISIC_DATA_ROOT" \
        "isic_setting_1_base_path=$ISIC_BASE_PATH"
done
