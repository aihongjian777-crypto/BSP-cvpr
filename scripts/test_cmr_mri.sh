#!/bin/bash
GPUID1=1
export CUDA_VISIBLE_DEVICES=$GPUID1

DATASET='CMR'
NWORKER=8
RUNS=1
ALL_EV=(0 1 2 3 4)
TEST_LABEL=[1,2,3]
NSTEP=50000
DECAY=0.98

MAX_ITER=5000
SNAPSHOT_INTERVAL=50000
SEED=2025

N_PART=3
ALL_SUPP=(2)
echo ========================================================================
for EVAL_FOLD in "${ALL_EV[@]}"
do
  PREFIX="test_${DATASET}_cv${EVAL_FOLD}"
  echo $PREFIX
  LOGDIR="./results"

  if [ ! -d $LOGDIR ]
  then
    mkdir -p $LOGDIR
  fi
  for SUPP_IDX in "${ALL_SUPP[@]}"
  do
    RELOAD_MODEL_PATH="/media/BSP/exps_on_CMR/BSP_train_CMR_cv${EVAL_FOLD}/1/snapshots/50000.pth"
    python3 test.py with \
    mode="test" \
    dataset=$DATASET \
    num_workers=$NWORKER \
    n_steps=$NSTEP \
    eval_fold=$EVAL_FOLD \
    max_iters_per_load=$MAX_ITER \
    supp_idx=$SUPP_IDX \
    test_label=$TEST_LABEL \
    seed=$SEED \
    n_part=$N_PART \
    reload_model_path=$RELOAD_MODEL_PATH \
    save_snapshot_every=$SNAPSHOT_INTERVAL \
    lr_step_gamma=$DECAY \
    path.log_dir=$LOGDIR
    done
done
