#!/bin/bash
# Local single-GPU debug run for fine-tuning on bunker2025.
# Tests the full pipeline (data loading, model forward, checkpoint save)
# with minimal compute so issues surface quickly.
#
# Usage: bash run_debug.sh
# Requirements: one CUDA GPU, conda env activated

set -e

cd "$(dirname "$0")"

# Short intervals so you see checkpoints/logs within a few steps
LOG_EVERY=5
CKPT_EVERY=20     # save checkpoint every 20 steps
EVAL_EVERY=999999  # skip eval locally (dreamsim is slow; remove this on cluster)
EPOCHS=2

torchrun \
    --standalone \
    --nproc_per_node=1 \
    train.py \
    --config config/ft_bunker2025_debug.yaml \
    --epochs $EPOCHS \
    --log-every $LOG_EVERY \
    --ckpt-every $CKPT_EVERY \
    --eval-every $EVAL_EVERY \
    --bfloat16 1 \
    --torch-compile 0
