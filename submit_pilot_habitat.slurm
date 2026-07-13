#!/bin/bash
#SBATCH --job-name=ocnwm_pilot_habitat
#SBATCH --account=pi_hzhang2_umass_edu
#SBATCH --partition=gpu-preempt
#SBATCH --requeue
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --constraint=a100|l40s
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/workspace/suyoungkang_umass_edu-ocnwm/ocnwm/nwm/logs/slurm/%j_pilot_out.txt
#SBATCH --error=/scratch/workspace/suyoungkang_umass_edu-ocnwm/ocnwm/nwm/logs/slurm/%j_pilot_err.txt
#SBATCH --open-mode=append

# OCNWM pilot training on habitat_replica_random (CDiT-S/2, slot_mode=full, 2000 steps).
# Preemption-safe: --requeue + --auto-resume (resumes from checkpoints/latest.pth.tar).

set -euo pipefail

PROJECT_DIR=/scratch/workspace/suyoungkang_umass_edu-ocnwm/ocnwm/nwm
module purge 2>/dev/null || true
module load conda/latest 2>/dev/null || true
source activate /work/pi_hzhang2_umass_edu/suyoungkang_umass_edu/.conda/envs/nwm

cd ${PROJECT_DIR}      # REQUIRED: datasets.py reads config/data_config.yaml relative to CWD
mkdir -p logs/slurm

# wandb: offline-safe default (no API key needed; job never blocks). Sync later with
#   wandb sync logs/ocnwm_pilot_habitat/tb/wandb/offline-run-*
# To go live: export WANDB_API_KEY=... and WANDB_MODE=online before sbatch.
export WANDB_PROJECT=ocnwm
export WANDB_MODE=${WANDB_MODE:-offline}
export PYTHONFAULTHANDLER=1

MASTER_PORT=$(shuf -i 20000-30000 -n 1)
torchrun --nnodes=1 --nproc_per_node=1 \
    --rdzv_backend=c10d --rdzv_endpoint=localhost:${MASTER_PORT} \
    train.py --config config/ocnwm_pilot_habitat.yaml \
    --epochs 100 --log-every 10 --ckpt-every 500 --eval-every 1000 \
    --bfloat16 1 --torch-compile 0 --auto-resume
