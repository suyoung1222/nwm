#!/bin/bash
#SBATCH --mail-user=suyoungkang@umass.edu
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --job-name=ocnwm_habitat_big
#SBATCH -o /scratch/workspace/suyoungkang_umass_edu-ocnwm/ocnwm/nwm/logs/slurm/slurm-ocnwm_habitat_big-%j.out  # %j = job ID
#SBATCH --cpus-per-gpu 8  # Number of Cores per Task
#SBATCH -p gpu-preempt   # preemption-safe: --requeue + --auto-resume (latest.pth.tar)
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --mem-per-cpu=8GB
#SBATCH -t 24:00:00  # Job time limit
#SBATCH --gres=gpu:4
#SBATCH --gpus=4
#SBATCH --constraint=a100-80g|h100   # 80GB-class x86 only: OCNWM-XL needs ~27GB static/GPU (exo adds +335M params), so bs4 OOMs on 40/48GB; gh200 is ARM (env won't run)

set -euo pipefail

# Stage 3 of the big pipeline (runs with --dependency=afterok:<cache job>):
# OCNWM CDiT-XL/2 finetune (from cdit_xl_ego4d_200000) + slot_mode=full on
# habitat_replica_random_big, 4 GPUs.

# --- repo root ---
REPO_DIR="/scratch/workspace/suyoungkang_umass_edu-ocnwm/ocnwm/nwm"
cd "${REPO_DIR}"      # REQUIRED: datasets.py reads config/data_config.yaml relative to CWD
mkdir -p logs/slurm

# --- environment: conda env `nwm` ---
module load conda/latest 2>/dev/null || true
source activate /work/pi_hzhang2_umass_edu/suyoungkang_umass_edu/.conda/envs/nwm

# --- logging: tensorboard always; wandb offline-safe default (sync later with
#     `wandb sync logs/ocnwm_habitat_big/tb/wandb/offline-run-*`; export WANDB_API_KEY
#     + WANDB_MODE=online before sbatch to go live) ---
export WANDB_PROJECT=ocnwm
export WANDB_MODE=${WANDB_MODE:-offline}
export PYTHONFAULTHANDLER=1
export NCCL_DEBUG=WARN
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True   # reduce fragmentation (1.1GB reserved-unallocated seen in OOM)

# --- torchrun rendezvous (single node) ---
export NUM_NODES=1
export NPROC_PER_NODE="${SLURM_GPUS_ON_NODE:-$(nvidia-smi -L | wc -l)}"      # GPUs on this node
export MASTER_ADDR="127.0.0.1"
export MASTER_PORT=$(( 20000 + (SLURM_JOB_ID % 20000) ))                     # avoid port clashes
echo "Launching torchrun with nproc-per-node=${NPROC_PER_NODE}"

# --- launch OCNWM finetune ---
# bfloat16 on, torch-compile OFF (stability). from_checkpoint / finetune_mode / slot_mode
# come from the config; the exo branch + slot_model load non-strict and open from zero-init.
torchrun \
  --nnodes="${NUM_NODES}" \
  --nproc-per-node="${NPROC_PER_NODE}" \
  --rdzv-backend=c10d \
  --rdzv-endpoint="${MASTER_ADDR}:${MASTER_PORT}" \
  train.py \
    --config config/ocnwm_habitat_big.yaml \
    --bfloat16 1 \
    --torch-compile 0 \
    --epochs 100 \
    --log-every 50 \
    --ckpt-every 1000 \
    --eval-every 5000 \
    --auto-resume

echo "Done. Latest checkpoint: logs/ocnwm_habitat_big/checkpoints/latest.pth.tar"
