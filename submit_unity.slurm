#!/bin/bash
#SBATCH --job-name=nwm_ft_bunker2025
#SBATCH --account=pi_hzhang2_umass_edu
#SBATCH --partition=gpu-preempt         # change to "gpu" for non-preemptible
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4             # one task per GPU
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=24:00:00
#SBATCH --output=logs/slurm/%j_out.txt
#SBATCH --error=logs/slurm/%j_err.txt
#SBATCH --signal=SIGUSR1@120            # give 2-min warning before timeout

# ── environment ──────────────────────────────────────────────────────────────
PROJECT_DIR=/project/pi_hzhang2_umass_edu/suyoungkang_umass_edu/Git/nwm
CONDA_ENV=nwm   # change to your actual conda environment name

module purge
module load conda/latest 2>/dev/null || true
conda activate $CONDA_ENV

cd $PROJECT_DIR

# ── distributed setup ─────────────────────────────────────────────────────────
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=$(shuf -i 20000-30000 -n 1)
export NCCL_DEBUG=WARN
export PYTHONFAULTHANDLER=1

echo "MASTER_ADDR=$MASTER_ADDR  MASTER_PORT=$MASTER_PORT"
echo "Nodes=$SLURM_NNODES  GPUs/node=$SLURM_NTASKS_PER_NODE"

# ── create slurm log dir if missing ───────────────────────────────────────────
mkdir -p logs/slurm

# ── training ──────────────────────────────────────────────────────────────────
# torchrun sets RANK/LOCAL_RANK/WORLD_SIZE, picked up by distributed.py
srun torchrun \
    --nnodes=$SLURM_NNODES \
    --nproc_per_node=$SLURM_NTASKS_PER_NODE \
    --rdzv_backend=c10d \
    --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
    train.py \
    --config config/ft_bunker2025.yaml \
    --epochs 100 \
    --log-every 50 \
    --ckpt-every 500 \
    --eval-every 2000 \
    --bfloat16 1 \
    --torch-compile 1
