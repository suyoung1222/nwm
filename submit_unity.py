"""
Submit fine-tuning job to Unity (UMass HPC) cluster.

Usage:
    python submit_unity.py                         # default: 1 node, 4 GPUs, 24h
    python submit_unity.py --nodes 2 --ngpus 4     # 2 nodes x 4 GPUs = 8 GPUs total
    python submit_unity.py --partition gpu          # non-preemptible queue
    python submit_unity.py --resume                 # resume from logs/ft_bunker2025/checkpoints/latest.pth.tar
    python submit_unity.py --dry-run               # print the sbatch script without submitting

The script writes a temporary .slurm file, then calls `sbatch`.
"""

import argparse
import os
import subprocess
import tempfile
from pathlib import Path

PROJECT_DIR = "/project/pi_hzhang2_umass_edu/suyoungkang_umass_edu/Git/nwm"
CONDA_ENV = "nwm"  # change to your conda env name


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--nodes", type=int, default=1)
    p.add_argument("--ngpus", type=int, default=4, help="GPUs per node")
    p.add_argument("--partition", default="gpu-preempt", help="Slurm partition")
    p.add_argument("--time", default="24:00:00", help="Wall-clock limit (HH:MM:SS)")
    p.add_argument("--config", default="config/ft_bunker2025.yaml")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--ckpt-every", type=int, default=500)
    p.add_argument("--eval-every", type=int, default=2000)
    p.add_argument("--resume", action="store_true",
                   help="Resume: do NOT set finetune_mode (loads optimizer too)")
    p.add_argument("--dry-run", action="store_true",
                   help="Print the sbatch script without submitting")
    return p.parse_args()


def make_slurm_script(args) -> str:
    total_gpus = args.nodes * args.ngpus
    mem_gb = 30 * args.ngpus  # ~30 GB per GPU

    # When resuming a previous fine-tuning run, the config already has finetune_mode=True
    # but train.py will prefer latest.pth.tar (which has optimizer state) — that's correct.
    # Nothing extra needed here.

    return f"""\
#!/bin/bash
#SBATCH --job-name=nwm_ft_bunker2025
#SBATCH --account=pi_hzhang2_umass_edu
#SBATCH --partition={args.partition}
#SBATCH --nodes={args.nodes}
#SBATCH --ntasks-per-node={args.ngpus}
#SBATCH --gres=gpu:{args.ngpus}
#SBATCH --cpus-per-task=8
#SBATCH --mem={mem_gb}G
#SBATCH --time={args.time}
#SBATCH --output={PROJECT_DIR}/logs/slurm/%j_out.txt
#SBATCH --error={PROJECT_DIR}/logs/slurm/%j_err.txt
#SBATCH --signal=SIGUSR1@120

module purge
module load conda/latest 2>/dev/null || true
conda activate {CONDA_ENV}

cd {PROJECT_DIR}

export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=$(shuf -i 20000-30000 -n 1)
export NCCL_DEBUG=WARN
export PYTHONFAULTHANDLER=1

echo "job=$SLURM_JOB_ID  nodes={args.nodes}  gpus/node={args.ngpus}  total={total_gpus}"
echo "MASTER=$MASTER_ADDR:$MASTER_PORT"

mkdir -p {PROJECT_DIR}/logs/slurm

srun torchrun \\
    --nnodes={args.nodes} \\
    --nproc_per_node={args.ngpus} \\
    --rdzv_backend=c10d \\
    --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \\
    train.py \\
    --config {args.config} \\
    --epochs {args.epochs} \\
    --log-every {args.log_every} \\
    --ckpt-every {args.ckpt_every} \\
    --eval-every {args.eval_every} \\
    --bfloat16 1 \\
    --torch-compile 1
"""


def main():
    args = parse_args()
    script = make_slurm_script(args)

    if args.dry_run:
        print(script)
        return

    # Create slurm log dir locally (may not exist yet)
    os.makedirs("logs/slurm", exist_ok=True)

    with tempfile.NamedTemporaryFile("w", suffix=".slurm", delete=False) as f:
        f.write(script)
        tmp_path = f.name

    try:
        result = subprocess.run(["sbatch", tmp_path], capture_output=True, text=True)
        print(result.stdout.strip())
        if result.returncode != 0:
            print("sbatch error:", result.stderr.strip())
    finally:
        os.unlink(tmp_path)


if __name__ == "__main__":
    main()
