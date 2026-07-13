#!/bin/bash
#SBATCH --job-name=cache_slots_pilot
#SBATCH --account=pi_hzhang2_umass_edu
#SBATCH --partition=gpu-preempt
#SBATCH --requeue
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/workspace/suyoungkang_umass_edu-ocnwm/ocnwm/nwm/logs/slurm/%j_cache_pilot_out.txt
#SBATCH --error=/scratch/workspace/suyoungkang_umass_edu-ocnwm/ocnwm/nwm/logs/slurm/%j_cache_pilot_err.txt
#SBATCH --open-mode=append

# Slot caching for the habitat_replica_random PILOT dataset (30 trajs).
# Uses the verified apptainer + .venv_slot pattern (the nwm conda env has a timm 1.x
# conflict with VideoSAUR's timm 0.9.16). Resumable: cache_slots.py skips existing npz,
# so gpu-preempt + --requeue is safe.

set -euo pipefail

WS=/scratch/workspace/suyoungkang_umass_edu-ocnwm
NWM=${WS}/ocnwm/nwm
DATASET=habitat_replica_random

mkdir -p ${NWM}/logs/slurm

apptainer exec --nv -B ${WS} ${WS}/pytorch.sif \
  ${WS}/.venv_slot/bin/python ${NWM}/cache_slots.py \
    --data-folder ${WS}/data/preprocessed/datasets/${DATASET} \
    --dataset-name ${DATASET} \
    --cache-root ${WS}/data/preprocessed/slot_cache/ytvis_dino224_k7 \
    --checkpoint ${NWM}/thirdparty/videosaur/checkpoints/videosaur-ytvis.ckpt \
    --videosaur-config ${NWM}/thirdparty/videosaur/configs/videosaur/ytvis.yml \
    --n-slots 7 --save-masks --viz 4 --device cuda

# Completion gate: all 30 trajs cached, else fail the job.
N=$(ls ${WS}/data/preprocessed/slot_cache/ytvis_dino224_k7/${DATASET}/*.npz 2>/dev/null | wc -l)
echo "cached npz count: ${N}"
[ "${N}" -eq 30 ] || { echo "ERROR: expected 30 npz, got ${N}"; exit 1; }
echo "pilot slot cache complete."
