#!/bin/bash
#SBATCH --job-name=ocnwm_cache_slots
#SBATCH --account=pi_hzhang2_umass_edu
#SBATCH --partition=gpu-preempt          # caching is resumable, so preemptible is fine
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1              # single-process, single-GPU inference
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --output=logs/slurm/%j_cache_out.txt
#SBATCH --error=logs/slurm/%j_cache_err.txt
#
# Caching-ONLY job: precompute frozen-VideoSAUR slots for the OCNWM datasets.
# Frozen inference (no training). Resumable — re-submitting skips trajectories whose
# .npz already exists, so preemption loses at most the trajectory in flight.
#
#   sbatch submit_unity_cache_slots.slurm
#
# Edit PROJECT_DIR / CONDA_ENV / paths below to match your Unity checkout.

# ── config ────────────────────────────────────────────────────────────────────────
PROJECT_DIR=/project/pi_hzhang2_umass_edu/suyoungkang_umass_edu/Git/ocnwm/nwm
CONDA_ENV=nwm                # env with the deps from requirements_slot.txt (see note)
SETUP_DEPS=0                 # set to 1 to pip-install requirements_slot.txt once

DATA_ROOT=${PROJECT_DIR}/../../data/preprocessed          # adjust if your data lives elsewhere
CACHE_ROOT=${DATA_ROOT}/slot_cache/ytvis_dino224_k7
CKPT=${PROJECT_DIR}/thirdparty/videosaur/checkpoints/videosaur-ytvis.ckpt
VS_CONFIG=${PROJECT_DIR}/thirdparty/videosaur/configs/videosaur/ytvis.yml
N_SLOTS=7

# datasets to cache: "name|data_folder" (add more lines as needed)
DATASETS=(
  "bunker2026|${DATA_ROOT}/datasets/bunker2026"
  # "recon|${DATA_ROOT}/datasets/recon"
)

# ── environment ───────────────────────────────────────────────────────────────────
module purge
module load conda/latest 2>/dev/null || true
conda activate ${CONDA_ENV}

cd ${PROJECT_DIR}
mkdir -p logs/slurm

if [ "${SETUP_DEPS}" = "1" ]; then
    echo "Installing slot-extraction deps into ${CONDA_ENV} ..."
    pip install -r requirements_slot.txt
fi

export PYTHONFAULTHANDLER=1
echo "job=$SLURM_JOB_ID  host=$(hostname)  cache_root=${CACHE_ROOT}"
nvidia-smi -L || true

# ── caching ───────────────────────────────────────────────────────────────────────
for entry in "${DATASETS[@]}"; do
    NAME="${entry%%|*}"
    FOLDER="${entry##*|}"
    echo "=== caching slots: ${NAME}  (${FOLDER}) ==="
    python cache_slots.py \
        --data-folder "${FOLDER}" \
        --dataset-name "${NAME}" \
        --cache-root "${CACHE_ROOT}" \
        --checkpoint "${CKPT}" \
        --videosaur-config "${VS_CONFIG}" \
        --n-slots ${N_SLOTS} \
        --save-masks \
        --viz 6 \
        --device cuda
done

echo "caching complete -> ${CACHE_ROOT}"
