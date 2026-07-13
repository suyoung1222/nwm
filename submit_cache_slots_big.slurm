#!/bin/bash
#SBATCH --mail-user=suyoungkang@umass.edu
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --job-name=cache_slots_big
#SBATCH -o /scratch/workspace/suyoungkang_umass_edu-ocnwm/ocnwm/nwm/logs/slurm/slurm-cache_slots_big-%j.out  # %j = job ID
#SBATCH --cpus-per-task 8  # Number of Cores per Task
#SBATCH -p gpu-preempt   # caching is npz-level resumable, so preemption is safe
#SBATCH --requeue
#SBATCH --mem=48G
#SBATCH -t 08:00:00  # Job time limit
#SBATCH --gres=gpu:1
#SBATCH --gpus=1
#SBATCH --constraint=a100|h100|a100-80g|l40s|a40   # x86 only: gh200 is ARM, pytorch.sif is x86_64

set -euo pipefail

# Stage 2 of the big pipeline (runs with --dependency=afterok:<gen array job>):
#   Phase 1 (CPU, idempotent): merge the 6 shard split files, carve a scene-stratified
#            test split (last 10 per scene -> 600 train / 60 test), merge the per-chunk
#            manifests and check mean_step vs data_config.yaml.
#   Phase 2 (GPU, resumable):  slot caching via apptainer + .venv_slot.

WS=/scratch/workspace/suyoungkang_umass_edu-ocnwm
NWM=${WS}/ocnwm/nwm
DATASET=habitat_replica_random_big
N_SCENES=6
N_PER_SCENE=110
N_TEST_PER_SCENE=10
N_TOTAL=$((N_SCENES * N_PER_SCENE))

mkdir -p ${NWM}/logs/slurm

# ---------- Phase 1: merge shard splits + carve test (idempotent rewrite) ----------
python3 - <<EOF
import json, os, glob

ws = "${WS}"; dataset = "${DATASET}"
n_test = ${N_TEST_PER_SCENE}
shard_glob = os.path.join(ws, "data/preprocessed/data_splits/_shards/hrrb_shard_*", dataset, "train/traj_names.txt")
shards = sorted(glob.glob(shard_glob))
assert len(shards) == ${N_SCENES}, f"expected ${N_SCENES} shard split files, found {len(shards)}: {shards}"

names = []
for s in shards:
    names += [l.strip() for l in open(s) if l.strip()]
assert len(names) == ${N_TOTAL}, f"expected ${N_TOTAL} trajs, found {len(names)}"
assert len(set(names)) == len(names), "duplicate traj names across shards!"

# scene-stratified test carve: last n_test (by numeric suffix) per scene
by_scene = {}
for n in names:
    scene = n.rsplit("_", 1)[0]
    by_scene.setdefault(scene, []).append(n)
train, test = [], []
for scene, lst in sorted(by_scene.items()):
    lst.sort(key=lambda n: int(n.rsplit("_", 1)[1]))
    train += lst[:-n_test]
    test += lst[-n_test:]

split_dir = os.path.join(ws, "data/preprocessed/data_splits", dataset)
for split, lst in (("train", train), ("test", test)):
    d = os.path.join(split_dir, split)
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "traj_names.txt"), "w") as f:
        f.write("\n".join(lst) + "\n")
print(f"[merge] train={len(train)} test={len(test)} -> {split_dir}")

# manifest merge + check: measured mean_step vs data_config.yaml value
mans = sorted(glob.glob(os.path.join(ws, "data/preprocessed/datasets", dataset, "_manifest_*.json")))
recs = [r for m in mans for r in json.load(open(m))]
merged = os.path.join(ws, "data/preprocessed/datasets", dataset, "_manifest.json")
json.dump(recs, open(merged, "w"), indent=2)
ms = sum(r["mean_step"] for r in recs) / len(recs)
print(f"[manifest] {len(mans)} files, {len(recs)} trajs, {sum(r['frames'] for r in recs)} frames, mean_step={ms:.4f}")
expected = 0.2326
if abs(ms - expected) / expected > 0.05:
    print(f"WARNING: mean_step {ms:.4f} deviates >5% from data_config.yaml "
          f"habitat_replica_random_big={expected} -- update the yaml before training!")
EOF

# ---------- Phase 2: slot caching (resumable) ----------
apptainer exec --nv -B ${WS} ${WS}/pytorch.sif \
  ${WS}/.venv_slot/bin/python ${NWM}/cache_slots.py \
    --data-folder ${WS}/data/preprocessed/datasets/${DATASET} \
    --dataset-name ${DATASET} \
    --cache-root ${WS}/data/preprocessed/slot_cache/ytvis_dino224_k7 \
    --checkpoint ${NWM}/thirdparty/videosaur/checkpoints/videosaur-ytvis.ckpt \
    --videosaur-config ${NWM}/thirdparty/videosaur/configs/videosaur/ytvis.yml \
    --n-slots 7 --save-masks --viz 6 --device cuda

# ---------- completion gate: all trajs cached, else fail so the train job's afterok blocks ----------
N=$(ls ${WS}/data/preprocessed/slot_cache/ytvis_dino224_k7/${DATASET}/*.npz 2>/dev/null | wc -l)
echo "cached npz count: ${N}"
[ "${N}" -eq "${N_TOTAL}" ] || { echo "ERROR: expected ${N_TOTAL} npz, got ${N}"; exit 1; }
echo "Done. Slot cache: ${WS}/data/preprocessed/slot_cache/ytvis_dino224_k7/${DATASET}"
