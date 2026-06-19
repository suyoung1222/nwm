"""
Format-parity check for the UMass bunker2025 dataset against an existing NWM
dataset (recon), BEFORE any fine-tuning.

It does NOT train and does NOT touch the VAE. It only builds the existing
`TrainingDataset` on both datasets, pulls one batch each, and reports:
  - tensor shapes (frames / actions / rel_time)
  - per-channel action (goal_pos) stats: min / max / mean / std + saturation
  - rel_time stats
  - side-by-side visualizations (context + goal frames, action scatter)

Usage:
  python validate_bunker2025.py
  python validate_bunker2025.py --ref_trajs 30 --batch_size 4

The two batches should have identical shapes/format and comparable action
distributions. Eyeball the saved PNGs to confirm the bunker2025 frames look
like normal driving frames and the action arrows point where the robot goes.
"""

import argparse
import os
import shutil
import tempfile

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from datasets import TrainingDataset
from misc import transform, unnormalize

# Params kept identical to training (config/nwm_cdit_xl.yaml + eval_config.yaml)
IMAGE_SIZE = 224
CONTEXT_SIZE = 4
LEN_TRAJ_PRED = 64
MIN_DIST_CAT = -64
MAX_DIST_CAT = 64
GOALS_PER_OBS = 4

# bunker2025 (UMass) — data already in NWM format
BUNKER_DATA = "/home/suyoung/mydata/NWM/preprocessed/datasets/bunker2025"
BUNKER_SPLIT = "/home/suyoung/mydata/NWM/preprocessed/data_splits/bunker2025/train"

# recon — reference existing dataset, available locally
RECON_DATA = "/home/suyoung/mydata/NWM/preprocessed/datasets/recon"
RECON_SPLIT = "data_splits/recon/train"


def build_dataset(name, data_folder, split_folder):
    return TrainingDataset(
        data_folder=data_folder,
        data_split_folder=split_folder,
        dataset_name=name,
        image_size=IMAGE_SIZE,
        min_dist_cat=MIN_DIST_CAT,
        max_dist_cat=MAX_DIST_CAT,
        len_traj_pred=LEN_TRAJ_PRED,
        context_size=CONTEXT_SIZE,
        normalize=True,
        goals_per_obs=GOALS_PER_OBS,
        transform=transform,
        predefined_index=None,
        traj_stride=1,
    )


def make_ref_subset(src_split, n_trajs):
    """Copy the first n_trajs traj names into a temp split so index build is fast."""
    names = [l.strip() for l in open(os.path.join(src_split, "traj_names.txt")) if l.strip()]
    names = names[:n_trajs]
    tmp = tempfile.mkdtemp(prefix="recon_subset_")
    with open(os.path.join(tmp, "traj_names.txt"), "w") as f:
        f.write("\n".join(names) + "\n")
    return tmp, len(names)


def report_batch(tag, obs, act, rel_t):
    print(f"\n========== {tag} ==========")
    print(f"  obs_image : shape={tuple(obs.shape)} dtype={obs.dtype} "
          f"min={obs.min():.3f} max={obs.max():.3f} mean={obs.mean():.3f}")
    print(f"            (expected per-sample = (ctx+goals, 3, {IMAGE_SIZE}, {IMAGE_SIZE}) "
          f"= ({CONTEXT_SIZE + GOALS_PER_OBS}, 3, {IMAGE_SIZE}, {IMAGE_SIZE}))")
    print(f"  goal_pos  : shape={tuple(act.shape)} dtype={act.dtype}")
    a = act.reshape(-1, act.shape[-1]).numpy()
    chan = ["dx (norm)", "dy (norm)", "dyaw (rad)"]
    for c in range(a.shape[1]):
        col = a[:, c]
        extra = ""
        if c < 2:
            sat = np.mean(np.abs(col) > 1.0) * 100
            extra = f" | |.|>1 (saturated): {sat:.1f}%"
        print(f"    {chan[c]:11s}: min={col.min():7.3f} max={col.max():7.3f} "
              f"mean={col.mean():7.3f} std={col.std():6.3f}{extra}")
    rt = rel_t.reshape(-1).numpy()
    print(f"  rel_time  : shape={tuple(rel_t.shape)} min={rt.min():.4f} "
          f"max={rt.max():.4f} mean={rt.mean():.4f}  (= goal_offset / 128)")


def visualize(tag, obs, act, out_path):
    """Sample 0: show context + goal frames on top row, action scatter below."""
    s_obs = obs[0]              # (ctx+goals, 3, H, W)
    s_act = act[0].numpy()      # (goals, 3)
    n = s_obs.shape[0]
    fig, axes = plt.subplots(2, n, figsize=(2.2 * n, 5))
    for i in range(n):
        img = unnormalize(s_obs[i]).permute(1, 2, 0).clamp(0, 1).numpy()
        axes[0, i].imshow(img)
        kind = "ctx" if i < CONTEXT_SIZE else f"goal{i - CONTEXT_SIZE}"
        axes[0, i].set_title(kind, fontsize=9)
        axes[0, i].axis("off")
    # action scatter (normalized dx,dy): NWM plots x forward, -y to the left
    gs = axes[1, 0].get_gridspec()
    for ax in axes[1, :]:
        ax.remove()
    axbig = fig.add_subplot(gs[1, :])
    axbig.plot(-s_act[:, 1], s_act[:, 0], "o-", color="green")
    for j in range(s_act.shape[0]):
        axbig.annotate(f"g{j}", (-s_act[j, 1], s_act[j, 0]), fontsize=8)
    axbig.set_title(f"{tag}: goal_pos (normalized dx/dy)", fontsize=10)
    axbig.set_xlabel("-dy (left +)")
    axbig.set_ylabel("dx (forward +)")
    axbig.set_aspect("equal", adjustable="datalim")
    axbig.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=130)
    plt.close()
    print(f"  saved viz -> {out_path}")


def main(args):
    out_dir = "logs/validate_bunker2025"
    os.makedirs(out_dir, exist_ok=True)
    torch.manual_seed(0)
    np.random.seed(0)

    print("Building bunker2025 (UMass) dataset ...")
    bunker = build_dataset("bunker2025", BUNKER_DATA, BUNKER_SPLIT)
    print(f"  bunker2025 samples: {len(bunker)}")

    print(f"Building recon reference subset ({args.ref_trajs} trajs) ...")
    ref_split, n = make_ref_subset(RECON_SPLIT, args.ref_trajs)
    try:
        recon = build_dataset("recon", RECON_DATA, ref_split)
        print(f"  recon subset samples: {len(recon)} (from {n} trajs)")

        def first_batch(ds):
            loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True,
                                num_workers=0, drop_last=True)
            return next(iter(loader))

        b_obs, b_act, b_rt = first_batch(bunker)
        r_obs, r_act, r_rt = first_batch(recon)
    finally:
        shutil.rmtree(ref_split, ignore_errors=True)

    report_batch("bunker2025 (UMass)", b_obs, b_act, b_rt)
    report_batch("recon (reference)", r_obs, r_act, r_rt)

    # shape parity assertion
    assert b_obs.shape[1:] == r_obs.shape[1:], "frame shape mismatch!"
    assert b_act.shape[1:] == r_act.shape[1:], "action shape mismatch!"
    print("\n[OK] frame & action tensor shapes match between bunker2025 and recon.")

    visualize("bunker2025", b_obs, b_act, os.path.join(out_dir, "bunker2025_sample.png"))
    visualize("recon", r_obs, r_act, os.path.join(out_dir, "recon_sample.png"))
    print(f"\nDone. Compare the two PNGs in {out_dir}/")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--ref_trajs", type=int, default=30,
                   help="how many recon trajectories to load for the reference batch")
    p.add_argument("--batch_size", type=int, default=4)
    args = p.parse_args()
    main(args)
