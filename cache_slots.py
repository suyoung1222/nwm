# OCNWM Stage-1 support: precompute & cache frozen-VideoSAUR slots for dataset frames.
#
# Additive tool. Iterates every trajectory of a dataset, runs the frozen SlotExtractor
# over its full frame sequence (temporally-consistent video slots), and writes one
# compressed .npz per trajectory:  <cache_root>/<dataset>/<traj_name>.npz
#   slots : float16  (L, K, slot_dim)
#   masks : float16  (L, K, num_patches)   [only if --save-masks]
# where L = number of frames = len(traj_data['position']). Frame t of the trajectory
# maps to row t of `slots` (same indexing the dataloader uses: <traj>/<t>.jpg).
#
# Resumable: trajectories whose .npz already exists are skipped unless --overwrite.
# Both on-the-fly extraction (via slot_extractor.SlotExtractor in the train loop) and
# this offline cache are supported; caching is strongly preferred for training speed.
#
# Example (run inside the env that has the videosaur deps, see requirements_slot.txt):
#   python cache_slots.py \
#     --data-folder data/preprocessed/datasets/bunker2026 \
#     --dataset-name bunker2026 \
#     --cache-root data/preprocessed/slot_cache/ytvis_dino224_k7 \
#     --save-masks --viz 6

import argparse
import glob
import json
import os
import pickle
import time
from typing import List, Optional

import numpy as np
import torch
from PIL import Image

from slot_extractor import (
    SlotExtractor,
    DEFAULT_CONFIG,
    DEFAULT_CHECKPOINT,
    build_slot_transform,
)

# Distinct RGB palette for up to ~12 slots (viz only).
_PALETTE = np.array(
    [
        [230, 25, 75], [60, 180, 75], [255, 225, 25], [0, 130, 200],
        [245, 130, 48], [145, 30, 180], [70, 240, 240], [240, 50, 230],
        [210, 245, 60], [250, 190, 212], [0, 128, 128], [170, 110, 40],
    ],
    dtype=np.uint8,
)


def discover_trajectories(data_folder: str, traj_names_file: Optional[str]) -> List[str]:
    """Return the list of trajectory dir names (each containing traj_data.pkl)."""
    if traj_names_file:
        with open(traj_names_file) as f:
            names = [ln.strip() for ln in f if ln.strip()]
        return names
    names = []
    for p in sorted(glob.glob(os.path.join(data_folder, "*"))):
        if os.path.isfile(os.path.join(p, "traj_data.pkl")):
            names.append(os.path.basename(p))
    return names


def trajectory_length(data_folder: str, traj: str) -> int:
    with open(os.path.join(data_folder, traj, "traj_data.pkl"), "rb") as f:
        traj_data = pickle.load(f)
    return int(len(traj_data["position"]))


def frame_paths(data_folder: str, traj: str, length: int) -> List[str]:
    return [os.path.join(data_folder, traj, f"{t}.jpg") for t in range(length)]


def save_attention_viz(
    extractor: SlotExtractor,
    data_folder: str,
    traj: str,
    length: int,
    masks: torch.Tensor,   # (L, K, P)
    out_dir: str,
    n: int,
):
    """Overlay per-pixel argmax-slot segmentation on evenly-spaced frames. Pure PIL."""
    os.makedirs(out_dir, exist_ok=True)
    P = masks.shape[-1]
    g = int(round(P ** 0.5))
    assert g * g == P, f"non-square patch grid P={P}"
    # display transform: same crop/resize as the model input, but WITHOUT normalization
    disp_tf = build_slot_transform(extractor.input_size, extractor.crop_mode, "movi")
    idxs = np.linspace(0, length - 1, num=min(n, length)).round().astype(int)
    for t in idxs:
        img = Image.open(os.path.join(data_folder, traj, f"{t}.jpg")).convert("RGB")
        # undo movi normalization (x*0.5+0.5) -> [0,1] display image
        disp = (disp_tf(img) * 0.5 + 0.5).clamp(0, 1)
        disp = (disp.permute(1, 2, 0).numpy() * 255).astype(np.uint8)  # (H,W,3)
        H, W = disp.shape[:2]
        m = masks[t]  # (K, P)
        seg = m.argmax(dim=0).reshape(g, g).cpu().numpy()  # (g,g) slot id per patch
        seg_img = Image.fromarray(_PALETTE[seg % len(_PALETTE)]).resize((W, H), Image.NEAREST)
        seg_arr = np.asarray(seg_img)
        blend = (0.55 * disp + 0.45 * seg_arr).astype(np.uint8)
        Image.fromarray(blend).save(os.path.join(out_dir, f"{traj}_f{t:04d}_seg.png"))
    print(f"  [viz] wrote {len(idxs)} attention overlays to {out_dir}")


def main():
    ap = argparse.ArgumentParser(description="Cache frozen VideoSAUR slots for a dataset.")
    ap.add_argument("--data-folder", required=True, help="dir of trajectory subfolders")
    ap.add_argument("--dataset-name", required=True, help="cache subdir name")
    ap.add_argument("--cache-root", required=True, help="root output dir for the cache")
    ap.add_argument("--traj-names-file", default=None,
                    help="optional split traj_names.txt; default = all trajectories in data-folder")
    ap.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    ap.add_argument("--videosaur-config", default=DEFAULT_CONFIG)
    ap.add_argument("--n-slots", type=int, default=None, help="override K (RandomInit only)")
    ap.add_argument("--input-size", type=int, default=224)
    ap.add_argument("--crop-mode", default="nwm_ar", choices=["nwm_ar", "none"])
    ap.add_argument("--normalization", default="imagenet", choices=["imagenet", "movi"])
    ap.add_argument("--encoder-chunk", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save-masks", action="store_true",
                    help="also cache slot-attention masks (needed for viz/diagnostics)")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--viz", type=int, default=0,
                    help="save N attention overlays for the first processed trajectory")
    ap.add_argument("--limit", type=int, default=0, help="cache at most N trajectories (0=all)")
    args = ap.parse_args()

    out_dir = os.path.join(args.cache_root, args.dataset_name)
    os.makedirs(out_dir, exist_ok=True)

    trajs = discover_trajectories(args.data_folder, args.traj_names_file)
    if args.limit:
        trajs = trajs[: args.limit]
    print(f"[cache_slots] {args.dataset_name}: {len(trajs)} trajectories -> {out_dir}")

    extractor = SlotExtractor(
        checkpoint=args.checkpoint,
        videosaur_config=args.videosaur_config,
        device=args.device,
        n_slots=args.n_slots,
        input_size=args.input_size,
        crop_mode=args.crop_mode,
        normalization=args.normalization,
        encoder_chunk=args.encoder_chunk,
        seed=args.seed,
    )
    print(f"[cache_slots] K={extractor.num_slots} slot_dim={extractor.slot_dim} "
          f"crop={args.crop_mode} norm={args.normalization}")

    stats = {"n_frames": 0, "slot_mean": [], "slot_std": [], "consec_cos": []}
    did_viz = False
    n_done = n_skip = 0
    t0 = time.time()
    for i, traj in enumerate(trajs):
        out_path = os.path.join(out_dir, f"{traj}.npz")
        if os.path.exists(out_path) and not args.overwrite:
            n_skip += 1
            continue
        try:
            length = trajectory_length(args.data_folder, traj)
            paths = frame_paths(args.data_folder, traj, length)
            missing = [p for p in paths if not os.path.exists(p)]
            if missing:
                print(f"  [warn] {traj}: {len(missing)} missing frames (e.g. {missing[0]}); skipping")
                continue
            frames = extractor.preprocess_paths(paths)
            res = extractor.extract(frames, return_masks=args.save_masks)
            slots = res["slots"].numpy().astype(np.float16)  # (L,K,D)
            save_kw = {"slots": slots}
            if args.save_masks:
                save_kw["masks"] = res["masks"].numpy().astype(np.float16)  # (L,K,P)
            # atomic-ish write
            tmp = out_path + ".tmp.npz"
            np.savez_compressed(tmp, **save_kw)
            os.replace(tmp, out_path)
            n_done += 1

            # accumulate stats
            s = res["slots"]
            stats["n_frames"] += length
            stats["slot_mean"].append(float(s.mean()))
            stats["slot_std"].append(float(s.std()))
            if length > 1:
                cos = torch.nn.functional.cosine_similarity(s[:-1], s[1:], dim=-1).mean()
                stats["consec_cos"].append(float(cos))
            print(f"  [{i+1}/{len(trajs)}] {traj}: L={length} slots{tuple(slots.shape)} "
                  f"({time.time()-t0:.1f}s elapsed)")

            if args.viz and not did_viz and args.save_masks:
                save_attention_viz(extractor, args.data_folder, traj, length,
                                   res["masks"], os.path.join(out_dir, "_viz"), args.viz)
                did_viz = True
        except Exception as e:
            print(f"  [error] {traj}: {type(e).__name__}: {e}")
            raise

    # meta + stats
    meta = {
        "dataset_name": args.dataset_name,
        "data_folder": os.path.abspath(args.data_folder),
        "checkpoint": os.path.abspath(args.checkpoint),
        "videosaur_config": os.path.abspath(args.videosaur_config),
        "num_slots": extractor.num_slots,
        "slot_dim": extractor.slot_dim,
        "num_patches": extractor.num_patches,
        "input_size": args.input_size,
        "crop_mode": args.crop_mode,
        "normalization": args.normalization,
        "encoder_chunk": args.encoder_chunk,
        "seed": args.seed,
        "save_masks": bool(args.save_masks),
    }
    with open(os.path.join(out_dir, "_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    def _agg(xs):
        return float(np.mean(xs)) if xs else None
    summary = {
        "trajectories_cached": n_done,
        "trajectories_skipped_existing": n_skip,
        "total_frames": stats["n_frames"],
        "slot_mean": _agg(stats["slot_mean"]),
        "slot_std": _agg(stats["slot_std"]),
        "mean_consecutive_frame_slot_cosine": _agg(stats["consec_cos"]),
    }
    with open(os.path.join(out_dir, "_stats.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[cache_slots] done: {json.dumps(summary)}")


if __name__ == "__main__":
    main()
