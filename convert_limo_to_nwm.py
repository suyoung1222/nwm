"""
Convert limo robot data (timestamped PNGs + pose txt) to NWM inference format.

Usage:
  python3 convert_limo_to_nwm.py \
      --image_dir <path/to/images> \
      --pose_file <path/to/pose.txt> \
      --output_data_folder <path/to/nwm_dataset> \
      --dataset_name limo2 \
      --traj_name limo2_bag

Output structure:
  <output_data_folder>/
    <traj_name>/
      0.jpg, 1.jpg, ..., N.jpg
      traj_data.pkl
  data_splits/<dataset_name>/test/
    traj_names.txt
    rollout.pkl
    time.pkl
"""

import os
import sys
import pickle
import argparse
import numpy as np
from PIL import Image

# ── eval params (must match eval_config.yaml) ────────────────────────────────
CONTEXT_SIZE   = 4
LEN_TRAJ_PRED  = 64
TRAJ_STRIDE    = 8
MIN_DIST_CAT   = -64
MAX_DIST_CAT   = 64

# ── helpers ───────────────────────────────────────────────────────────────────

def quat_to_yaw(qx, qy, qz, qw):
    """Quaternion → yaw (rotation around Z axis), in radians."""
    return np.arctan2(2.0 * (qw * qz + qx * qy),
                      1.0 - 2.0 * (qy * qy + qz * qz))


def load_pose(pose_file):
    """Returns sorted list of (timestamp, x, y, yaw_rad). Pose format: ts x y z qx qy qz qw"""
    entries = []
    with open(pose_file, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 8:
                continue
            ts = int(parts[0])
            x, y = float(parts[1]), float(parts[2])
            qx, qy, qz, qw = float(parts[4]), float(parts[5]), float(parts[6]), float(parts[7])
            yaw = quat_to_yaw(qx, qy, qz, qw)
            entries.append((ts, x, y, yaw))
    entries.sort(key=lambda e: e[0])
    return entries


def build_index(traj_name, traj_len, context_size, len_traj_pred, traj_stride, min_dist, max_dist):
    samples = []
    begin_time = context_size - 1
    end_time   = traj_len - len_traj_pred
    for curr_time in range(begin_time, end_time, traj_stride):
        max_goal_dist = min(max_dist, traj_len - curr_time - 1)
        min_goal_dist = max(min_dist, -curr_time)
        samples.append((traj_name, curr_time, min_goal_dist, max_goal_dist))
    return samples


# ── main ──────────────────────────────────────────────────────────────────────

def main(args):
    nwm_root   = os.path.dirname(os.path.abspath(__file__))
    split_dir  = os.path.join(nwm_root, "data_splits", args.dataset_name, "test")
    traj_out_dir = os.path.join(args.output_data_folder, args.traj_name)

    print(f"Loading poses from {args.pose_file} …")
    poses = load_pose(args.pose_file)
    pose_by_ts = {ts: (x, y, yaw) for ts, x, y, yaw in poses}

    # Collect sorted image timestamps
    img_files = sorted(
        [f for f in os.listdir(args.image_dir) if f.endswith(".png")],
        key=lambda f: int(os.path.splitext(f)[0])
    )

    # Match images to poses by timestamp
    matched = []
    for fname in img_files:
        ts = int(os.path.splitext(fname)[0])
        if ts in pose_by_ts:
            matched.append((ts, fname))

    if not matched:
        print("ERROR: No matching timestamps between images and pose file.")
        sys.exit(1)

    print(f"Matched {len(matched)} frames.")

    # Compute average waypoint spacing
    positions = np.array([(pose_by_ts[ts][0], pose_by_ts[ts][1]) for ts, _ in matched])
    if len(positions) > 1:
        dists = np.linalg.norm(np.diff(positions, axis=0), axis=1)
        avg_spacing = float(np.mean(dists[dists > 0]))
        print(f"Average waypoint spacing: {avg_spacing:.4f} m")
    else:
        avg_spacing = 0.25

    # ── create output trajectory folder ──────────────────────────────────────
    os.makedirs(traj_out_dir, exist_ok=True)

    position_list = []
    yaw_list      = []

    print("Copying images and building traj_data …")
    for i, (ts, fname) in enumerate(matched):
        src = os.path.join(args.image_dir, fname)
        dst = os.path.join(traj_out_dir, f"{i}.jpg")
        img = Image.open(src).convert("RGB")
        img.save(dst, "JPEG", quality=95)

        x, y, yaw = pose_by_ts[ts]
        position_list.append([x, y])
        yaw_list.append(yaw)

    traj_data = {
        "position": np.array(position_list, dtype=np.float64),
        "yaw":      np.array(yaw_list,      dtype=np.float64),
    }
    pkl_path = os.path.join(traj_out_dir, "traj_data.pkl")
    with open(pkl_path, "wb") as f:
        pickle.dump(traj_data, f)
    print(f"Saved traj_data.pkl  (position={traj_data['position'].shape}, yaw={traj_data['yaw'].shape})")

    # ── create data_splits ────────────────────────────────────────────────────
    os.makedirs(split_dir, exist_ok=True)

    traj_names_path = os.path.join(split_dir, "traj_names.txt")
    with open(traj_names_path, "w") as f:
        f.write(args.traj_name + "\n")
    print(f"Wrote {traj_names_path}")

    traj_len = len(matched)
    index = build_index(args.traj_name, traj_len, CONTEXT_SIZE, LEN_TRAJ_PRED, TRAJ_STRIDE, MIN_DIST_CAT, MAX_DIST_CAT)
    print(f"Built index with {len(index)} samples (traj_len={traj_len})")

    for split_name in ("rollout", "time"):
        out_pkl = os.path.join(split_dir, f"{split_name}.pkl")
        with open(out_pkl, "wb") as f:
            pickle.dump(index, f)
        print(f"Wrote {out_pkl}")

    # ── print config snippets ─────────────────────────────────────────────────
    print("\n── Add to config/eval_config.yaml ───────────────────────────────")
    print(f"  {args.dataset_name}:")
    print(f"    data_folder: {os.path.abspath(args.output_data_folder)}")
    print(f"    test: data_splits/{args.dataset_name}/test/")
    print(f"    goals_per_obs: 4")

    print("\n── Add to config/data_config.yaml ───────────────────────────────")
    print(f"  {args.dataset_name}:")
    print(f"    metric_waypoint_spacing: {avg_spacing:.4f}")
    print("\nDone.")


if __name__ == "__main__":
    BASE = "/home/suyoung/Documents/limo/agilex_open_class/limo/limo_gazebo_sim/scripts"

    parser = argparse.ArgumentParser()
    parser.add_argument("--image_dir",           type=str, required=True)
    parser.add_argument("--pose_file",           type=str, required=True)
    parser.add_argument("--output_data_folder",  type=str, required=True)
    parser.add_argument("--dataset_name",        type=str, required=True)
    parser.add_argument("--traj_name",           type=str, required=True)
    args = parser.parse_args()
    main(args)
