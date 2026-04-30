"""
Convert a ROS1 bag (image + odom) into an NWM-compatible dataset.

Single pass over the bag:
  - reads `--odom_topic` to track the latest pose
  - reads `--image_topic` and keeps a frame every 1/fps seconds, pairing it
    with the latest odom pose
  - writes 0.jpg, 1.jpg, ... and traj_data.pkl
  - writes data_splits/<dataset_name>/test/{traj_names.txt, rollout.pkl, time.pkl}

Requires:
    pip install rosbags pillow numpy

Example:
    python bag_to_nwm.py \
        --bag /home/suyoung/mydata/topomaps/bags/run1.bag \
        --image_topic /camera/color/image_raw \
        --odom_topic /odom \
        --output_data_folder /home/suyoung/mydata/nwm \
        --dataset_name mybag --traj_name run1 \
        --fps 4

To list topics first:
    python bag_to_nwm.py --bag <path>.bag --list_topics
"""

import argparse
import io
import os
import pickle
import sys
from pathlib import Path

import numpy as np
from PIL import Image

from rosbags.highlevel import AnyReader

# Match config/eval_config.yaml defaults
CONTEXT_SIZE  = 4
LEN_TRAJ_PRED = 64
TRAJ_STRIDE   = 8
MIN_DIST_CAT  = -64
MAX_DIST_CAT  = 64


def quat_to_yaw(qx, qy, qz, qw):
    return np.arctan2(2.0 * (qw * qz + qx * qy),
                      1.0 - 2.0 * (qy * qy + qz * qz))


def msg_to_pil(msg, msgtype):
    """Decode sensor_msgs/Image or sensor_msgs/CompressedImage to PIL RGB."""
    if 'CompressedImage' in msgtype:
        return Image.open(io.BytesIO(bytes(msg.data))).convert('RGB')

    h, w, enc = msg.height, msg.width, msg.encoding
    raw = bytes(msg.data)

    if enc in ('rgb8', 'bgr8'):
        arr = np.frombuffer(raw, dtype=np.uint8).reshape(h, w, 3)
        if enc == 'bgr8':
            arr = arr[..., ::-1]
        return Image.fromarray(arr, 'RGB')
    if enc in ('rgba8', 'bgra8'):
        arr = np.frombuffer(raw, dtype=np.uint8).reshape(h, w, 4)
        if enc == 'bgra8':
            arr = arr[..., [2, 1, 0, 3]]
        return Image.fromarray(arr, 'RGBA').convert('RGB')
    if enc == 'mono8':
        arr = np.frombuffer(raw, dtype=np.uint8).reshape(h, w)
        return Image.fromarray(arr, 'L').convert('RGB')
    if enc in ('16UC1', 'mono16'):
        arr = np.frombuffer(raw, dtype=np.uint16).reshape(h, w)
        norm = (arr / max(1, arr.max()) * 255).astype(np.uint8)
        return Image.fromarray(norm, 'L').convert('RGB')

    raise ValueError(f"Unsupported image encoding: {enc!r}")


def list_topics(bag_path):
    with AnyReader([bag_path]) as reader:
        rows = [(c.topic, c.msgtype, c.msgcount) for c in reader.connections]
    rows.sort()
    print(f"{'topic':<50} {'msgtype':<40} {'count':>8}")
    for t, mt, n in rows:
        print(f"{t:<50} {mt:<40} {n:>8}")


def main(args):
    bag_path = Path(args.bag)
    if not bag_path.exists():
        sys.exit(f"Bag not found: {bag_path}")

    if args.list_topics:
        list_topics(bag_path)
        return

    if not args.image_topic or not args.odom_topic:
        sys.exit("--image_topic and --odom_topic are required (or use --list_topics).")

    target_dt = 1.0 / args.fps
    out_traj_dir = Path(args.output_data_folder) / args.traj_name
    out_traj_dir.mkdir(parents=True, exist_ok=True)

    positions, yaws = [], []
    latest_pose = None       # (x, y, yaw, pose_t)
    last_kept_t = None
    img_idx = 0
    n_skipped_no_odom = 0
    n_skipped_stale_odom = 0

    with AnyReader([bag_path]) as reader:
        topic_set = {c.topic for c in reader.connections}
        for needed in (args.image_topic, args.odom_topic):
            if needed not in topic_set:
                sys.exit(f"Topic {needed!r} not in bag. Use --list_topics to see available.")

        wanted = [c for c in reader.connections
                  if c.topic in (args.image_topic, args.odom_topic)]

        for connection, timestamp, rawdata in reader.messages(connections=wanted):
            t = timestamp * 1e-9

            if connection.topic == args.odom_topic:
                msg = reader.deserialize(rawdata, connection.msgtype)
                p = msg.pose.pose.position
                q = msg.pose.pose.orientation
                latest_pose = (p.x, p.y, quat_to_yaw(q.x, q.y, q.z, q.w), t)
                continue

            # image
            if latest_pose is None:
                n_skipped_no_odom += 1
                continue
            if last_kept_t is not None and (t - last_kept_t) < target_dt:
                continue
            x, y, yaw, pose_t = latest_pose
            if abs(pose_t - t) > args.max_sync_gap:
                n_skipped_stale_odom += 1
                continue

            msg = reader.deserialize(rawdata, connection.msgtype)
            img = msg_to_pil(msg, connection.msgtype)
            img.save(out_traj_dir / f"{img_idx}.jpg", "JPEG", quality=95)
            positions.append([x, y])
            yaws.append(yaw)
            img_idx += 1
            last_kept_t = t

    if n_skipped_no_odom:
        print(f"  skipped {n_skipped_no_odom} early frames (no odom yet)")
    if n_skipped_stale_odom:
        print(f"  skipped {n_skipped_stale_odom} frames (odom older than --max_sync_gap)")

    if img_idx < CONTEXT_SIZE + 1:
        sys.exit(f"Too few synced frames: {img_idx}. Need >= {CONTEXT_SIZE + 1}.")

    print(f"Saved {img_idx} frames at ~{args.fps} Hz to {out_traj_dir}")

    traj_data = {
        "position": np.asarray(positions, dtype=np.float64),
        "yaw":      np.asarray(yaws,      dtype=np.float64),
    }
    pkl_path = out_traj_dir / "traj_data.pkl"
    with open(pkl_path, "wb") as f:
        pickle.dump(traj_data, f)
    print(f"Wrote {pkl_path}  position={traj_data['position'].shape} yaw={traj_data['yaw'].shape}")

    # data_splits relative to this script (nwm repo root)
    nwm_root  = Path(__file__).resolve().parent
    split_dir = nwm_root / "data_splits" / args.dataset_name / "test"
    split_dir.mkdir(parents=True, exist_ok=True)

    (split_dir / "traj_names.txt").write_text(args.traj_name + "\n")
    print(f"Wrote {split_dir / 'traj_names.txt'}")

    index = []
    begin_time = CONTEXT_SIZE - 1
    end_time   = img_idx - LEN_TRAJ_PRED
    for curr_time in range(begin_time, end_time, TRAJ_STRIDE):
        max_goal_dist = min(MAX_DIST_CAT, img_idx - curr_time - 1)
        min_goal_dist = max(MIN_DIST_CAT, -curr_time)
        index.append((args.traj_name, curr_time, min_goal_dist, max_goal_dist))
    print(f"Built index with {len(index)} samples (traj_len={img_idx})")
    if len(index) == 0:
        print("WARNING: index is empty — trajectory shorter than LEN_TRAJ_PRED. "
              "Capture a longer bag, raise --fps, or shorten LEN_TRAJ_PRED.")

    for split_name in ("rollout", "time"):
        with open(split_dir / f"{split_name}.pkl", "wb") as f:
            pickle.dump(index, f)
        print(f"Wrote {split_dir / f'{split_name}.pkl'}")

    pos = np.asarray(positions)
    if len(pos) > 1:
        d = np.linalg.norm(np.diff(pos, axis=0), axis=1)
        avg_spacing = float(d[d > 0].mean()) if np.any(d > 0) else 0.25
    else:
        avg_spacing = 0.25

    print("\n── Add to config/eval_config.yaml under eval_datasets ──")
    print(f"  {args.dataset_name}:")
    print(f"    data_folder: {Path(args.output_data_folder).resolve()}")
    print(f"    test: data_splits/{args.dataset_name}/test/")
    print(f"    goals_per_obs: 4")

    print("\n── Add to config/data_config.yaml ──")
    print(f"  {args.dataset_name}:")
    print(f"    metric_waypoint_spacing: {avg_spacing:.4f}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--bag", required=True)
    p.add_argument("--image_topic", default=None)
    p.add_argument("--odom_topic",  default=None)
    p.add_argument("--output_data_folder", default=None)
    p.add_argument("--dataset_name", default=None)
    p.add_argument("--traj_name",    default=None)
    p.add_argument("--fps", type=float, default=4.0,
                   help="target sampling rate, NWM is trained at 4 Hz")
    p.add_argument("--max_sync_gap", type=float, default=0.2,
                   help="max seconds between an image and the latest odom")
    p.add_argument("--list_topics", action="store_true",
                   help="just print topics in the bag and exit")
    args = p.parse_args()

    if not args.list_topics:
        for required in ("output_data_folder", "dataset_name", "traj_name"):
            if getattr(args, required) is None:
                sys.exit(f"--{required} is required when not using --list_topics")

    main(args)
