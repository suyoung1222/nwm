"""Encode a bunker2025 sequence's frames into a video using ffmpeg.

Usage:
    python frames_to_video.py                       # default: seq1_tothelake @ 4 fps
    python frames_to_video.py --seq seq3_alongthelake
    python frames_to_video.py --seq seq2_tothelake --fps 10 --out my.mp4
"""

import argparse
import re
import shutil
import subprocess
import sys
from pathlib import Path

DATA_ROOT = Path(__file__).resolve().parent / "data" / "bunker2025"


def list_sequences(root: Path) -> list[str]:
    return sorted(p.name for p in root.iterdir() if p.is_dir())


def collect_frames(seq_dir: Path) -> list[Path]:
    frames = list(seq_dir.glob("*.jpg"))
    if not frames:
        raise FileNotFoundError(f"No .jpg frames in {seq_dir}")
    # Sort numerically by the integer in the filename (0.jpg, 1.jpg, ..., 10.jpg).
    def key(p: Path) -> int:
        m = re.match(r"(\d+)", p.stem)
        if not m:
            raise ValueError(f"Unexpected frame filename: {p.name}")
        return int(m.group(1))

    frames.sort(key=key)
    return frames


def write_concat_list(frames: list[Path], fps: float, list_path: Path) -> None:
    # ffmpeg concat demuxer: lines alternating `file '...'` and `duration <seconds>`.
    # The last frame needs both a duration line AND a repeated file line per docs.
    dt = 1.0 / fps
    with list_path.open("w") as f:
        for p in frames:
            f.write(f"file '{p.as_posix()}'\n")
            f.write(f"duration {dt}\n")
        f.write(f"file '{frames[-1].as_posix()}'\n")


def encode(frames: list[Path], fps: float, out_path: Path) -> None:
    if shutil.which("ffmpeg") is None:
        sys.exit("ffmpeg not found on PATH. Run: module load ffmpeg/7.0.2")

    list_path = out_path.with_suffix(".concat.txt")
    write_concat_list(frames, fps, list_path)
    try:
        cmd = [
            "ffmpeg", "-y",
            "-f", "concat", "-safe", "0",
            "-i", str(list_path),
            "-fps_mode", "cfr",
            "-r", str(fps),
            "-pix_fmt", "yuv420p",
            "-c:v", "libx264",
            "-crf", "18",
            "-preset", "medium",
            str(out_path),
        ]
        subprocess.run(cmd, check=True)
    finally:
        list_path.unlink(missing_ok=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-root", type=Path, default=DATA_ROOT)
    ap.add_argument("--seq", default="seq1_tothelake",
                    help="Sequence folder name under data/bunker2025/")
    ap.add_argument("--fps", type=float, default=4.0,
                    help="Output frame rate (recording was ~4 Hz).")
    ap.add_argument("--out", type=Path, default=None,
                    help="Output video path (default: <seq>.mp4 in cwd).")
    ap.add_argument("--list", action="store_true",
                    help="List available sequences and exit.")
    args = ap.parse_args()

    if args.list:
        for name in list_sequences(args.data_root):
            print(name)
        return

    seq_dir = args.data_root / args.seq
    if not seq_dir.is_dir():
        sys.exit(f"Sequence not found: {seq_dir}\n"
                 f"Available: {list_sequences(args.data_root)}")

    frames = collect_frames(seq_dir)
    out = args.out or Path(f"{args.seq}.mp4")
    print(f"Sequence : {seq_dir}")
    print(f"Frames   : {len(frames)} ({frames[0].name} .. {frames[-1].name})")
    print(f"FPS      : {args.fps}")
    print(f"Output   : {out}")
    encode(frames, args.fps, out)
    print(f"Done: {out}")


if __name__ == "__main__":
    main()
