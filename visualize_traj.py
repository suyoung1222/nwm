import argparse
import pickle
import numpy as np
import matplotlib.pyplot as plt

try:
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers '3d' projection)
    _HAS_3D = True
except ImportError:
    _HAS_3D = False


def load_traj(pkl_path):
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)
    pos = np.asarray(data["position"], dtype=np.float64)
    yaw = np.asarray(data["yaw"], dtype=np.float64).reshape(-1)
    if pos.shape[0] != yaw.shape[0]:
        raise ValueError(f"position ({pos.shape}) and yaw ({yaw.shape}) length mismatch")
    return pos, yaw


def plot_topdown(ax, pos, yaw, arrow_stride):
    x, y = pos[:, 0], pos[:, 1]
    t = np.arange(len(x))

    sc = ax.scatter(x, y, c=t, cmap="viridis", s=8, zorder=3)
    ax.plot(x, y, color="gray", linewidth=0.8, alpha=0.6, zorder=2)

    idx = np.arange(0, len(x), max(1, arrow_stride))
    ax.quiver(
        x[idx], y[idx],
        np.cos(yaw[idx]), np.sin(yaw[idx]),
        angles="xy", scale_units="xy", scale=2.0,
        color="red", width=0.003, alpha=0.7, zorder=4,
    )

    ax.scatter(x[0], y[0], c="lime", s=120, edgecolor="black", marker="o",
               label=f"start (t=0)", zorder=5)
    ax.scatter(x[-1], y[-1], c="red", s=140, edgecolor="black", marker="*",
               label=f"end (t={len(x)-1})", zorder=5)

    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title("Top-down trajectory (color = frame index, red arrows = yaw)")
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    return sc


def plot_yaw(ax, yaw):
    ax.plot(np.degrees(yaw), linewidth=1.0)
    ax.set_xlabel("frame index")
    ax.set_ylabel("yaw [deg]")
    ax.set_title("Yaw over time")
    ax.grid(True, alpha=0.3)


def plot_xy_time(ax, pos):
    ax.plot(pos[:, 0], label="x", linewidth=1.0)
    ax.plot(pos[:, 1], label="y", linewidth=1.0)
    ax.set_xlabel("frame index")
    ax.set_ylabel("position [m]")
    ax.set_title("x, y over time")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")


def plot_3d(ax, pos, yaw):
    t = np.arange(len(pos))
    x, y = pos[:, 0], pos[:, 1]
    ax.plot(x, y, t, color="gray", linewidth=0.8, alpha=0.6)
    ax.scatter(x, y, t, c=t, cmap="viridis", s=6)
    ax.scatter(x[0], y[0], 0, c="lime", s=80, edgecolor="black")
    ax.scatter(x[-1], y[-1], len(pos) - 1, c="red", s=100, marker="*", edgecolor="black")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_zlabel("frame index")
    ax.set_title("3D view (z = frame index)")


def main():
    parser = argparse.ArgumentParser(description="Visualize NWM traj_data.pkl")
    parser.add_argument("pkl", type=str, help="path to traj_data.pkl")
    parser.add_argument("--arrow_stride", type=int, default=10,
                        help="draw a yaw arrow every N frames (default 10)")
    parser.add_argument("--save", type=str, default=None,
                        help="optional path to save figure as PNG instead of showing")
    parser.add_argument("--mode", choices=["all", "2d", "3d"], default="all",
                        help="2d = topdown only, 3d = topdown + 3d, all = topdown+yaw+xy+3d")
    args = parser.parse_args()

    pos, yaw = load_traj(args.pkl)
    print(f"loaded {args.pkl}")
    print(f"  frames: {len(pos)}")
    print(f"  x range: [{pos[:, 0].min():.3f}, {pos[:, 0].max():.3f}] m  span={pos[:, 0].ptp():.3f}")
    print(f"  y range: [{pos[:, 1].min():.3f}, {pos[:, 1].max():.3f}] m  span={pos[:, 1].ptp():.3f}")
    print(f"  path length: {np.linalg.norm(np.diff(pos, axis=0), axis=1).sum():.3f} m")
    print(f"  yaw range: [{np.degrees(yaw).min():.1f}, {np.degrees(yaw).max():.1f}] deg")

    want_3d = args.mode in ("3d", "all") and _HAS_3D
    if args.mode in ("3d", "all") and not _HAS_3D:
        print("WARNING: matplotlib 3D backend unavailable, falling back to 2D-only layout")

    if args.mode == "2d" or not want_3d and args.mode == "3d":
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        sc = plot_topdown(ax, pos, yaw, args.arrow_stride)
        fig.colorbar(sc, ax=ax, label="frame index", shrink=0.8)
    elif args.mode == "3d":
        fig = plt.figure(figsize=(14, 6))
        ax1 = fig.add_subplot(1, 2, 1)
        sc = plot_topdown(ax1, pos, yaw, args.arrow_stride)
        fig.colorbar(sc, ax=ax1, label="frame index", shrink=0.8)
        ax2 = fig.add_subplot(1, 2, 2, projection="3d")
        plot_3d(ax2, pos, yaw)
    else:  # mode == "all"
        if want_3d:
            fig = plt.figure(figsize=(14, 10))
            ax1 = fig.add_subplot(2, 2, 1)
            sc = plot_topdown(ax1, pos, yaw, args.arrow_stride)
            fig.colorbar(sc, ax=ax1, label="frame index", shrink=0.8)
            ax2 = fig.add_subplot(2, 2, 2, projection="3d")
            plot_3d(ax2, pos, yaw)
            ax3 = fig.add_subplot(2, 2, 3)
            plot_xy_time(ax3, pos)
            ax4 = fig.add_subplot(2, 2, 4)
            plot_yaw(ax4, yaw)
        else:
            fig, axes = plt.subplots(1, 3, figsize=(18, 6))
            sc = plot_topdown(axes[0], pos, yaw, args.arrow_stride)
            fig.colorbar(sc, ax=axes[0], label="frame index", shrink=0.8)
            plot_xy_time(axes[1], pos)
            plot_yaw(axes[2], yaw)

    fig.tight_layout()

    if args.save:
        fig.savefig(args.save, dpi=150, bbox_inches="tight")
        print(f"saved figure to {args.save}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
