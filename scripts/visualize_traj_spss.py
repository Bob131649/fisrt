import argparse
import os

import h5py
import numpy as np
import matplotlib.pyplot as plt


def load_xy_from_hdf5(h5file, use_qpos=False):
    """
    Load xy positions from an HDF5 dataset.

    Priority:
    - if use_qpos=True and infos/qpos exists: use infos/qpos[:, :2]
    - otherwise use observations[:, :2]
    """
    if use_qpos and "infos/qpos" in h5file:
        qpos = h5file["infos/qpos"][:]
        if qpos.ndim == 2 and qpos.shape[1] >= 2:
            return np.asarray(qpos[:, :2], dtype=np.float64), "infos/qpos[:, :2]"

    if "observations" in h5file:
        obs = h5file["observations"][:]
        if obs.ndim == 2 and obs.shape[1] >= 2:
            return np.asarray(obs[:, :2], dtype=np.float64), "observations[:, :2]"

    raise ValueError("Could not find usable xy positions in dataset.")


def filter_valid_xy(xy):
    xy = np.asarray(xy, dtype=np.float64)
    mask = np.isfinite(xy).all(axis=1)
    xy = xy[mask]
    if xy.shape[0] == 0:
        raise ValueError("No valid finite xy points found.")
    return xy


def sample_points(xy, max_points=None, seed=0):
    if max_points is None or xy.shape[0] <= max_points:
        return xy

    rng = np.random.default_rng(seed)
    indices = rng.choice(xy.shape[0], size=max_points, replace=False)
    return xy[indices]


def plot_scatter(
    xy,
    title=None,
    output_path=None,
    figsize=(6.5, 6.0),
    dpi=300,
    marker="x",
    color="#00c8ff",
    alpha=0.35,
    size=40,
    linewidths=1.5,
    show_axes=True,
):
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    ax.scatter(
        xy[:, 0],
        xy[:, 1],
        s=size,
        c=color,
        alpha=alpha,
        marker=marker,
        linewidths=linewidths,
    )

    ax.set_aspect("equal", adjustable="box")

    if title is not None:
        ax.set_title(title, fontsize=14)

    if show_axes:
        ax.set_xlabel("x", fontsize=11)
        ax.set_ylabel("y", fontsize=11)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    else:
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

    plt.tight_layout()

    if output_path is not None:
        fig.savefig(output_path, bbox_inches="tight")
        print(f"Saved figure to: {output_path}")

    plt.show()


def main():
    parser = argparse.ArgumentParser(
        description="Plot dataset state coverage as scatter points from an HDF5 file."
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        required=True,
        help="Path to a .hdf5 or .h5 file.",
    )
    parser.add_argument(
        "--use_qpos",
        action="store_true",
        help="Use infos/qpos[:, :2] instead of observations[:, :2]. Recommended for AntMaze.",
    )
    parser.add_argument(
        "--max_points",
        type=int,
        default=10000,
        help="Maximum number of points to draw. Randomly subsamples if dataset is larger.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for point subsampling.",
    )
    parser.add_argument(
        "--marker",
        type=str,
        default="x",
        help="Matplotlib marker style, e.g. x, o, ., +",
    )
    parser.add_argument(
        "--color",
        type=str,
        default="#00c8ff",
        help="Marker color.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.25,
        help="Marker transparency.",
    )
    parser.add_argument(
        "--size",
        type=float,
        default=40,
        help="Marker size.",
    )
    parser.add_argument(
        "--linewidths",
        type=float,
        default=1.5,
        help="Marker line width.",
    )
    parser.add_argument(
        "--title",
        type=str,
        default=None,
        help="Optional figure title.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="dataset_scatter.png",
        help="Output figure path.",
    )
    parser.add_argument(
        "--hide_axes",
        action="store_true",
        help="Hide axes for a cleaner paper-style figure.",
    )
    args = parser.parse_args()

    if not os.path.isfile(args.dataset_path):
        raise FileNotFoundError(f"Dataset file not found: {args.dataset_path}")

    with h5py.File(args.dataset_path, "r") as f:
        xy, source = load_xy_from_hdf5(f, use_qpos=args.use_qpos)

    xy = filter_valid_xy(xy)
    xy = sample_points(xy, max_points=args.max_points, seed=args.seed)

    print(f"Dataset visualization summary:{xy.shape[0]}")
    print(f"Loaded points from: {source}")
    print(f"Number of plotted points: {xy.shape[0]}")
    print(f"x range: [{xy[:, 0].min():.4f}, {xy[:, 0].max():.4f}]")
    print(f"y range: [{xy[:, 1].min():.4f}, {xy[:, 1].max():.4f}]")

    title = args.title
    if title is None:
        title = os.path.basename(args.dataset_path).replace(".hdf5", "")

    plot_scatter(
        xy=xy,
        title=title,
        output_path=args.output_path,
        marker=args.marker,
        color=args.color,
        alpha=args.alpha,
        size=args.size,
        linewidths=args.linewidths,
        show_axes=not args.hide_axes,
    )


if __name__ == "__main__":
    main()