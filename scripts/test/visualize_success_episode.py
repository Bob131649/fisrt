import argparse
import os

import h5py
import matplotlib.pyplot as plt
import numpy as np


def compute_episode_ranges(terminals, timeouts):
    """Use terminals/timeouts only to split trajectories."""
    done = np.logical_or(terminals.astype(bool), timeouts.astype(bool))
    done_indices = np.where(done)[0]

    if len(done_indices) == 0:
        return [(0, len(terminals))]

    episode_starts = np.concatenate([[0], done_indices[:-1] + 1])
    episode_ranges = [(int(s), int(e) + 1) for s, e in zip(episode_starts, done_indices)]

    last_end = done_indices[-1] + 1
    if last_end < len(terminals):
        episode_ranges.append((int(last_end), len(terminals)))

    return episode_ranges


def infer_episode_success(rewards, start_idx, end_idx, success_reward=0.9):
    """For sparse-reward datasets, reward >= threshold marks success."""
    return bool(np.any(rewards[start_idx:end_idx] >= success_reward))


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


def load_successful_xy_from_hdf5(h5file, use_qpos=False, success_reward=0.9):
    xy, source = load_xy_from_hdf5(h5file, use_qpos=use_qpos)

    required_keys = ["rewards", "terminals", "timeouts"]
    missing_keys = [key for key in required_keys if key not in h5file]
    if missing_keys:
        raise ValueError(
            "Dataset is missing keys required for success filtering: "
            + ", ".join(missing_keys)
        )

    rewards = np.asarray(h5file["rewards"][:]).reshape(-1)
    terminals = np.asarray(h5file["terminals"][:]).reshape(-1)
    timeouts = np.asarray(h5file["timeouts"][:]).reshape(-1)

    if not (len(xy) == len(rewards) == len(terminals) == len(timeouts)):
        raise ValueError(
            "Dataset arrays have inconsistent lengths: "
            f"xy={len(xy)}, rewards={len(rewards)}, terminals={len(terminals)}, "
            f"timeouts={len(timeouts)}"
        )

    episode_ranges = compute_episode_ranges(terminals, timeouts)
    successful_ranges = [
        (start_idx, end_idx)
        for start_idx, end_idx in episode_ranges
        if infer_episode_success(rewards, start_idx, end_idx, success_reward)
    ]

    if not successful_ranges:
        raise ValueError(
            f"No successful episodes found with success_reward >= {success_reward}."
        )

    successful_segments = []
    reward_hit_segments = []
    start_points = []
    for start_idx, end_idx in successful_ranges:
        episode_xy = xy[start_idx:end_idx]
        reward_hit_mask = np.isclose(rewards[start_idx:end_idx], 1.0)
        if episode_xy.shape[0] > 0:
            start_points.append(episode_xy[0:1])
        successful_segments.append(episode_xy[~reward_hit_mask])
        reward_hit_segments.append(episode_xy[reward_hit_mask])

    non_empty_successful_segments = [
        successful_segment
        for successful_segment in successful_segments
        if successful_segment.shape[0] > 0
    ]
    if non_empty_successful_segments:
        successful_xy = np.concatenate(non_empty_successful_segments, axis=0)
    else:
        successful_xy = np.empty((0, 2), dtype=np.float64)

    non_empty_reward_hit_segments = [
        reward_hit_segment
        for reward_hit_segment in reward_hit_segments
        if reward_hit_segment.shape[0] > 0
    ]
    if non_empty_reward_hit_segments:
        reward_hit_xy = np.concatenate(non_empty_reward_hit_segments, axis=0)
    else:
        reward_hit_xy = np.empty((0, 2), dtype=np.float64)

    if start_points:
        start_xy = np.concatenate(start_points, axis=0)
    else:
        start_xy = np.empty((0, 2), dtype=np.float64)

    return (
        successful_xy,
        reward_hit_xy,
        start_xy,
        source,
        len(episode_ranges),
        len(successful_ranges),
    )


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
    reward_hit_xy=None,
    start_xy=None,
    title=None,
    output_path=None,
    figsize=(6.5, 6.0),
    dpi=300,
    marker="x",
    color="#00c8ff",
    alpha=0.15,
    size=1,
    linewidths=0.1,
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

    if reward_hit_xy is not None and reward_hit_xy.shape[0] > 0:
        ax.scatter(
            reward_hit_xy[:, 0],
            reward_hit_xy[:, 1],
            s=size,
            c="#ffd400",
            alpha=0.05,
            marker="x",
            linewidths=max(linewidths, 0.5),
        )

    if start_xy is not None and start_xy.shape[0] > 0:
        ax.scatter(
            start_xy[:, 0],
            start_xy[:, 1],
            s=max(size * 8, 8),
            c="#ff2d2d",
            alpha=0.9,
            marker="o",
            linewidths=0.0,
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
        description="Plot successful-episode state coverage as scatter points from an HDF5 file."
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
        "--success_reward",
        type=float,
        default=1.0,
        help="Reward threshold used to decide whether an episode is successful.",
    )
    parser.add_argument(
        "--max_points",
        type=int,
        default=1000000,
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
        default=0.05,
        help="Marker transparency.",
    )
    parser.add_argument(
        "--size",
        type=float,
        default=1,
        help="Marker size.",
    )
    parser.add_argument(
        "--linewidths",
        type=float,
        default=1.0,
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
        default="success_episode_scatter.png",
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
        xy, reward_hit_xy, start_xy, source, total_episodes, successful_episodes = load_successful_xy_from_hdf5(
            f,
            use_qpos=args.use_qpos,
            success_reward=args.success_reward,
        )

    xy = filter_valid_xy(xy)
    xy = sample_points(xy, max_points=args.max_points, seed=args.seed)
    if reward_hit_xy.shape[0] > 0:
        reward_hit_xy = filter_valid_xy(reward_hit_xy)
        reward_hit_xy = sample_points(
            reward_hit_xy,
            max_points=args.max_points,
            seed=args.seed,
        )
    if start_xy.shape[0] > 0:
        start_xy = filter_valid_xy(start_xy)

    print(f"Dataset visualization summary: {xy.shape[0]}")
    print(f"Loaded points from: {source}")
    print(f"Total episodes: {total_episodes}")
    print(f"Successful episodes: {successful_episodes}")
    print(f"Number of plotted points: {xy.shape[0]}")
    print(f"Number of reward-hit points: {reward_hit_xy.shape[0]}")
    print(f"Number of start points: {start_xy.shape[0]}")
    print(f"x range: [{xy[:, 0].min():.4f}, {xy[:, 0].max():.4f}]")
    print(f"y range: [{xy[:, 1].min():.4f}, {xy[:, 1].max():.4f}]")

    title = args.title
    if title is None:
        dataset_name = os.path.splitext(os.path.basename(args.dataset_path))[0]
        title = f"{dataset_name} successful episodes"

    plot_scatter(
        xy=xy,
        reward_hit_xy=reward_hit_xy,
        start_xy=start_xy,
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
