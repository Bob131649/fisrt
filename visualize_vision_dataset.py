#!/usr/bin/env python3

import argparse
import json
import os

import h5py
import numpy as np
import matplotlib.pyplot as plt
import torch

import algos.vision_algos_guide as algos
from replay_buffer.vision_numpy_buffer import WaterPipeDataset


EXPERT_PATHS = [
    "./datasets/real_dataset/expert/franka_pickplace_ep20_20260629_152135_add_rewarding20.hdf5",
    "./datasets/real_dataset/expert/franka_pickplace_ep30_20260626_162422_add_rewarding20.hdf5",
    "./datasets/real_dataset/expert/franka_pickplace_ep31_20260625_112544_add_rewarding20.hdf5",
    "./datasets/real_dataset/expert/franka_pickplace_ep39_20260626_143912_add_rewarding20.hdf5",
]

RANDOM_PATHS = [
    "./datasets/real_dataset/random/franka_random_20260701_170633_step1909.hdf5",
    "./datasets/real_dataset/random/franka_random_20260702_154715_step2056.hdf5",
    "./datasets/real_dataset/random/franka_random_20260702_155326_step1535.hdf5",
    "./datasets/real_dataset/random/franka_random_20260702_155755_step1499.hdf5",
    "./datasets/real_dataset/random/franka_random_20260703_142557_step2013.hdf5",
]


def get_real_dataset(dataset_path):
    if not os.path.isfile(dataset_path):
        raise FileNotFoundError(f"Dataset file not found: {dataset_path}")
    with h5py.File(dataset_path, "r") as f:
        translation = f["translation"][:]
        rotation = f["rotation"][:]
        gripper = f["gripper_w"][:].reshape(-1, 1)
        rewards = f["reward"][:]

        if "terminals" in f:
            terminals = f["terminals"][:]
        elif "terminal" in f:
            terminals = f["terminal"][:]
        elif "done" in f:
            terminals = f["done"][:]
        else:
            terminals = np.zeros(len(rewards), dtype=bool)

        if "timeout" in f:
            terminals = np.logical_or(terminals, f["timeout"][:])

    observations = np.concatenate([translation, rotation, gripper], axis=1).astype(np.float32)
    return {
        "observations": observations,
        "rewards": rewards.astype(np.float32),
        "terminals": terminals.astype(bool),
        "_dataset_path": dataset_path,
    }


def selected_paths(args):
    if args.dataset_path:
        return args.dataset_path
    if args.split == "expert":
        return EXPERT_PATHS
    if args.split == "random":
        return RANDOM_PATHS
    return EXPERT_PATHS + RANDOM_PATHS


def print_reward_stats(name, rewards):
    rewards = np.asarray(rewards).reshape(-1)
    reward_eq_one = int(np.isclose(rewards, 1.0).sum())
    reward_positive = int((rewards > 0).sum())
    print(
        f"{name}: reward==1 {reward_eq_one} | reward>0 {reward_positive} | "
        f"total {len(rewards)} | min/max/mean {rewards.min():.3f}/{rewards.max():.3f}/{rewards.mean():.3f}"
    )


def reward_indices(pipe):
    rewards = pipe.storage["reward"].reshape(-1)
    return np.flatnonzero(rewards > 0)


def source_split_name(path):
    normalized = path.replace("\\", "/")
    if "/expert/" in normalized:
        return "expert"
    if "/random/" in normalized:
        return "random"
    return "dataset"


def set_axes_equal(ax, xyz):
    center = xyz.mean(axis=0)
    radius = (xyz.max(axis=0) - xyz.min(axis=0)).max() / 2.0
    if radius <= 0:
        radius = 1.0
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)


def enable_scroll_zoom(fig, ax, scale=0.85):
    def zoom(event):
        if event.inaxes != ax:
            return
        factor = scale if event.button == "up" else 1.0 / scale
        for getter, setter in (
            (ax.get_xlim3d, ax.set_xlim3d),
            (ax.get_ylim3d, ax.set_ylim3d),
            (ax.get_zlim3d, ax.set_zlim3d),
        ):
            low, high = getter()
            center = (low + high) * 0.5
            radius = (high - low) * 0.5 * factor
            setter(center - radius, center + radius)
        fig.canvas.draw_idle()

    fig.canvas.mpl_connect("scroll_event", zoom)


def load_normalization_stats(model_dir):
    path = os.path.join(model_dir, "normalization_stats.json")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Normalization stats not found: {path}")
    with open(path, "r") as f:
        stats = json.load(f)
    return {
        "s_mean": stats.get("s_mean", stats["state_mean"]),
        "s_std": stats.get("s_std", stats["state_std"]),
        "a_mean": stats.get("a_mean", stats["action_mean"]),
        "a_std": stats.get("a_std", stats["action_std"]),
    }


def load_train_all_policy(args):
    latent_dim = max(4, int(args.action_dim * 2.0))
    policy = algos.Latent(
        args.state_dim,
        args.action_dim,
        latent_dim,
        0,
        100,
        device=args.device,
        max_latent_action=0.675,
    )
    policy.load_policy("model", args.model_dir)
    policy.eval()
    return policy


def make_state_batch(pipe, indices):
    source_indices = pipe.storage["image_source_index"][indices]
    image_indices = pipe.storage["image_index"][indices]
    images = []
    for source_idx, image_idx in zip(source_indices, image_indices):
        rgb_cache = pipe._load_rgb_cache(int(source_idx))
        images.append(pipe._image_tensor(rgb_cache, int(image_idx)))
    image = torch.stack(images, dim=0).to(pipe.device).float() / 255.0
    proprio = torch.from_numpy(pipe.storage["state"][indices]).float().to(pipe.device)
    return {"proprio": proprio, "image": image}


def select_plot_indices(pipe, sample_size, reward_only=False, seed=0):
    candidates = reward_indices(pipe) if reward_only else np.arange(pipe.size)
    if len(candidates) == 0:
        raise RuntimeError("No samples to plot after filtering.")
    if sample_size <= 0 or sample_size >= len(candidates):
        return candidates
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(candidates, size=sample_size, replace=False))


def compute_policy_q(policy, pipe, indices, batch_size=256):
    values = []
    was_training = policy.training
    policy.eval()
    with torch.no_grad():
        for start in range(0, len(indices), batch_size):
            batch_indices = indices[start:start + batch_size]
            state = make_state_batch(pipe, batch_indices)
            latent = policy.actor(state)
            action = policy.actor_vae_target.decode(state, z=latent)
            q1, q2 = policy.critic(state, action)
            values.append(torch.min(q1, q2).detach().cpu().numpy().reshape(-1))
    if was_training:
        policy.train()
    return np.concatenate(values, axis=0)


def plot_dataset_xyz(pipe, xyz, indices, values):
    rewards = pipe.storage["reward"].reshape(-1)
    source_indices = pipe.storage["image_source_index"].reshape(-1)

    xyz_plot = xyz[indices]
    values_plot = values
    rewards_plot = rewards[indices]
    sources_plot = source_indices[indices]
    vmin, vmax = np.percentile(values_plot, [5, 95])
    if np.isclose(vmin, vmax):
        vmin, vmax = None, None

    fig = plt.figure(figsize=(10, 8))
    try:
        fig.canvas.manager.set_window_title("train-all critic heatmap")
    except Exception:
        pass
    ax = fig.add_subplot(111, projection="3d")

    styles = {
        "expert": {"marker": "o", "label": "expert", "edgecolor": "#1b4332"},
        "random": {"marker": "^", "label": "random", "edgecolor": "#7f4f24"},
        "dataset": {"marker": "o", "label": "dataset", "edgecolor": "#222222"},
    }
    last_scatter = None
    for source_idx, path in enumerate(pipe.dataset_paths):
        split_name = source_split_name(path)
        mask = sources_plot == source_idx
        if not np.any(mask):
            continue
        style = styles[split_name]
        points = xyz_plot[mask]
        last_scatter = ax.scatter(
            points[:, 0],
            points[:, 1],
            points[:, 2],
            s=22,
            alpha=0.45,
            marker=style["marker"],
            c=values_plot[mask],
            cmap="viridis",
            vmin=vmin,
            vmax=vmax,
            edgecolors=style["edgecolor"],
            linewidths=0.35,
            label=style["label"] if style["label"] not in ax.get_legend_handles_labels()[1] else None,
        )

    success_mask = rewards_plot > 0
    if np.any(success_mask):
        points = xyz_plot[success_mask]
        ax.scatter(
            points[:, 0],
            points[:, 1],
            points[:, 2],
            s=70,
            alpha=0.9,
            marker="*",
            c=values_plot[success_mask],
            cmap="viridis",
            vmin=vmin,
            vmax=vmax,
            edgecolors="#d62828",
            linewidths=0.8,
            label="reward > 0",
        )

    set_axes_equal(ax, xyz_plot)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_title(f"Train-all critic heatmap Q(s, pi(s)) ({len(indices)} sampled transitions)")
    ax.legend(loc="best")
    fig.colorbar(last_scatter, ax=ax, label="min Q(policy action)")
    enable_scroll_zoom(fig, ax)
    fig.tight_layout()
    print("matplotlib window: drag to rotate, mouse wheel to zoom, close window to exit.")
    plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=("expert", "random", "all"), default="all")
    parser.add_argument("--dataset-path", action="append", default=None)
    parser.add_argument("--env-name", default="franka")
    parser.add_argument("--state-dim", type=int, default=8)
    parser.add_argument("--action-dim", type=int, default=8)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--model-dir", default="./results/Exp0111/franka")
    parser.add_argument("--sample-size", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--reward-only", action="store_true")
    args = parser.parse_args()

    paths = selected_paths(args)
    datasets = [get_real_dataset(path) for path in paths]

    print("raw hdf5 reward stats:")
    for path, dataset in zip(paths, datasets):
        print_reward_stats(path, dataset["rewards"])

    print("\nafter delta-threshold transition stats:")
    total_processed_rewards = []
    for path, dataset in zip(paths, datasets):
        single_pipe = WaterPipeDataset(args.env_name, args.state_dim, args.action_dim, args.device)
        single_pipe.load(dataset)
        rewards = single_pipe.storage["reward"].reshape(-1)
        total_processed_rewards.append(rewards)
        print_reward_stats(path, rewards)

    pipe = WaterPipeDataset(args.env_name, args.state_dim, args.action_dim, args.device)
    pipe.load(datasets)
    xyz = pipe.storage["state"][:, :3].copy()
    pipe.apply_stats(load_normalization_stats(args.model_dir))
    policy = load_train_all_policy(args)
    indices = select_plot_indices(pipe, args.sample_size, reward_only=args.reward_only, seed=args.seed)
    values = compute_policy_q(policy, pipe, indices)

    rewards = pipe.storage["reward"].reshape(-1)
    successes = reward_indices(pipe)
    if total_processed_rewards:
        print_reward_stats("merged processed dataset", np.concatenate(total_processed_rewards, axis=0))
    print(f"\nloaded transitions: {pipe.size}")
    print(f"positive reward transitions used by viewer: {len(successes)}")
    if len(successes):
        print(f"first positive reward sample indices: {successes[:20].tolist()}")
    print(f"reward min/max/mean: {rewards.min():.3f} / {rewards.max():.3f} / {rewards.mean():.3f}")
    print(f"plot sampled transitions: {len(indices)}")
    print(f"critic q min/max/mean: {values.min():.3f} / {values.max():.3f} / {values.mean():.3f}")
    plot_dataset_xyz(pipe, xyz, indices, values)


if __name__ == "__main__":
    main()
