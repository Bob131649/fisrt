#!/usr/bin/env python3

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from datasets.utils import get_real_dataset
from networks.net import Value
from replay_buffer.vision_numpy_buffer import WaterPipeDataset


DEFAULT_EXPERT_DATASETS = [
    "./datasets/real_dataset/expert/franka_pickplace_ep20_20260629_152135_add_rewarding20.hdf5",
    "./datasets/real_dataset/expert/franka_pickplace_ep30_20260626_162422_add_rewarding20.hdf5",
    "./datasets/real_dataset/expert/franka_pickplace_ep31_20260625_112544_add_rewarding20.hdf5",
    "./datasets/real_dataset/expert/franka_pickplace_ep39_20260626_143912_add_rewarding20.hdf5",
]

DEFAULT_RANDOM_DATASETS = [
    "./datasets/real_dataset/random/franka_random_20260701_170633_step1909.hdf5",
    "./datasets/real_dataset/random/franka_random_20260702_154715_step2056.hdf5",
    "./datasets/real_dataset/random/franka_random_20260702_155326_step1535.hdf5",
    "./datasets/real_dataset/random/franka_random_20260702_155755_step1499.hdf5",
    "./datasets/real_dataset/random/franka_random_20260703_142557_step2013.hdf5",
]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ExpID", default=11, type=int)
    parser.add_argument("--env_name", default="franka")
    parser.add_argument("--log_dir", default="./results", type=str)
    parser.add_argument("--device", default="cuda:0", type=str)
    parser.add_argument("--sample_size", default=1500, type=int)
    parser.add_argument("--seed", default=789, type=int)
    parser.add_argument("--reward_one_samples", default=10, type=int)
    parser.add_argument(
        "--dataset_index",
        default=None,
        type=int,
        help="Optional index into both default expert/random dataset lists. If omitted, all datasets in each group are used.",
    )
    parser.add_argument(
        "--expert_dataset_path",
        action="append",
        default=[],
        help="Optional custom expert dataset path. Can be passed multiple times.",
    )
    parser.add_argument(
        "--random_dataset_path",
        action="append",
        default=[],
        help="Optional custom random dataset path. Can be passed multiple times.",
    )
    parser.add_argument(
        "--output",
        default="",
        help="Output png path. Defaults to results/ExpXXXX/env_name/vnet_xyz_mixed.png",
    )
    return parser.parse_args()


def choose_dataset_paths(default_paths, custom_paths, dataset_index):
    if custom_paths:
        return list(custom_paths)
    if dataset_index is None:
        return list(default_paths)
    if dataset_index < 0 or dataset_index >= len(default_paths):
        raise ValueError(f"--dataset_index must be in [0, {len(default_paths) - 1}]")
    return [default_paths[dataset_index]]


def build_water_pipe(dataset_paths, device, terminal_reward_count=None):
    dataset = [get_real_dataset(path) for path in dataset_paths]
    pipe = WaterPipeDataset("franka", proprio_dim=8, action_dim=8, device=device)
    pipe.load(dataset, terminal_reward_count=terminal_reward_count)
    return pipe


def apply_mixed_stats(expert_pipe, random_pipe):
    stats = WaterPipeDataset.compute_joint_stats([expert_pipe, random_pipe])
    expert_pipe.apply_stats(stats)
    random_pipe.apply_stats(stats)
    return stats


def load_vnet(model_dir, device):
    vnet = Value(8).to(device)
    state_dict = torch.load(model_dir / "model_vnet.pth", map_location=device)
    vnet.load_state_dict(state_dict)
    vnet.eval()
    return vnet


def sample_xyz_and_value(water_pipe, vnet, sample_size):
    loader = water_pipe.make_dataloader(sample_size, epoch_size=1, num_workers=0)
    state, _, _, _, _, _ = water_pipe.batch_to_device(next(iter(loader)))
    raw_state = water_pipe.unnormalize_state(state)
    xyz = raw_state["proprio"][:, :3].detach().cpu().numpy()
    with torch.no_grad():
        value = vnet(state).detach().cpu().numpy().squeeze()
    return xyz, value


def print_reward_one_xyz_and_value(expert_pipe, vnet, sample_size, seed):
    reward = expert_pipe.storage["reward"].reshape(-1)
    reward_one_indices = np.flatnonzero(reward == 1.0)
    if len(reward_one_indices) == 0 or sample_size <= 0:
        print(f"expert reward=1 transitions: {len(reward_one_indices)}")
        return

    sample_size = min(sample_size, len(reward_one_indices))
    rng = np.random.default_rng(seed)
    selected_indices = rng.choice(reward_one_indices, size=sample_size, replace=False)
    selected_pipe = expert_pipe.subset(selected_indices)
    loader = DataLoader(selected_pipe, batch_size=sample_size, shuffle=False, num_workers=0)
    state, _, _, _, _, _ = selected_pipe.batch_to_device(next(iter(loader)))
    raw_state = selected_pipe.unnormalize_state(state)
    xyz = raw_state["proprio"][:, :3].detach().cpu().numpy()
    with torch.no_grad():
        values = vnet(state).detach().cpu().numpy().reshape(-1)

    print(
        f"expert reward=1 transitions: total={len(reward_one_indices)}, "
        f"showing={sample_size}"
    )
    for index, point, value in zip(selected_indices, xyz, values):
        print(
            f"index={index} reward=1 "
            f"xyz=({point[0]:.6f}, {point[1]:.6f}, {point[2]:.6f}) "
            f"vnet={value:.6f}"
        )


def save_plot(xyz_expert, value_expert, xyz_random, value_random, output_path, title):
    all_value = np.concatenate([value_expert, value_random], axis=0)
    vmin = float(all_value.min())
    vmax = float(all_value.max())
    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, projection="3d")
    sc_random = ax.scatter(
        xyz_random[:, 0],
        xyz_random[:, 1],
        xyz_random[:, 2],
        c=value_random,
        cmap="viridis",
        alpha=0.35,
        s=9,
        vmin=vmin,
        vmax=vmax,
        marker="o",
        label="random",
    )
    ax.scatter(
        xyz_expert[:, 0],
        xyz_expert[:, 1],
        xyz_expert[:, 2],
        c=value_expert,
        cmap="viridis",
        alpha=0.75,
        s=14,
        vmin=vmin,
        vmax=vmax,
        marker="^",
        label="expert",
    )
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_title(title)
    ax.legend(loc="upper right")
    fig.colorbar(sc_random, ax=ax, label="vnet value")
    fig.tight_layout()
    fig.savefig(output_path, dpi=250)
    plt.close(fig)


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    model_dir = Path(args.log_dir) / f"Exp{args.ExpID:04d}" / args.env_name
    if not (model_dir / "model_vnet.pth").exists():
        raise FileNotFoundError(f"vnet checkpoint not found: {model_dir / 'model_vnet.pth'}")

    expert_dataset_paths = choose_dataset_paths(
        DEFAULT_EXPERT_DATASETS, args.expert_dataset_path, args.dataset_index
    )
    random_dataset_paths = choose_dataset_paths(
        DEFAULT_RANDOM_DATASETS, args.random_dataset_path, args.dataset_index
    )
    expert_pipe = build_water_pipe(
        expert_dataset_paths, args.device, terminal_reward_count=2
    )
    random_pipe = build_water_pipe(
        random_dataset_paths, args.device, terminal_reward_count=0
    )
    apply_mixed_stats(expert_pipe, random_pipe)

    output_path = Path(args.output) if args.output else model_dir / "vnet_xyz_mixed.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    normalization_path = output_path.parent / "plot_vnet_normalization_stats.json"
    expert_pipe.save_stats(normalization_path)

    vnet = load_vnet(model_dir, args.device)
    print_reward_one_xyz_and_value(
        expert_pipe, vnet, args.reward_one_samples, args.seed
    )
    expert_sample_size = max(1, args.sample_size // 2)
    random_sample_size = max(1, args.sample_size - expert_sample_size)
    xyz_expert, value_expert = sample_xyz_and_value(expert_pipe, vnet, expert_sample_size)
    xyz_random, value_random = sample_xyz_and_value(random_pipe, vnet, random_sample_size)

    save_plot(
        xyz_expert,
        value_expert,
        xyz_random,
        value_random,
        output_path,
        "V over XYZ (expert/random, mixed normalization)",
    )

    all_value = np.concatenate([value_expert, value_random], axis=0)
    value_p10, value_p90 = np.percentile(all_value, [10, 90])
    print(f"saved plot to: {output_path}")
    print(f"saved normalization stats to: {normalization_path}")
    print(f"expert dataset paths: {expert_dataset_paths}")
    print(f"random dataset paths: {random_dataset_paths}")
    print(
        "value stats:",
        f"min={all_value.min():.6f}",
        f"max={all_value.max():.6f}",
        f"mean={all_value.mean():.6f}",
        f"p10={value_p10:.6f}",
        f"p90={value_p90:.6f}",
    )


if __name__ == "__main__":
    main()
