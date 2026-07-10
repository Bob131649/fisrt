#!/usr/bin/env python3

import argparse
from pathlib import Path

from datasets.utils import get_real_dataset
from replay_buffer.vision_numpy_buffer import WaterPipeDataset


EXPERT_DATASET_PATHS = [
    "./datasets/real_dataset/expert/franka_pickplace_ep20_20260629_152135_add_rewarding20.hdf5",
    "./datasets/real_dataset/expert/franka_pickplace_ep30_20260626_162422_add_rewarding20.hdf5",
    "./datasets/real_dataset/expert/franka_pickplace_ep31_20260625_112544_add_rewarding20.hdf5",
    "./datasets/real_dataset/expert/franka_pickplace_ep39_20260626_143912_add_rewarding20.hdf5",
]

RANDOM_DATASET_PATHS = [
    "./datasets/real_dataset/random/franka_random_20260701_170633_step1909.hdf5",
    "./datasets/real_dataset/random/franka_random_20260702_154715_step2056.hdf5",
    "./datasets/real_dataset/random/franka_random_20260702_155326_step1535.hdf5",
    "./datasets/real_dataset/random/franka_random_20260702_155755_step1499.hdf5",
    "./datasets/real_dataset/random/franka_random_20260703_142557_step2013.hdf5",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute expert, random, and mixed dataset normalization statistics."
    )
    parser.add_argument("--env_name", default="franka")
    parser.add_argument("--output_dir", default="./normalize", type=Path)
    return parser.parse_args()


def load_pipe(dataset_paths, env_name):
    datasets = [get_real_dataset(path) for path in dataset_paths]
    pipe = WaterPipeDataset(env_name, proprio_dim=8, action_dim=8, device="cpu")
    pipe.load(datasets)
    return pipe


def save_stats(pipe, stats, output_path):
    pipe.apply_stats(stats, normalize=False)
    pipe.save_stats(output_path)
    print(f"saved: {output_path}")


def main():
    args = parse_args()
    expert_pipe = load_pipe(EXPERT_DATASET_PATHS, args.env_name)
    random_pipe = load_pipe(RANDOM_DATASET_PATHS, args.env_name)

    mix_pipe = WaterPipeDataset.concat([random_pipe, expert_pipe])

    # Compute every result before attaching any stats to a pipe, so all three
    # calculations operate on all original, unnormalized loaded data.
    expert_stats = WaterPipeDataset.compute_joint_stats([expert_pipe])
    random_stats = WaterPipeDataset.compute_joint_stats([random_pipe])
    mix_stats = WaterPipeDataset.compute_joint_stats([mix_pipe])

    args.output_dir.mkdir(parents=True, exist_ok=True)
    save_stats(
        expert_pipe,
        expert_stats,
        args.output_dir / "expert_normalization_stats.json",
    )
    save_stats(
        random_pipe,
        random_stats,
        args.output_dir / "random_normalization_stats.json",
    )
    save_stats(
        mix_pipe,
        mix_stats,
        args.output_dir / "mix_normalization_stats.json",
    )

    print(
        "samples:",
        f"expert={expert_pipe.size}",
        f"random={random_pipe.size}",
        f"mix={mix_pipe.size}",
    )


if __name__ == "__main__":
    main()
