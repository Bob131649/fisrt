#!/usr/bin/env python3

import argparse
import shutil
from pathlib import Path

import h5py


def parse_args():
    parser = argparse.ArgumentParser(
        description="Set every value in an HDF5 reward dataset to zero."
    )
    parser.add_argument(
        "dataset_paths",
        nargs="+",
        type=Path,
        help="One or more input HDF5 files.",
    )
    parser.add_argument(
        "--in-place",
        action="store_true",
        help="Modify each input file directly instead of creating a copy.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for copied files (not valid with --in-place).",
    )
    parser.add_argument(
        "--reward-key",
        default="reward",
        help="HDF5 dataset key to reset (default: reward).",
    )
    return parser.parse_args()


def output_path_for(input_path, output_dir):
    filename = f"{input_path.stem}_reward_zero{input_path.suffix}"
    return (output_dir if output_dir is not None else input_path.parent) / filename


def reset_reward(path, reward_key):
    with h5py.File(path, "r+") as hdf5_file:
        if reward_key not in hdf5_file:
            raise KeyError(f"HDF5 dataset key not found: {reward_key}")
        reward = hdf5_file[reward_key]
        if not isinstance(reward, h5py.Dataset):
            raise TypeError(f"HDF5 key is not a dataset: {reward_key}")
        reward[...] = 0
        hdf5_file.flush()
        print(f"reset {reward.size} rewards to zero: {path}")


def main():
    args = parse_args()
    if args.in_place and args.output_dir is not None:
        raise ValueError("--output-dir cannot be used together with --in-place")

    if args.output_dir is not None:
        args.output_dir.mkdir(parents=True, exist_ok=True)

    for input_path in args.dataset_paths:
        if not input_path.is_file():
            raise FileNotFoundError(f"HDF5 file not found: {input_path}")

        target_path = input_path
        if not args.in_place:
            target_path = output_path_for(input_path, args.output_dir)
            if target_path.exists():
                raise FileExistsError(
                    f"output already exists: {target_path}; remove it or use --in-place"
                )
            shutil.copy2(input_path, target_path)

        reset_reward(target_path, args.reward_key)


if __name__ == "__main__":
    main()
