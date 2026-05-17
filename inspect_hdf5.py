#!/usr/bin/env python3

import argparse
import os

import h5py
import numpy as np


def format_array_preview(array, num_items, full):
    preview = array if full else array[:num_items]
    return np.array2string(
        preview,
        precision=6,
        threshold=np.inf if full else 200,
        edgeitems=3,
        suppress_small=False,
    )


def format_dataset_summary(name, dataset, num_items, full):
    lines = [
        f"[Dataset] {name}",
        f"  shape: {dataset.shape}",
        f"  dtype: {dataset.dtype}",
    ]

    if dataset.shape == ():
        value = dataset[()]
        lines.append(f"  scalar: {value}")
        return lines

    if len(dataset) == 0:
        lines.append("  preview: []")
        return lines

    data = dataset[:]
    if np.issubdtype(data.dtype, np.number) or np.issubdtype(data.dtype, np.bool_):
        lines.extend(
            [
                f"  min: {np.min(data)}",
                f"  max: {np.max(data)}",
                f"  mean: {np.mean(data)}",
            ]
        )

    if full:
        lines.append("  content:")
    else:
        lines.append(f"  preview first {min(num_items, len(dataset))}:")
    lines.append(format_array_preview(data, num_items, full))
    return lines


def inspect_hdf5(dataset_path, output_path, num_items, full):
    lines = [
        f"HDF5 file: {os.path.abspath(dataset_path)}",
        "",
    ]

    with h5py.File(dataset_path, "r") as h5_file:
        def collect(name, obj):
            if isinstance(obj, h5py.Group):
                lines.append(f"[Group] {name}/")
                lines.append("")
                return

            lines.extend(format_dataset_summary(name, obj, num_items, full))
            lines.append("")

        h5_file.visititems(collect)

    with open(output_path, "w", encoding="utf-8") as output_file:
        output_file.write("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(
        description="Inspect an HDF5 dataset and write all field contents to a txt file."
    )
    parser.add_argument("--dataset_path", required=True, help="Path to the .hdf5 file.")
    parser.add_argument(
        "--output",
        default="hdf5_inspect.txt",
        help="Output txt path. Default: hdf5_inspect.txt",
    )
    parser.add_argument(
        "--num_items",
        default=3,
        type=int,
        help="Number of leading rows/items to write when --preview is used.",
    )
    parser.add_argument(
        "--preview",
        action="store_true",
        help="Only write a short preview instead of full dataset contents.",
    )
    args = parser.parse_args()

    if not os.path.isfile(args.dataset_path):
        raise FileNotFoundError(f"Dataset file not found: {args.dataset_path}")

    output_dir = os.path.dirname(os.path.abspath(args.output))
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    inspect_hdf5(args.dataset_path, args.output, args.num_items, not args.preview)
    print(f"wrote hdf5 summary to: {args.output}")


if __name__ == "__main__":
    main()
