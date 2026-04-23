import argparse
import os

import h5py
import numpy as np


def print_dataset(name, obj, head=None):
    if isinstance(obj, h5py.Group):
        print(f"[GROUP] {name}")
        return

    if not isinstance(obj, h5py.Dataset):
        return

    data = obj[()]
    array = np.asarray(data)

    print("=" * 100)
    print(f"[DATASET] {name}")
    print(f"shape: {obj.shape}")
    print(f"dtype: {obj.dtype}")

    if obj.attrs:
        print("attrs:")
        for attr_key, attr_value in obj.attrs.items():
            print(f"  - {attr_key}: {attr_value}")

    if array.ndim == 0:
        print("value:")
        print(array.item())
        return

    print("content:")
    display_array = array
    if head is not None:
        if head < 0:
            raise ValueError("--head must be a non-negative integer.")
        if array.ndim >= 1:
            display_array = array[:head]
            print(f"showing first {min(head, array.shape[0])} rows along axis 0:")

    with np.printoptions(threshold=np.inf, linewidth=200):
        print(display_array)


def print_hdf5_contents(file_path, head=None):
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"Dataset file not found: {file_path}")

    with h5py.File(file_path, "r") as h5file:
        print(f"Opened HDF5 file: {file_path}")
        print("Listing all groups and datasets:")
        h5file.visititems(lambda name, obj: print_dataset(name, obj, head=head))


def main():
    parser = argparse.ArgumentParser(
        description="Load an HDF5 dataset and print every field with its content."
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        required=True,
        help="Path to a .hdf5 or .h5 dataset file.",
    )
    parser.add_argument(
        "--head",
        type=int,
        default=None,
        help="Only print the first N rows/elements of each dataset along axis 0.",
    )
    args = parser.parse_args()

    print_hdf5_contents(args.dataset_path, head=args.head)


if __name__ == "__main__":
    main()
