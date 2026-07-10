import os, h5py, d4rl
import numpy as np


def load_hdf5_dataset(dataset_path, skip_keys=()):
    dataset = {}
    skip_keys = set(skip_keys)

    def should_skip(key):
        return key in skip_keys or key.split("/")[0] in skip_keys

    with h5py.File(dataset_path, "r") as f:
        for key in f.keys():
            if should_skip(key):
                continue
            obj = f[key]
            if isinstance(obj, h5py.Dataset):
                dataset[key] = obj[:]
            elif isinstance(obj, h5py.Group):
                for sub_key in obj.keys():
                    flat_key = f"{key}/{sub_key}"
                    if not should_skip(flat_key):
                        dataset[flat_key] = obj[sub_key][:]

    required_keys = ["observations", "actions", "rewards", "terminals"]
    missing_keys = [key for key in required_keys if key not in dataset]
    if missing_keys:
        raise ValueError(
            f"Dataset file is missing required keys: {', '.join(missing_keys)}"
        )
    return dataset

def get_dataset(env, dataset_path="", preserve_keys=()):
    if dataset_path != "":
        if not os.path.isfile(dataset_path):
            raise FileNotFoundError(f"Dataset file not found: {dataset_path}")
        print(f"loading custom dataset from: {dataset_path}")
        raw_dataset = load_hdf5_dataset(dataset_path, skip_keys=preserve_keys)
        dataset = d4rl.qlearning_dataset(env, dataset=raw_dataset)
        for key in preserve_keys:
            if key in raw_dataset:
                dataset[key] = raw_dataset[key]
        dataset["_dataset_path"] = dataset_path
    else:
        dataset = d4rl.qlearning_dataset(env)  # Load d4rl dataset
    
    return dataset

def get_real_dataset(dataset_path):
    if not os.path.isfile(dataset_path):
        raise FileNotFoundError(f"Dataset file not found: {dataset_path}")
    print(f"loading real dataset from: {dataset_path}")
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
