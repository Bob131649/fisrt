import os, h5py, d4rl


def load_hdf5_dataset(dataset_path):
    dataset = {}
    with h5py.File(dataset_path, "r") as f:
        for key in f.keys():
            obj = f[key]
            if isinstance(obj, h5py.Dataset):
                dataset[key] = obj[:]
            elif isinstance(obj, h5py.Group):
                for sub_key in obj.keys():
                    dataset[f"{key}/{sub_key}"] = obj[sub_key][:]

    required_keys = ["observations", "actions", "rewards", "terminals"]
    missing_keys = [key for key in required_keys if key not in dataset]
    if missing_keys:
        raise ValueError(
            f"Dataset file is missing required keys: {', '.join(missing_keys)}"
        )
    return dataset

def get_dataset(env, dataset_path=""):
    if dataset_path != "":
        if not os.path.isfile(dataset_path):
            raise FileNotFoundError(f"Dataset file not found: {dataset_path}")
        print(f"loading custom dataset from: {dataset_path}")
        raw_dataset = load_hdf5_dataset(dataset_path)
        dataset = d4rl.qlearning_dataset(env, dataset=raw_dataset)
    else:
        dataset = d4rl.qlearning_dataset(env)  # Load d4rl dataset
    
    return dataset