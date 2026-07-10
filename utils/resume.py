import os

import numpy as np
import torch


def torch_load(path, map_location=None):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def get_rng_state():
    state = {
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def set_rng_state(state):
    if not state:
        return
    if "numpy" in state:
        np.random.set_state(state["numpy"])
    if "torch" in state:
        torch.set_rng_state(state["torch"])
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def save_training_state(directory, epoch_idx, args):
    torch.save(
        {
            "epoch_idx": epoch_idx,
            "rng_state": get_rng_state(),
            "args": vars(args),
        },
        os.path.join(directory, "training_state.pth"),
    )


def load_training_state(directory):
    state_path = os.path.join(directory, "training_state.pth")
    if not os.path.exists(state_path):
        raise FileNotFoundError(f"Resume checkpoint not found: {state_path}")
    return torch_load(state_path, map_location="cpu")
