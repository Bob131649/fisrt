import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data._utils.collate import default_collate


class NoAugmentSubset(torch.utils.data.Dataset):
    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = list(indices)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        return self.dataset.get_item(self.indices[idx], augment=False)


def safe_mean(values):
    return float(np.mean(values)) if values else float("nan")


def safe_column_mean(values):
    return np.mean(np.asarray(values, dtype=np.float64), axis=0) if values else np.asarray([])


def unwrap_dataset(dataset):
    while hasattr(dataset, "dataset"):
        dataset = dataset.dataset
    return dataset


def maybe_unnormalize_action(action, dataset):
    dataset = unwrap_dataset(dataset)
    if not getattr(dataset, "normalize_action_flag", False):
        return action
    action_mean = torch.as_tensor(
        dataset.action_mean,
        dtype=action.dtype,
        device=action.device,
    )
    action_std = torch.as_tensor(
        dataset.action_std,
        dtype=action.dtype,
        device=action.device,
    )
    return action * (action_std + 0.000001) + action_mean


def split_train_val_indices(dataset_size, val_fraction, seed):
    indices = np.arange(dataset_size)
    rng = np.random.default_rng(seed)
    rng.shuffle(indices)

    val_size = int(round(dataset_size * val_fraction))
    if val_fraction > 0 and dataset_size > 1:
        val_size = min(max(val_size, 1), dataset_size - 1)
    else:
        val_size = 0

    val_indices = indices[:val_size]
    train_indices = indices[val_size:]
    return train_indices.tolist(), val_indices.tolist()


def evaluate_vae_validation(policy, dataloader):
    if policy.actor_vae is None:
        return {
            "recon": float("nan"),
            "recon_sum": float("nan"),
            "raw_recon": float("nan"),
            "raw_recon_sum": float("nan"),
            "raw_recon_dim_mse": [],
            "raw_recon_dim_rmse": [],
            "kl": float("nan"),
        }

    recon_values = []
    recon_sum_values = []
    raw_recon_values = []
    raw_recon_sum_values = []
    raw_recon_dim_values = []
    kl_values = []
    was_training = policy.training
    policy.eval()

    with torch.no_grad():
        for batch in dataloader:
            action = batch["action"].to(policy.device).float()
            state = policy._batch_feature(batch)
            recons_action, mu, log_var = policy.actor_vae(state, action)

            recon_per_dim = F.mse_loss(recons_action, action, reduction="none")
            recon_values.append(recon_per_dim.mean().item())
            recon_sum_values.append(torch.sum(recon_per_dim, dim=1).mean().item())

            raw_action = maybe_unnormalize_action(action, dataloader.dataset)
            raw_recons_action = maybe_unnormalize_action(recons_action, dataloader.dataset)
            raw_recon_per_dim = F.mse_loss(raw_recons_action, raw_action, reduction="none")
            raw_recon_values.append(raw_recon_per_dim.mean().item())
            raw_recon_sum_values.append(torch.sum(raw_recon_per_dim, dim=1).mean().item())
            raw_recon_dim_values.append(raw_recon_per_dim.mean(dim=0).cpu().numpy())

            free_bits = 0.5
            kl_per_dim = -0.5 * (1 + log_var - mu.pow(2) - log_var.exp())
            kl_freebits = torch.maximum(
                kl_per_dim,
                torch.tensor(free_bits, device=kl_per_dim.device),
            )
            kl_values.append(kl_freebits.sum(dim=1).mean().item())

    if was_training:
        policy.train()

    raw_recon_dim_mse = safe_column_mean(raw_recon_dim_values)
    raw_recon_dim_rmse = np.sqrt(raw_recon_dim_mse)

    return {
        "recon": safe_mean(recon_values),
        "recon_sum": safe_mean(recon_sum_values),
        "raw_recon": safe_mean(raw_recon_values),
        "raw_recon_sum": safe_mean(raw_recon_sum_values),
        "raw_recon_dim_mse": raw_recon_dim_mse.tolist(),
        "raw_recon_dim_rmse": raw_recon_dim_rmse.tolist(),
        "kl": safe_mean(kl_values),
    }


def build_episode_indices(dataset, allowed_indices=None, min_len=2):
    allowed = set(allowed_indices) if allowed_indices is not None else None
    episodes = []
    current = []
    prev_segment = None
    prev_raw_idx = None

    for idx in range(len(dataset)):
        if allowed is not None and idx not in allowed:
            if len(current) >= min_len:
                episodes.append(current)
            current = []
            prev_segment = None
            prev_raw_idx = None
            continue

        segment = int(dataset.segment_ids[idx])
        raw_idx = int(dataset.indices[idx])
        is_new_episode = (
            prev_segment is None
            or segment != prev_segment
            or prev_raw_idx is None
            or raw_idx != prev_raw_idx + 1
        )
        if is_new_episode and current:
            if len(current) >= min_len:
                episodes.append(current)
            current = []
        current.append(idx)
        prev_segment = segment
        prev_raw_idx = raw_idx

    if len(current) >= min_len:
        episodes.append(current)
    return episodes


def chain_threshold_episode_indices(dataset, episode_indices):
    """Keep threshold transitions as a chain: after start->end, continue at end."""
    if not episode_indices:
        return []

    raw_to_episode_idx = {
        int(dataset.indices[int(dataset_idx)]): int(dataset_idx)
        for dataset_idx in episode_indices
    }
    sorted_raw_starts = sorted(raw_to_episode_idx)
    max_raw_start = sorted_raw_starts[-1]

    chained = []
    current_dataset_idx = int(episode_indices[0])
    visited = set()
    while current_dataset_idx not in visited:
        visited.add(current_dataset_idx)
        chained.append(current_dataset_idx)

        next_raw = int(dataset.next_indices[current_dataset_idx])
        if next_raw > max_raw_start:
            break
        if next_raw in raw_to_episode_idx:
            current_dataset_idx = raw_to_episode_idx[next_raw]
            continue

        later_starts = [raw for raw in sorted_raw_starts if raw > next_raw]
        if not later_starts:
            break
        current_dataset_idx = raw_to_episode_idx[later_starts[0]]

    return chained


def chain_threshold_episodes(dataset, episodes, min_len=1):
    chained_episodes = [
        chain_threshold_episode_indices(dataset, episode) for episode in episodes
    ]
    return [episode for episode in chained_episodes if len(episode) >= min_len]


def split_train_val_episode_indices(dataset, val_fraction, seed):
    episodes = build_episode_indices(dataset)
    if not episodes:
        train_indices, val_indices = split_train_val_indices(len(dataset), val_fraction, seed)
        return train_indices, val_indices, []

    episode_ids = np.arange(len(episodes))
    rng = np.random.default_rng(seed)
    rng.shuffle(episode_ids)

    val_size = int(round(len(episodes) * val_fraction))
    if val_fraction > 0 and len(episodes) > 1:
        val_size = min(max(val_size, 1), len(episodes) - 1)
    else:
        val_size = 0

    val_episode_ids = set(episode_ids[:val_size].tolist())
    train_episodes = [
        episode for idx, episode in enumerate(episodes) if idx not in val_episode_ids
    ]
    val_episodes = [
        episode for idx, episode in enumerate(episodes) if idx in val_episode_ids
    ]

    train_indices = [idx for episode in train_episodes for idx in episode]
    val_indices = [idx for episode in val_episodes for idx in episode]
    return train_indices, val_indices, val_episodes


def describe_episode(dataset, episode_indices):
    first_idx = int(episode_indices[0])
    segment_id = int(dataset.segment_ids[first_idx])
    raw_start = int(dataset.indices[first_idx])
    raw_end = int(dataset.next_indices[int(episode_indices[-1])])

    dataset_path = None
    if hasattr(dataset, "dataset_paths") and segment_id < len(dataset.dataset_paths):
        dataset_path = dataset.dataset_paths[segment_id]
    dataset_name = os.path.basename(dataset_path) if dataset_path is not None else f"segment_{segment_id}"

    episode_id_in_dataset = 0
    for episode in build_episode_indices(dataset):
        episode_first_idx = int(episode[0])
        episode_segment_id = int(dataset.segment_ids[episode_first_idx])
        if episode_segment_id != segment_id:
            continue
        episode_raw_start = int(dataset.indices[episode_first_idx])
        if episode_raw_start >= raw_start:
            break
        episode_id_in_dataset += 1

    return {
        "segment_id": segment_id,
        "dataset_path": dataset_path,
        "dataset_name": dataset_name,
        "episode_id_in_dataset": episode_id_in_dataset,
        "raw_start": raw_start,
        "raw_end": raw_end,
    }


def batch_to_device(batch, device):
    moved = {}
    for key, value in batch.items():
        if torch.is_tensor(value):
            moved[key] = value.to(device)
        else:
            moved[key] = value
    return moved


def compute_episode_losses(policy, dataset, episode_indices, batch_size, free_bits=0.5):
    recon_mean_values = []
    recon_sum_values = []
    raw_recon_mean_values = []
    raw_recon_sum_values = []
    raw_recon_dim_values = []
    kl_values = []
    was_training = policy.training
    policy.eval()

    with torch.no_grad():
        for start in range(0, len(episode_indices), batch_size):
            chunk_indices = episode_indices[start : start + batch_size]
            samples = [dataset.get_item(idx, augment=False) for idx in chunk_indices]
            batch = batch_to_device(default_collate(samples), policy.device)

            action = batch["action"].float()
            state = policy._batch_feature(batch)
            recons_action, mu, log_var = policy.actor_vae(state, action)

            recon_per_dim = F.mse_loss(recons_action, action, reduction="none")
            recon_mean_values.extend(recon_per_dim.mean(dim=1).cpu().numpy().tolist())
            recon_sum_values.extend(recon_per_dim.sum(dim=1).cpu().numpy().tolist())

            raw_action = maybe_unnormalize_action(action, dataset)
            raw_recons_action = maybe_unnormalize_action(recons_action, dataset)
            raw_recon_per_dim = F.mse_loss(raw_recons_action, raw_action, reduction="none")
            raw_recon_mean_values.extend(raw_recon_per_dim.mean(dim=1).cpu().numpy().tolist())
            raw_recon_sum_values.extend(raw_recon_per_dim.sum(dim=1).cpu().numpy().tolist())
            raw_recon_dim_values.append(raw_recon_per_dim.mean(dim=0).cpu().numpy())

            kl_per_dim = -0.5 * (1 + log_var - mu.pow(2) - log_var.exp())
            if free_bits is not None and free_bits > 0:
                kl_per_dim = torch.maximum(
                    kl_per_dim,
                    torch.tensor(float(free_bits), device=kl_per_dim.device),
                )
            kl_values.extend(kl_per_dim.sum(dim=1).cpu().numpy().tolist())

    if was_training:
        policy.train()

    raw_recon_dim_mse = safe_column_mean(raw_recon_dim_values)
    raw_recon_dim_rmse = np.sqrt(raw_recon_dim_mse)

    return {
        "recon_mean": np.asarray(recon_mean_values, dtype=np.float32),
        "recon_sum": np.asarray(recon_sum_values, dtype=np.float32),
        "raw_recon_mean": np.asarray(raw_recon_mean_values, dtype=np.float32),
        "raw_recon_sum": np.asarray(raw_recon_sum_values, dtype=np.float32),
        "raw_recon_dim_mse": raw_recon_dim_mse.astype(np.float32),
        "raw_recon_dim_rmse": raw_recon_dim_rmse.astype(np.float32),
        "kl": np.asarray(kl_values, dtype=np.float32),
    }


def plot_episode_losses(losses, output_path, title):
    steps = np.arange(len(losses["kl"]))
    fig, ax_recon = plt.subplots(figsize=(12, 5))
    ax_kl = ax_recon.twinx()

    line_recon, = ax_recon.plot(
        steps,
        losses["raw_recon_mean"],
        color="tab:blue",
        label="Raw action Recon MSE",
        linewidth=2,
    )
    line_kl, = ax_kl.plot(
        steps,
        losses["kl"],
        color="tab:orange",
        label="KL",
        linewidth=2,
    )

    ax_recon.set_xlabel("Transition Step")
    ax_recon.set_ylabel("Raw action recon MSE per dim", color="tab:blue")
    ax_kl.set_ylabel("KL loss", color="tab:orange")
    ax_recon.tick_params(axis="y", labelcolor="tab:blue")
    ax_kl.tick_params(axis="y", labelcolor="tab:orange")
    ax_recon.grid(True, alpha=0.25)
    ax_recon.set_title(title)

    lines = [line_recon, line_kl]
    ax_recon.legend(lines, [line.get_label() for line in lines], loc="upper right")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def save_epoch_loss_plot(policy, dataset, val_episodes, epoch_idx, output_dir, batch_size, seed):
    if policy.actor_vae is None or not val_episodes:
        return None

    rng = np.random.default_rng(seed + int(epoch_idx))
    episode_list_idx = int(rng.integers(len(val_episodes)))
    episode_indices = val_episodes[episode_list_idx]
    episode_info = describe_episode(dataset, episode_indices)
    losses = compute_episode_losses(policy, dataset, episode_indices, batch_size=batch_size)

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(
        output_dir,
        f"epoch_{int(epoch_idx):04d}_episode_{episode_list_idx:04d}.png",
    )
    title = (
        f"Epoch {int(epoch_idx)} validation episode {episode_list_idx} "
        f"| dataset {episode_info['segment_id']}:{episode_info['dataset_name']} "
        f"| file episode {episode_info['episode_id_in_dataset']} "
        f"| frames {episode_info['raw_start']}..{episode_info['raw_end']} "
        f"| n={len(episode_indices)}, recon={losses['recon_mean'].mean():.4f}, "
        f"raw_recon={losses['raw_recon_mean'].mean():.6f}, "
        f"kl={losses['kl'].mean():.4f}"
    )
    plot_episode_losses(losses, output_path, title)
    return {
        "path": output_path,
        "episode": episode_list_idx,
        "dataset": episode_info["dataset_name"],
        "dataset_path": episode_info["dataset_path"],
        "dataset_episode": episode_info["episode_id_in_dataset"],
        "raw_start": episode_info["raw_start"],
        "raw_end": episode_info["raw_end"],
        "transitions": len(episode_indices),
        "recon": float(losses["recon_mean"].mean()),
        "raw_recon": float(losses["raw_recon_mean"].mean()),
        "raw_recon_dim_mse": losses["raw_recon_dim_mse"].astype(float).tolist(),
        "raw_recon_dim_rmse": losses["raw_recon_dim_rmse"].astype(float).tolist(),
        "kl": float(losses["kl"].mean()),
    }
