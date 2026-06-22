#!/usr/bin/env python3

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data._utils.collate import default_collate

import algos.algos_v2_vision as algos
from dataset.franka_dataset import FrankaImageDataset


def str2bool(value):
    if isinstance(value, bool):
        return value
    value = value.lower()
    if value in ("yes", "true", "t", "1", "y"):
        return True
    if value in ("no", "false", "f", "0", "n"):
        return False
    raise argparse.ArgumentTypeError("Boolean value expected.")


def load_json_if_exists(path):
    if os.path.isfile(path):
        with open(path, "r") as file:
            return json.load(file)
    return {}


def apply_normalization_stats(dataset, stats):
    if not stats:
        return

    dataset.state_mean = np.asarray(stats["state_mean"], dtype=np.float32)
    dataset.state_std = np.asarray(stats["state_std"], dtype=np.float32)
    dataset.action_mean = np.asarray(stats["action_mean"], dtype=np.float32)
    dataset.action_std = np.asarray(stats["action_std"], dtype=np.float32)

    if dataset.normalize_proprio:
        dataset.states = dataset.normalize_state(dataset.raw_states)
        dataset.next_states = dataset.normalize_state(dataset.raw_next_states)
    else:
        dataset.states = dataset.raw_states.astype(np.float32)
        dataset.next_states = dataset.raw_next_states.astype(np.float32)

    raw_actions = getattr(dataset, "raw_actions", dataset.actions)
    if dataset.normalize_action_flag:
        dataset.actions = dataset.normalize_action(raw_actions)
    else:
        dataset.actions = raw_actions.astype(np.float32)

    dataset.state_dim = dataset.states.shape[1]
    dataset.action_dim = dataset.actions.shape[1]


def apply_top_half_image_mask_for_eval(dataset, enabled=False):
    """Small eval-only test hook: mask the top half of image observations."""
    if not enabled:
        return dataset

    original_get_item = dataset.get_item

    def get_item_with_top_half_mask(idx, augment=True):
        sample = original_get_item(idx, augment=augment)
        for key in ("image", "next_image"):
            if key not in sample or not torch.is_tensor(sample[key]):
                continue
            image = sample[key].clone()
            height = image.shape[-2]
            image[..., : height // 2, :] = 0
            sample[key] = image
        return sample

    dataset.get_item = get_item_with_top_half_mask
    print("eval image mask test enabled: zeroed top 50% of image and next_image")
    return dataset


def build_episode_indices(dataset, min_len=2):
    episodes = []
    current = []
    prev_segment = None
    prev_raw_idx = None

    for idx in range(len(dataset)):
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
    """Keep threshold transitions as a chain: start at selected end frame next."""
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


def batch_to_device(batch, device):
    moved = {}
    for key, value in batch.items():
        if torch.is_tensor(value):
            moved[key] = value.to(device)
        else:
            moved[key] = value
    return moved


def compute_episode_losses(policy, dataset, episode_indices, batch_size, free_bits):
    recon_mean_values = []
    recon_sum_values = []
    kl_values = []

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

            kl_per_dim = -0.5 * (1 + log_var - mu.pow(2) - log_var.exp())
            if free_bits is not None and free_bits > 0:
                kl_per_dim = torch.maximum(
                    kl_per_dim,
                    torch.tensor(float(free_bits), device=kl_per_dim.device),
                )
            kl_values.extend(kl_per_dim.sum(dim=1).cpu().numpy().tolist())

    return {
        "recon_mean": np.asarray(recon_mean_values, dtype=np.float32),
        "recon_sum": np.asarray(recon_sum_values, dtype=np.float32),
        "kl": np.asarray(kl_values, dtype=np.float32),
    }


def plot_losses(losses, output_path, title):
    steps = np.arange(len(losses["kl"]))
    fig, ax_recon = plt.subplots(figsize=(12, 5))
    ax_kl = ax_recon.twinx()

    line_recon, = ax_recon.plot(
        steps,
        losses["recon_mean"],
        color="tab:blue",
        label="Recon MSE",
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
    ax_recon.set_ylabel("Recon MSE per action dim", color="tab:blue")
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", required=True, type=str)
    parser.add_argument("--checkpoint_name", default="model", type=str)
    parser.add_argument("--load_best", nargs="?", const=True, default=False, type=str2bool)
    parser.add_argument("--dataset_path", required=True, nargs="+", type=str)
    parser.add_argument("--output_path", default="", type=str)
    parser.add_argument("--episode_id", default=None, type=int)
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--batch_size", default=128, type=int)
    parser.add_argument("--free_bits", default=0.5, type=float)
    parser.add_argument("--device", default="cuda", type=str)
    parser.add_argument(
        "--mask_top_half_image",
        action="store_true",
        help="Eval-only test: zero the top 50% of image observations before the normal eval pipeline.",
    )
    parser.add_argument(
        "--chain_threshold_transitions",
        nargs="?",
        const=True,
        default=True,
        type=str2bool,
        help=(
            "Eval episode as chained threshold transitions: after start->end, skip raw frames before end. "
            "Use --chain_threshold_transitions false to keep every dataset transition start."
        ),
    )

    parser.add_argument("--image_size", nargs=2, default=None, type=int)
    parser.add_argument("--reward_scale", default=100.0, type=float)
    parser.add_argument("--success_reward", default=1.0, type=float)
    parser.add_argument("--normalize_proprio", nargs="?", const=True, default=None, type=str2bool)
    parser.add_argument("--normalize_action", nargs="?", const=True, default=None, type=str2bool)
    parser.add_argument("--robomimic_feature_dim", default=None, type=int)
    parser.add_argument("--robomimic_crop_shape", nargs=2, default=None, type=int)
    parser.add_argument("--robomimic_backbone_class", default=None, type=str)
    parser.add_argument("--robomimic_pool_class", default=None, type=str)
    parser.add_argument("--max_latent_action", default=None, type=float)
    parser.add_argument("--discount", default=None, type=float)
    parser.add_argument("--tau", default=None, type=float)
    parser.add_argument("--vae_lr", default=None, type=float)
    parser.add_argument("--actor_lr", default=None, type=float)
    parser.add_argument("--critic_lr", default=None, type=float)
    parser.add_argument("--obs_encoder_lr", default=None, type=float)
    parser.add_argument("--expectile", default=None, type=float)
    parser.add_argument("--kl_beta", default=None, type=float)
    parser.add_argument("--doubleq_min", default=None, type=float)
    args = parser.parse_args()

    model_dir = os.path.expanduser(args.model_dir)
    variant = load_json_if_exists(os.path.join(model_dir, "variant.json"))
    stats = load_json_if_exists(os.path.join(model_dir, "normalization_stats.json"))

    image_size = tuple(args.image_size or stats.get("image_size") or variant.get("image_size", [224, 224]))
    normalize_proprio = (
        args.normalize_proprio
        if args.normalize_proprio is not None
        else bool(stats.get("normalize_proprio", variant.get("normalize_proprio", True)))
    )
    normalize_action = (
        args.normalize_action
        if args.normalize_action is not None
        else bool(stats.get("normalize_action", variant.get("normalize_action", True)))
    )

    dataset = FrankaImageDataset(
        args.dataset_path,
        image_size=image_size,
        reward_scale=args.reward_scale,
        success_reward=args.success_reward,
        normalize_proprio=normalize_proprio,
        normalize_action=normalize_action,
    )
    apply_normalization_stats(dataset, stats)
    apply_top_half_image_mask_for_eval(dataset, enabled=args.mask_top_half_image)

    state_dim = int(dataset.state_dim)
    action_dim = int(dataset.action_dim)
    latent_dim = int(action_dim * 2)
    image_shape = (3, image_size[0], image_size[1])

    policy = algos.Latent(
        state_dim,
        action_dim,
        latent_dim,
        min_v=0.0,
        max_v=float(max(dataset.rewards.max(), args.reward_scale * args.success_reward, 1.0)),
        device=args.device,
        discount=float(args.discount if args.discount is not None else variant.get("discount", 0.99)),
        tau=float(args.tau if args.tau is not None else variant.get("tau", 0.005)),
        vae_lr=float(args.vae_lr if args.vae_lr is not None else variant.get("vae_lr", 2e-4)),
        actor_lr=float(args.actor_lr if args.actor_lr is not None else variant.get("actor_lr", 2e-4)),
        critic_lr=float(args.critic_lr if args.critic_lr is not None else variant.get("critic_lr", 2e-4)),
        obs_encoder_lr=args.obs_encoder_lr if args.obs_encoder_lr is not None else variant.get("obs_encoder_lr", None),
        max_latent_action=float(
            args.max_latent_action
            if args.max_latent_action is not None
            else variant.get("max_latent_action", 0.675)
        ),
        expectile=float(args.expectile if args.expectile is not None else variant.get("expectile", 0.9)),
        kl_beta=float(args.kl_beta if args.kl_beta is not None else variant.get("kl_beta", 1.0)),
        doubleq_min=float(args.doubleq_min if args.doubleq_min is not None else variant.get("doubleq_min", 1.0)),
        image_shape=image_shape,
        robomimic_feature_dim=int(
            args.robomimic_feature_dim
            if args.robomimic_feature_dim is not None
            else variant.get("robomimic_feature_dim", 256)
        ),
        robomimic_crop_shape=tuple(args.robomimic_crop_shape)
        if args.robomimic_crop_shape is not None
        else tuple(variant["robomimic_crop_shape"])
        if variant.get("robomimic_crop_shape") is not None
        else None,
        robomimic_backbone_class=args.robomimic_backbone_class
        or variant.get("robomimic_backbone_class", "ResNet18Conv"),
        robomimic_pool_class=args.robomimic_pool_class
        or variant.get("robomimic_pool_class", "SpatialSoftmax"),
        encoder_mode=variant.get("encoder_mode", "robomimic"),
        concat_mode=variant.get("concat_mode", "default"),
        zipper_backbone=variant.get("zipper_backbone", "resnet18"),
        zipper_normalize_image=bool(variant.get("zipper_normalize_image", True)),
        vae=bool(variant.get("vae", False)),
    )
    policy.load(args.checkpoint_name, model_dir, load_best=args.load_best)
    policy.eval()

    episodes = build_episode_indices(dataset)
    if not episodes:
        raise RuntimeError("No episode with at least 2 transitions was found in the dataset.")

    rng = np.random.default_rng(args.seed)
    episode_id = args.episode_id
    if episode_id is None:
        episode_id = int(rng.integers(len(episodes)))
    if not 0 <= episode_id < len(episodes):
        raise IndexError(f"episode_id must be in [0, {len(episodes) - 1}], got {episode_id}.")

    episode_indices = episodes[episode_id]
    original_episode_len = len(episode_indices)
    if args.chain_threshold_transitions:
        episode_indices = chain_threshold_episode_indices(dataset, episode_indices)
    losses = compute_episode_losses(
        policy,
        dataset,
        episode_indices,
        batch_size=args.batch_size,
        free_bits=args.free_bits,
    )

    output_path = args.output_path
    if not output_path:
        output_path = os.path.join(model_dir, f"episode_{episode_id}_vae_losses.png")
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    title = (
        f"Episode {episode_id} VAE losses "
        f"(n={len(episode_indices)}, recon={losses['recon_mean'].mean():.4f}, "
        f"kl={losses['kl'].mean():.4f}, chained={args.chain_threshold_transitions})"
    )
    plot_losses(losses, output_path, title)

    csv_path = os.path.splitext(output_path)[0] + ".csv"
    np.savetxt(
        csv_path,
        np.column_stack([losses["recon_mean"], losses["recon_sum"], losses["kl"]]),
        delimiter=",",
        header="recon_mean,recon_sum,kl",
        comments="",
    )

    print(f"episodes found: {len(episodes)}")
    print(
        f"selected episode: {episode_id}, transitions: {len(episode_indices)}"
        f" (before chain filter: {original_episode_len})"
    )
    print(f"chain threshold transitions: {args.chain_threshold_transitions}")
    print(f"mean recon mse: {losses['recon_mean'].mean():.6f}")
    print(f"mean recon sum: {losses['recon_sum'].mean():.6f}")
    print(f"mean kl: {losses['kl'].mean():.6f}")
    print(f"saved plot to: {output_path}")
    print(f"saved csv to: {csv_path}")


if __name__ == "__main__":
    main()
