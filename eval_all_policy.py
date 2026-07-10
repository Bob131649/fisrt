#!/usr/bin/env python3

import argparse
import csv
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

import algos.vision_algos_guide as algos
from replay_buffer.vision_numpy_buffer import WaterPipeDataset


def default_franka_paths():
    expert_paths = [
        "./datasets/real_dataset/expert/franka_pickplace_ep20_20260629_152135_add_rewarding20.hdf5",
        "./datasets/real_dataset/expert/franka_pickplace_ep30_20260626_162422_add_rewarding20.hdf5",
        "./datasets/real_dataset/expert/franka_pickplace_ep31_20260625_112544_add_rewarding20.hdf5",
        "./datasets/real_dataset/expert/franka_pickplace_ep39_20260626_143912_add_rewarding20.hdf5",
    ]
    random_paths = [
        "./datasets/real_dataset/random/franka_random_20260701_170633_step1909.hdf5",
        "./datasets/real_dataset/random/franka_random_20260702_154715_step2056.hdf5",
        "./datasets/real_dataset/random/franka_random_20260702_155326_step1535.hdf5",
        "./datasets/real_dataset/random/franka_random_20260702_155755_step1499.hdf5",
        "./datasets/real_dataset/random/franka_random_20260703_142557_step2013.hdf5",
    ]
    return expert_paths, random_paths


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def load_normalization_stats(path):
    stats = load_json(path)
    return {
        "s_mean": np.asarray(stats["state_mean"], dtype=np.float32),
        "s_std": np.asarray(stats["state_std"], dtype=np.float32),
        "a_mean": np.asarray(stats["action_mean"], dtype=np.float32),
        "a_std": np.asarray(stats["action_std"], dtype=np.float32),
    }


def get_real_dataset(dataset_path):
    import h5py

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


def resolve_device(device):
    if device.startswith("cuda") and not torch.cuda.is_available():
        print(f"cuda is not available, falling back from {device} to cpu")
        return "cpu"
    return device


def build_policy(args, variant, state_dim, action_dim):
    latent_dim = max(4, int(action_dim * 2.0))
    policy = algos.Latent(
        state_dim,
        action_dim,
        latent_dim,
        min_v=0.0,
        max_v=100.0,
        device=args.device,
        discount=float(variant.get("discount", args.discount)),
        tau=float(variant.get("tau", args.tau)),
        vae_lr=float(variant.get("vae_lr", args.vae_lr)),
        actor_lr=float(variant.get("actor_lr", args.actor_lr)),
        critic_lr=float(variant.get("critic_lr", args.critic_lr)),
        max_latent_action=float(variant.get("max_latent_action", args.max_latent_action)),
        expectile=float(variant.get("expectile", args.expectile)),
        kl_beta=float(variant.get("kl_beta", args.kl_beta)),
        doubleq_min=float(variant.get("doubleq_min", args.doubleq_min)),
    )
    return policy


def load_eval_checkpoint(policy, model_dir, model_name, device):
    model_dir = Path(model_dir)
    map_location = torch.device(device)
    modules = [
        ("critic", policy.critic, True),
        ("critic_target", policy.critic_target, False),
        ("actor", policy.actor, False),
        ("actor_target", policy.actor_target, False),
        ("actor_vae", policy.actor_vae, True),
        ("actor_vae_target", policy.actor_vae_target, False),
    ]
    loaded = []
    for suffix, module, required in modules:
        path = model_dir / f"{model_name}_{suffix}.pth"
        if not path.exists():
            if required:
                raise FileNotFoundError(f"required checkpoint file not found: {path}")
            continue
        module.load_state_dict(torch.load(path, map_location=map_location))
        loaded.append(path.name)
    print("loaded:", ", ".join(loaded))


def make_pipe(env_name, paths, state_dim, action_dim, device, stats):
    datasets = [get_real_dataset(path) for path in paths]
    pipe = WaterPipeDataset(env_name, state_dim, action_dim, device)
    pipe.load(datasets)
    pipe.apply_stats(stats)
    return pipe


def build_eval_pipe(args, stats):
    expert_paths, random_paths = default_franka_paths()
    if args.expert_paths:
        expert_paths = args.expert_paths
    if args.random_paths:
        random_paths = args.random_paths

    pipes = []
    if not args.random_only:
        pipes.append(make_pipe(args.env_name, expert_paths, args.state_dim, args.action_dim, args.device, stats))
    if not args.expert_only:
        pipes.append(make_pipe(args.env_name, random_paths, args.state_dim, args.action_dim, args.device, stats))
    if not pipes:
        raise ValueError("no dataset selected")
    if len(pipes) == 1:
        return pipes[0]
    return WaterPipeDataset.concat(pipes).apply_stats(stats, normalize=False)


def batch_to_device_eval(pipe, batch):
    state, action, next_state, reward, not_done, next_action = batch
    state = {
        "proprio": state["proprio"].to(pipe.device, non_blocking=True),
        "image": pipe._prepare_image_batch(state["image"].to(pipe.device, non_blocking=True)),
    }
    next_state = {
        "proprio": next_state["proprio"].to(pipe.device, non_blocking=True),
        "image": pipe._prepare_image_batch(next_state["image"].to(pipe.device, non_blocking=True)),
    }
    return (
        state,
        action.to(pipe.device, non_blocking=True),
        next_state,
        reward.to(pipe.device, non_blocking=True).view(-1, 1),
        not_done.to(pipe.device, non_blocking=True).view(-1, 1),
        next_action.to(pipe.device, non_blocking=True),
    )


def make_eval_loader(pipe, batch_size, max_samples, seed, workers):
    dataset = pipe
    if max_samples and max_samples < len(pipe):
        rng = np.random.default_rng(seed)
        indices = rng.choice(len(pipe), size=max_samples, replace=False)
        dataset = Subset(pipe, indices.tolist())
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=True,
        drop_last=False,
    )


def reduce_q(q1, q2, mode):
    if mode == "min":
        return torch.min(q1, q2)
    if mode == "max":
        return torch.max(q1, q2)
    if mode == "q1":
        return q1
    if mode == "q2":
        return q2
    return 0.5 * (q1 + q2)


def evaluate(policy, pipe, loader, args):
    rows = []
    recon_batch = []
    was_training = policy.training
    policy.eval()
    with torch.no_grad():
        for batch_idx, cpu_batch in enumerate(loader):
            state, action, _, _, _, _ = batch_to_device_eval(pipe, cpu_batch)
            raw_state = pipe.unnormalize_state(state)["proprio"]

            recon_action, _, _ = policy.actor_vae(state, action)
            recon_per_sample = F.mse_loss(recon_action, action, reduction="none").mean(dim=1)
            recon_sum_per_sample = F.mse_loss(recon_action, action, reduction="none").sum(dim=1)

            if args.value_action == "policy":
                latent = policy.actor(state)
                value_action = policy.actor_vae_target.decode(state, z=latent)
            elif args.value_action == "recon":
                value_action = recon_action
            else:
                value_action = action

            q1, q2 = policy.critic(state, value_action)
            q_value = reduce_q(q1, q2, args.q_reduce).squeeze(1)

            xyz = raw_state[:, :3].detach().cpu().numpy()
            q_np = q_value.detach().cpu().numpy()
            recon_np = recon_per_sample.detach().cpu().numpy()
            recon_sum_np = recon_sum_per_sample.detach().cpu().numpy()

            for i in range(xyz.shape[0]):
                rows.append((xyz[i, 0], xyz[i, 1], xyz[i, 2], q_np[i], recon_np[i], recon_sum_np[i]))
            recon_batch.append(
                (
                    batch_idx,
                    float(np.mean(recon_np)),
                    float(np.mean(recon_sum_np)),
                    float(np.mean(q_np)),
                )
            )
    if was_training:
        policy.train()
    return rows, recon_batch


def save_heatmap(rows, path, title):
    arr = np.asarray(rows, dtype=np.float32)
    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection="3d")
    sc = ax.scatter(
        arr[:, 0],
        arr[:, 1],
        arr[:, 2],
        c=arr[:, 3],
        cmap="viridis",
        s=8,
        alpha=0.62,
    )
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_title(title)
    fig.colorbar(sc, ax=ax, label="critic Q")
    fig.tight_layout()
    fig.savefig(path, dpi=250)
    plt.close(fig)


def save_recon_curve(recon_batch, path):
    arr = np.asarray(recon_batch, dtype=np.float32)
    fig, ax1 = plt.subplots(figsize=(10, 4))
    ax1.plot(arr[:, 0], arr[:, 1], label="recon mse per dim", linewidth=2)
    ax1.plot(arr[:, 0], arr[:, 2], label="recon mse sum", linewidth=1.5, alpha=0.75)
    ax1.set_xlabel("Eval Batch")
    ax1.set_ylabel("Recon Loss")
    ax1.grid(True, alpha=0.25)
    ax1.legend(loc="best")
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def write_csv(rows, recon_batch, output_dir):
    with open(output_dir / "eval_all_policy_points.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["x", "y", "z", "q", "recon_mse", "recon_mse_sum"])
        writer.writerows(rows)
    with open(output_dir / "eval_all_policy_recon_curve.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["batch", "recon_mse", "recon_mse_sum", "q_mean"])
        writer.writerows(recon_batch)


def print_stats(rows, recon_batch):
    arr = np.asarray(rows, dtype=np.float32)
    curve = np.asarray(recon_batch, dtype=np.float32)
    for name, values in [
        ("q", arr[:, 3]),
        ("recon_mse", arr[:, 4]),
        ("recon_mse_sum", arr[:, 5]),
    ]:
        print(
            f"{name}: mean={values.mean():.6f}, std={values.std():.6f}, "
            f"min={values.min():.6f}, max={values.max():.6f}"
        )
    print(f"evaluated points: {len(rows)}, batches: {len(recon_batch)}")
    print(f"last batch recon_mse={curve[-1, 1]:.6f}, q_mean={curve[-1, 3]:.6f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ExpID", default=2, type=int)
    parser.add_argument("--env_name", default="franka")
    parser.add_argument("--log_dir", default="./results/", type=str)
    parser.add_argument("--model_name", default="model", type=str)
    parser.add_argument("--output_dir", default=None, type=str)
    parser.add_argument("--device", default="cuda:0", type=str)
    parser.add_argument("--seed", default=789, type=int)
    parser.add_argument("--batch_size", default=256, type=int)
    parser.add_argument("--max_samples", default=4096, type=int)
    parser.add_argument("--loader_workers", default=0, type=int)
    parser.add_argument("--value_action", default="dataset", choices=["dataset", "policy", "recon"])
    parser.add_argument("--q_reduce", default="min", choices=["min", "mean", "max", "q1", "q2"])
    parser.add_argument("--expert_only", action="store_true")
    parser.add_argument("--random_only", action="store_true")
    parser.add_argument("--expert_paths", nargs="*", default=None)
    parser.add_argument("--random_paths", nargs="*", default=None)

    parser.add_argument("--state_dim", default=8, type=int)
    parser.add_argument("--action_dim", default=8, type=int)
    parser.add_argument("--vae_lr", default=2e-4, type=float)
    parser.add_argument("--actor_lr", default=2e-4, type=float)
    parser.add_argument("--critic_lr", default=2e-4, type=float)
    parser.add_argument("--tau", default=0.005, type=float)
    parser.add_argument("--discount", default=0.99, type=float)
    parser.add_argument("--expectile", default=0.85, type=float)
    parser.add_argument("--kl_beta", default=1.0, type=float)
    parser.add_argument("--max_latent_action", default=0.675, type=float)
    parser.add_argument("--doubleq_min", default=1.0, type=float)
    args = parser.parse_args()

    args.device = resolve_device(args.device)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    model_dir = Path(args.log_dir) / f"Exp{args.ExpID:04d}" / args.env_name
    variant_path = model_dir / "variant.json"
    stats_path = model_dir / "normalization_stats.json"
    if not model_dir.exists():
        raise FileNotFoundError(f"model dir not found: {model_dir}")
    if not stats_path.exists():
        raise FileNotFoundError(f"normalization stats not found: {stats_path}")

    variant = load_json(variant_path) if variant_path.exists() else {}
    stats = load_normalization_stats(stats_path)
    args.state_dim = int(variant.get("state_dim", args.state_dim))
    args.action_dim = int(variant.get("action_dim", args.action_dim))

    output_dir = Path(args.output_dir) if args.output_dir else model_dir / "eval_all_policy"
    output_dir.mkdir(parents=True, exist_ok=True)

    pipe = build_eval_pipe(args, stats)
    loader = make_eval_loader(pipe, args.batch_size, args.max_samples, args.seed, args.loader_workers)
    policy = build_policy(args, variant, args.state_dim, args.action_dim)
    load_eval_checkpoint(policy, model_dir, args.model_name, args.device)

    rows, recon_batch = evaluate(policy, pipe, loader, args)
    heatmap_path = output_dir / f"eval_all_policy_3d_q_{args.value_action}_{args.q_reduce}.png"
    recon_path = output_dir / "eval_all_policy_recon_curve.png"
    save_heatmap(rows, heatmap_path, f"Critic Q over XYZ ({args.value_action} action, {args.q_reduce})")
    save_recon_curve(recon_batch, recon_path)
    write_csv(rows, recon_batch, output_dir)
    print_stats(rows, recon_batch)
    print(f"saved 3d heatmap: {heatmap_path}")
    print(f"saved recon curve: {recon_path}")
    print(f"saved csv dir: {output_dir}")


if __name__ == "__main__":
    main()
