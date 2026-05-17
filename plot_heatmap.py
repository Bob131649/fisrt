#!/usr/bin/env python3

import argparse
import json
import os

import h5py
import matplotlib

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import numpy as np
import torch

import d4rl
import gym

import algos.algos_v2 as algos
from algos.ope import build_wrapped_env, get_env_goal, get_env_start_xy
from dataset.d4rl_dataset import D4rlDataset


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


def build_qlearning_dataset(env, dataset_path=""):
    if dataset_path:
        if not os.path.isfile(dataset_path):
            raise FileNotFoundError(f"Dataset file not found: {dataset_path}")
        print(f"loading visualization dataset from: {dataset_path}")
        raw_dataset = load_hdf5_dataset(dataset_path)
        return d4rl.qlearning_dataset(env, dataset=raw_dataset)
    return d4rl.qlearning_dataset(env)


def load_variant(model_dir):
    variant_path = os.path.join(model_dir, "variant.json")
    if not os.path.isfile(variant_path):
        return {}
    with open(variant_path, "r", encoding="utf-8") as f:
        return json.load(f)


def resolve_arg(args, variant, key, default=None):
    value = getattr(args, key, None)
    if value not in (None, ""):
        return value
    return variant.get(key, default)


def preprocess_dataset_rewards(dataset, env_name, discount):
    if "antmaze" in env_name:
        dataset["rewards"] = dataset["rewards"] * 100
        min_v = 0
        max_v = 100
    else:
        dataset["rewards"] = dataset["rewards"] - dataset["rewards"].min()
        dataset["rewards"] = dataset["rewards"] / dataset["rewards"].max()
        min_v = dataset["rewards"].min() / (1 - discount)
        max_v = dataset["rewards"].max() / (1 - discount)
    return min_v, max_v


def collect_start_goal_points(env_name):
    env = build_wrapped_env(
        env_name,
        start_mode="cycle",
        start_noise_scale=0.0,
        goal_noise_scale=0.0,
    )
    starts = []
    if env.fixed_starts:
        for _ in range(len(env.fixed_starts)):
            env.reset()
            starts.append(get_env_start_xy(env))
    goal = get_env_goal(env)
    env.close()
    return np.array(starts, dtype=np.float32) if starts else np.zeros((0, 2), dtype=np.float32), goal


def load_ope_policy(args, env_name, ope_dataset, min_v, max_v, state_dim, action_dim):
    target_policy_dir = resolve_arg(args, args.variant, "target_policy_dir")
    target_policy_name = resolve_arg(args, args.variant, "target_policy_name", "model")
    target_policy_mode = resolve_arg(args, args.variant, "target_policy_mode", "vae")
    device = resolve_arg(args, args.variant, "device", "cuda")
    discount = float(resolve_arg(args, args.variant, "discount", 0.99))
    tau = float(resolve_arg(args, args.variant, "tau", 0.005))
    critic_lr = float(resolve_arg(args, args.variant, "critic_lr", 2e-4))
    max_latent_action = float(resolve_arg(args, args.variant, "max_latent_action", 0.675))
    doubleq_min = float(resolve_arg(args, args.variant, "doubleq_min", 1.0))

    if not target_policy_dir:
        raise ValueError("Could not resolve target_policy_dir from args or variant.json.")

    latent_dim = int(action_dim * 2)
    policy = algos.Latent(
        state_dim,
        action_dim,
        latent_dim,
        min_v,
        max_v,
        device=device,
        discount=discount,
        tau=tau,
        critic_lr=critic_lr,
        max_latent_action=max_latent_action,
        doubleq_min=doubleq_min,
        target_policy_dir=target_policy_dir,
        target_policy_name=target_policy_name,
        target_policy_mode=target_policy_mode,
        ope_state_mean=ope_dataset.state_mean,
        ope_state_std=ope_dataset.state_std,
        ope_action_mean=ope_dataset.action_mean,
        ope_action_std=ope_dataset.action_std,
        target_policy_state_mean=ope_dataset.state_mean,
        target_policy_state_std=ope_dataset.state_std,
        target_policy_action_mean=ope_dataset.action_mean,
        target_policy_action_std=ope_dataset.action_std,
    )
    policy.load(args.model_name, args.ope_model_dir)
    policy.eval()
    return policy


def predict_dataset_values(policy, raw_observations, ope_dataset, batch_size):
    values = []
    for start in range(0, raw_observations.shape[0], batch_size):
        obs_batch = raw_observations[start : start + batch_size]
        norm_states = ope_dataset.normalize_state(obs_batch)
        states_tensor = torch.FloatTensor(norm_states).to(policy.device)
        with torch.no_grad():
            value_batch = policy.estimate_value(states_tensor).detach().cpu().numpy().reshape(-1)
        values.append(value_batch)
    return np.concatenate(values, axis=0)


def build_heatmap(xy, values, grid_size):
    x = xy[:, 0]
    y = xy[:, 1]
    x_edges = np.linspace(np.min(x), np.max(x), grid_size + 1)
    y_edges = np.linspace(np.min(y), np.max(y), grid_size + 1)

    value_sum, _, _ = np.histogram2d(x, y, bins=[x_edges, y_edges], weights=values)
    counts, _, _ = np.histogram2d(x, y, bins=[x_edges, y_edges])
    heatmap = np.divide(
        value_sum,
        counts,
        out=np.full_like(value_sum, np.nan, dtype=np.float64),
        where=counts > 0,
    )
    return heatmap.T, counts.T, x_edges, y_edges


def plot_heatmap(heatmap, counts, x_edges, y_edges, starts, goal, output_path, title):
    fig, ax = plt.subplots(figsize=(9, 7))
    extent = [x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]]
    image = ax.imshow(
        heatmap,
        origin="lower",
        extent=extent,
        aspect="auto",
        cmap="viridis",
    )
    plt.colorbar(image, ax=ax, label="Predicted V(s)")

    if starts.size > 0:
        ax.scatter(starts[:, 0], starts[:, 1], c="red", s=55, label="Fixed starts")
    if goal is not None:
        ax.scatter(goal[0], goal[1], c="white", edgecolors="black", s=85, marker="*", label="Goal")

    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ope_model_dir", required=True, type=str)
    parser.add_argument("--model_name", default="model", type=str)
    parser.add_argument("--env_name", default="", type=str)
    parser.add_argument("--ope_dataset_path", default="", type=str)
    parser.add_argument("--grid_size", default=400, type=int)
    parser.add_argument("--batch_size", default=4096, type=int)
    parser.add_argument("--output", default="", type=str)
    parser.add_argument("--save_npz", action="store_true")
    args = parser.parse_args()

    args.variant = load_variant(args.ope_model_dir)
    env_name = resolve_arg(args, args.variant, "env_name")
    if not env_name:
        raise ValueError("Could not resolve env_name from args or variant.json.")

    dataset_path = resolve_arg(args, args.variant, "ope_dataset_path")
    if not dataset_path:
        dataset_path = resolve_arg(args, args.variant, "dataset_path", "")

    discount = float(resolve_arg(args, args.variant, "discount", 0.99))
    output_path = args.output or os.path.join(args.ope_model_dir, "ope_value_heatmap.png")

    env = gym.make(env_name)
    dataset = build_qlearning_dataset(env, dataset_path)
    raw_observations = np.array(dataset["observations"], dtype=np.float32)
    min_v, max_v = preprocess_dataset_rewards(dataset, env_name, discount)
    ope_dataset = D4rlDataset(dataset, env_name)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    policy = load_ope_policy(args, env_name, ope_dataset, min_v, max_v, state_dim, action_dim)

    values = predict_dataset_values(policy, raw_observations, ope_dataset, args.batch_size)
    xy = raw_observations[:, :2]
    heatmap, counts, x_edges, y_edges = build_heatmap(xy, values, args.grid_size)
    starts, goal = collect_start_goal_points(env_name)

    plot_heatmap(
        heatmap=heatmap,
        counts=counts,
        x_edges=x_edges,
        y_edges=y_edges,
        starts=starts,
        goal=goal,
        output_path=output_path,
        title=f"OPE Value Heatmap: {env_name}",
    )
    print(f"saved heatmap to: {output_path}")

    if args.save_npz:
        npz_path = os.path.splitext(output_path)[0] + ".npz"
        np.savez_compressed(
            npz_path,
            heatmap=heatmap,
            counts=counts,
            x_edges=x_edges,
            y_edges=y_edges,
            xy=xy,
            values=values,
            starts=starts,
            goal=goal,
        )
        print(f"saved raw arrays to: {npz_path}")

    env.close()


if __name__ == "__main__":
    main()
