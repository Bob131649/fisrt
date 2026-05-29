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


def build_norm_dataset(env, env_name, dataset_path, discount):
    dataset = build_qlearning_dataset(env, dataset_path)
    min_v, max_v = preprocess_dataset_rewards(dataset, env_name, discount)
    norm_dataset = D4rlDataset(dataset, env_name)
    raw_observations = np.array(dataset["observations"], dtype=np.float32)
    return dataset, raw_observations, norm_dataset, min_v, max_v


def load_ope_policy(
    args,
    env_name,
    ope_dataset,
    target_policy_dataset,
    min_v,
    max_v,
    state_dim,
    action_dim,
):
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
        target_policy_state_mean=target_policy_dataset.state_mean,
        target_policy_state_std=target_policy_dataset.state_std,
        target_policy_action_mean=target_policy_dataset.action_mean,
        target_policy_action_std=target_policy_dataset.action_std,
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


def collect_success_rollout_states(
    policy,
    ope_dataset,
    env_name,
    episodes_per_start,
    max_episode_steps=None,
):
    env = build_wrapped_env(
        env_name,
        start_mode="cycle",
        start_noise_scale=0.0,
        goal_noise_scale=0.0,
    )
    if not env.fixed_starts:
        raise ValueError("Wrapper did not provide any fixed starts for this env.")

    rollout_limit = max_episode_steps or getattr(env, "_max_episode_steps", 1000)
    total_episodes = len(env.fixed_starts) * episodes_per_start
    successful_observations = []
    successful_xy = []

    for _ in range(total_episodes):
        state, done = env.reset(), False
        goal_xy = get_env_goal(env)
        episode_states = []
        episode_xy = []
        step_count = 0
        success = False

        while not done and step_count < rollout_limit:
            episode_states.append(np.array(state, dtype=np.float32))
            episode_xy.append(np.array(state[:2], dtype=np.float32))
            norm_state = ope_dataset.normalize_state(np.array(state))
            action, _, _ = policy.select_action(norm_state)
            env_action = ope_dataset.unnormalize_action(action)
            state, reward, done, _ = env.step(env_action)
            step_count += 1
            if "antmaze" in env_name:
                success = success or (reward >= 1.0)
            else:
                goal_distance = np.linalg.norm(state[:2] - goal_xy[:2])
                success = success or (goal_distance <= 0.5 or reward >= 0.95)

        if success and episode_states:
            successful_observations.extend(episode_states)
            successful_xy.extend(episode_xy)

    env.close()
    if not successful_observations:
        return np.zeros((0, 2), dtype=np.float32), np.zeros((0, 2), dtype=np.float32)
    observations = np.array(successful_observations, dtype=np.float32)
    xy = np.array(successful_xy, dtype=np.float32)
    return observations, xy


def build_heatmap(xy, values, grid_size, agg="mean"):
    x = xy[:, 0]
    y = xy[:, 1]
    x_edges = np.linspace(np.min(x), np.max(x), grid_size + 1)
    y_edges = np.linspace(np.min(y), np.max(y), grid_size + 1)
    x_idx = np.clip(np.digitize(x, x_edges) - 1, 0, grid_size - 1)
    y_idx = np.clip(np.digitize(y, y_edges) - 1, 0, grid_size - 1)
    flat_idx = x_idx * grid_size + y_idx

    order = np.argsort(flat_idx)
    flat_sorted = flat_idx[order]
    values_sorted = values[order]
    unique_bins, start_idx = np.unique(flat_sorted, return_index=True)
    end_idx = np.append(start_idx[1:], values_sorted.shape[0])

    heatmap = np.full((grid_size, grid_size), np.nan, dtype=np.float64)
    counts = np.zeros((grid_size, grid_size), dtype=np.float64)
    for flat_bin, begin, end in zip(unique_bins, start_idx, end_idx):
        bucket = values_sorted[begin:end]
        if agg == "mean":
            agg_value = float(np.mean(bucket))
        elif agg == "max":
            agg_value = float(np.max(bucket))
        elif agg == "p90":
            agg_value = float(np.percentile(bucket, 90))
        elif agg == "p75":
            agg_value = float(np.percentile(bucket, 75))
        else:
            raise ValueError(f"Unsupported agg mode: {agg}")

        xi = flat_bin // grid_size
        yi = flat_bin % grid_size
        heatmap[xi, yi] = agg_value
        counts[xi, yi] = end - begin

    return heatmap.T, counts.T, x_edges, y_edges


def plot_heatmap(
    heatmap,
    counts,
    x_edges,
    y_edges,
    starts,
    goal,
    output_path,
    title,
    overlay_xy=None,
    vmin=None,
    vmax=None,
):
    fig, ax = plt.subplots(figsize=(9, 7))
    extent = [x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]]
    image = ax.imshow(
        heatmap,
        origin="lower",
        extent=extent,
        aspect="auto",
        cmap="viridis",
        vmin=vmin,
        vmax=vmax,
    )
    plt.colorbar(image, ax=ax, label="Predicted V(s)")

    if overlay_xy is not None and overlay_xy.size > 0:
        ax.scatter(
            overlay_xy[:, 0],
            overlay_xy[:, 1],
            c="white",
            s=2,
            alpha=0.15,
            linewidths=0,
            label="Success traj",
        )

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
    parser.add_argument(
        "--source",
        default="dataset",
        choices=["dataset", "rollout_success"],
        help="States used to build the heatmap.",
    )
    parser.add_argument(
        "--agg",
        default="mean",
        choices=["mean", "max", "p90", "p75"],
        help="Aggregation used inside each spatial bin.",
    )
    parser.add_argument("--rollout_episodes_per_start", default=20, type=int)
    parser.add_argument("--rollout_max_episode_steps", default=None, type=int)
    parser.add_argument("--overlay_success_traj", action="store_true")
    parser.add_argument("--vmin", default=None, type=float)
    parser.add_argument("--vmax", default=None, type=float)
    parser.add_argument("--vmin_quantile", default=0.05, type=float)
    parser.add_argument("--vmax_quantile", default=0.95, type=float)
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
    _, raw_observations, ope_dataset, min_v, max_v = build_norm_dataset(
        env, env_name, dataset_path, discount
    )
    target_policy_dataset_path = resolve_arg(args, args.variant, "target_policy_dataset_path", dataset_path)
    if target_policy_dataset_path == dataset_path:
        target_policy_dataset = ope_dataset
    else:
        _, _, target_policy_dataset, _, _ = build_norm_dataset(
            env, env_name, target_policy_dataset_path, discount
        )

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    policy = load_ope_policy(
        args,
        env_name,
        ope_dataset,
        target_policy_dataset,
        min_v,
        max_v,
        state_dim,
        action_dim,
    )

    overlay_xy = None
    if args.source == "rollout_success":
        raw_observations, success_xy = collect_success_rollout_states(
            policy,
            ope_dataset,
            env_name,
            args.rollout_episodes_per_start,
            args.rollout_max_episode_steps,
        )
        xy = raw_observations[:, :2]
        overlay_xy = success_xy
    else:
        xy = raw_observations[:, :2]
        if args.overlay_success_traj:
            _, overlay_xy = collect_success_rollout_states(
                policy,
                ope_dataset,
                env_name,
                args.rollout_episodes_per_start,
                args.rollout_max_episode_steps,
            )

    values = predict_dataset_values(policy, raw_observations, ope_dataset, args.batch_size)
    heatmap, counts, x_edges, y_edges = build_heatmap(xy, values, args.grid_size, agg=args.agg)
    starts, goal = collect_start_goal_points(env_name)
    valid_values = heatmap[~np.isnan(heatmap)]
    vmin = None
    vmax = None
    if args.vmin is not None or args.vmax is not None:
        vmin = args.vmin
        vmax = args.vmax
    elif valid_values.size > 0:
        if args.vmin_quantile is not None:
            vmin = float(np.quantile(valid_values, args.vmin_quantile))
        if args.vmax_quantile is not None:
            vmax = float(np.quantile(valid_values, args.vmax_quantile))

    plot_heatmap(
        heatmap=heatmap,
        counts=counts,
        x_edges=x_edges,
        y_edges=y_edges,
        starts=starts,
        goal=goal,
        output_path=output_path,
        title=f"OPE Value Heatmap: {env_name} ({args.source}, {args.agg})",
        overlay_xy=overlay_xy,
        vmin=vmin,
        vmax=vmax,
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
