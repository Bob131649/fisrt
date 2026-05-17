#!/usr/bin/env python3

import argparse
import json
import os

import d4rl
import gym
import h5py
import numpy as np
import torch

import algos.algos_v2 as algos
from dataset.d4rl_dataset import D4rlDataset
from fixed_reset_wrapper import FixedResetWrapper


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
        print(f"loading dataset from: {dataset_path}")
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


def build_policy(
    model_dir,
    model_name,
    env_name,
    ope_dataset,
    target_policy_dataset,
    variant,
    state_dim,
    action_dim,
    min_v,
    max_v,
):
    target_policy_dir = variant.get("target_policy_dir")
    target_policy_name = variant.get("target_policy_name", "model")
    target_policy_mode = variant.get("target_policy_mode", "lapo")
    device = variant.get("device", "cuda")
    discount = float(variant.get("discount", 0.99))
    tau = float(variant.get("tau", 0.005))
    critic_lr = float(variant.get("critic_lr", 2e-4))
    max_latent_action = float(variant.get("max_latent_action", 0.675))
    doubleq_min = float(variant.get("doubleq_min", 1.0))

    if not target_policy_dir:
        raise ValueError("variant.json is missing target_policy_dir.")

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
    policy.load(model_name, model_dir)
    policy.eval()
    return policy


def build_query_state(raw_observations, x, y):
    xy = raw_observations[:, :2]
    distances = np.sum((xy - np.array([x, y], dtype=np.float32)) ** 2, axis=1)
    template_idx = int(np.argmin(distances))
    template_state = np.array(raw_observations[template_idx], dtype=np.float32).copy()
    template_state[0] = x
    template_state[1] = y
    return template_state, template_idx, float(np.sqrt(distances[template_idx]))


def set_env_pose(env, x, y):
    base_env = env.unwrapped
    target_xy = np.array([x, y], dtype=np.float32)

    if hasattr(env, "set_xy"):
        env.set_xy(target_xy)
        return
    if hasattr(base_env, "set_xy"):
        base_env.set_xy(target_xy)
        return
    if hasattr(base_env, "set_state") and hasattr(base_env, "sim"):
        qpos = np.array(base_env.sim.data.qpos).copy()
        qvel = np.array(base_env.sim.data.qvel).copy()
        qpos[:2] = target_xy
        base_env.set_state(qpos, qvel)
        return

    raise AttributeError("Environment does not provide a way to set pose to a query xy.")


def render_query_pose(env_name, x, y):
    base_env = gym.make(env_name)
    env = FixedResetWrapper(
        base_env,
        env_name=env_name,
        start_mode="cycle",
        start_noise_scale=0.0,
        goal_noise_scale=0.0,
    )
    env.reset()
    set_env_pose(env, x, y)
    env.render()
    return env


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ope_model_dir", required=True, type=str)
    parser.add_argument("--x", required=True, type=float)
    parser.add_argument("--y", required=True, type=float)
    parser.add_argument("--model_name", default="model", type=str)
    parser.add_argument("--env_name", default="", type=str)
    parser.add_argument("--ope_dataset_path", default="", type=str)
    parser.add_argument("--target_policy_dataset_path", default="", type=str)
    parser.add_argument("--render", action="store_true")
    args = parser.parse_args()

    variant = load_variant(args.ope_model_dir)
    env_name = resolve_arg(args, variant, "env_name")
    if not env_name:
        raise ValueError("Could not resolve env_name from args or variant.json.")

    ope_dataset_path = resolve_arg(args, variant, "ope_dataset_path")
    if not ope_dataset_path:
        ope_dataset_path = resolve_arg(args, variant, "dataset_path", "")
    target_policy_dataset_path = (
        args.target_policy_dataset_path
        or resolve_arg(args, variant, "target_policy_dataset_path")
        or ope_dataset_path
    )

    discount = float(resolve_arg(args, variant, "discount", 0.99))

    env = gym.make(env_name)
    ope_data = build_qlearning_dataset(env, ope_dataset_path)
    raw_observations = np.array(ope_data["observations"], dtype=np.float32)
    min_v, max_v = preprocess_dataset_rewards(ope_data, env_name, discount)
    ope_dataset = D4rlDataset(ope_data, env_name)

    if target_policy_dataset_path == ope_dataset_path:
        target_policy_dataset = ope_dataset
    else:
        target_policy_data = build_qlearning_dataset(env, target_policy_dataset_path)
        target_policy_dataset = D4rlDataset(target_policy_data, env_name)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    policy = build_policy(
        model_dir=args.ope_model_dir,
        model_name=args.model_name,
        env_name=env_name,
        ope_dataset=ope_dataset,
        target_policy_dataset=target_policy_dataset,
        variant=variant,
        state_dim=state_dim,
        action_dim=action_dim,
        min_v=min_v,
        max_v=max_v,
    )

    query_state_raw, template_idx, template_dist = build_query_state(
        raw_observations, args.x, args.y
    )
    query_state_norm = ope_dataset.normalize_state(query_state_raw.reshape(1, -1))
    state_tensor = torch.FloatTensor(query_state_norm).to(policy.device)

    with torch.no_grad():
        value = policy.estimate_value(state_tensor).item()
        action_tensor = policy.target_policy.select_action_tensor_ope(state_tensor)
        q1, q2 = policy.critic(state_tensor, action_tensor)
        q_value = policy.get_min_q(q1, q2).item()

    print(f"env_name: {env_name}")
    print(f"query_xy: [{args.x}, {args.y}]")
    print(f"nearest_template_index: {template_idx}")
    print(f"nearest_template_xy_distance: {template_dist:.6f}")
    print(f"predicted_V: {value:.6f}")
    print(f"predicted_Q_pi: {q_value:.6f}")

    render_env = None
    if args.render:
        render_env = render_query_pose(env_name, args.x, args.y)
        input("render opened; press Enter to close...")
        render_env.close()

    env.close()


if __name__ == "__main__":
    main()
