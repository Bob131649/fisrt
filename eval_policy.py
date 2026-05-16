#!/usr/bin/env python3

import argparse
import os

import h5py

import d4rl
import gym
import numpy as np
import torch

import algos.algos_v2 as algos
from dataset.d4rl_dataset import D4rlDataset
from fixed_reset_wrapper import FixedResetWrapper

SUCCESS_RADIUS = 0.5


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


def get_env_goal(env):
    if "maze2d" in env.spec.id:
        if hasattr(env, "get_target"):
            return np.array(env.get_target()[:2], dtype=np.float32)
        return np.array(getattr(env, "_target", np.zeros(2))[:2], dtype=np.float32)

    base_env = env.unwrapped
    if hasattr(base_env, "target_goal") and base_env.target_goal is not None:
        return np.array(base_env.target_goal[:2], dtype=np.float32)
    if hasattr(base_env, "_goal") and base_env._goal is not None:
        return np.array(base_env._goal[:2], dtype=np.float32)
    if hasattr(base_env, "get_target"):
        return np.array(base_env.get_target()[:2], dtype=np.float32)
    return np.array(getattr(base_env, "_target", np.zeros(2))[:2], dtype=np.float32)


def get_env_start_xy(env):
    base_env = env.unwrapped
    if hasattr(base_env, "get_xy"):
        return np.array(base_env.get_xy()[:2], dtype=np.float32)
    if hasattr(base_env, "sim") and hasattr(base_env.sim, "data"):
        return np.array(base_env.sim.data.qpos[:2], dtype=np.float32)
    if hasattr(base_env, "physics") and hasattr(base_env.physics, "data"):
        return np.array(base_env.physics.data.qpos[:2], dtype=np.float32)
    return np.array([np.nan, np.nan], dtype=np.float32)


def is_success(env_name, reward, obs, goal):
    if "antmaze" in env_name:
        return reward >= 1.0
    goal_distance = np.linalg.norm(obs[:2] - goal[:2])
    if "maze2d" in env_name:
        return goal_distance <= SUCCESS_RADIUS or reward >= 0.95
    return goal_distance <= SUCCESS_RADIUS


def build_policy(args):
    raw_env = gym.make(args.env_name)
    state_dim = raw_env.observation_space.shape[0]
    action_dim = raw_env.action_space.shape[0]

    if args.dataset_path:
        if not os.path.isfile(args.dataset_path):
            raise FileNotFoundError(f"Dataset file not found: {args.dataset_path}")
        print(f"loading normalization dataset from: {args.dataset_path}")
        raw_dataset = load_hdf5_dataset(args.dataset_path)
        dataset = d4rl.qlearning_dataset(raw_env, dataset=raw_dataset)
    else:
        dataset = d4rl.qlearning_dataset(raw_env)
    if "antmaze" in args.env_name:
        dataset["rewards"] = dataset["rewards"] * 100
        min_v = 0
        max_v = 100
    else:
        dataset["rewards"] = dataset["rewards"] - dataset["rewards"].min()
        dataset["rewards"] = dataset["rewards"] / dataset["rewards"].max()
        min_v = dataset["rewards"].min() / (1 - args.discount)
        max_v = dataset["rewards"].max() / (1 - args.discount)

    norm_dataset = D4rlDataset(dataset, args.env_name)
    latent_dim = int(action_dim * 2)
    policy = algos.Latent(
        state_dim,
        action_dim,
        latent_dim,
        min_v,
        max_v,
        device=args.device,
        discount=args.discount,
        tau=args.tau,
        vae_lr=args.vae_lr,
        actor_lr=args.actor_lr,
        critic_lr=args.critic_lr,
        max_latent_action=args.max_latent_action,
        expectile=args.expectile,
        kl_beta=args.kl_beta,
        doubleq_min=args.doubleq_min,
        policy_mode=args.policy_mode,
    )
    policy.load(args.model_name, args.model_dir)
    policy.eval()
    policy.copy_bn_param()
    raw_env.close()
    return norm_dataset, policy


def build_env(args):
    base_env = gym.make(args.env_name)
    env = FixedResetWrapper(
        base_env,
        env_name=args.env_name,
        start_mode="cycle",
        start_noise_scale=args.start_noise_scale,
        goal_noise_scale=args.goal_noise_scale,
    )
    env.seed(args.seed)
    env.action_space.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    return env


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_name", default="maze2d-large-v1", type=str)
    parser.add_argument("--model_dir", required=True, type=str)
    parser.add_argument("--model_name", default="model", type=str)
    parser.add_argument("--dataset_path", default="", type=str)
    parser.add_argument("--episodes_per_start", default=5, type=int)
    parser.add_argument("--seed", default=789, type=int)
    parser.add_argument("--device", default="cuda", type=str)
    parser.add_argument("--policy_mode", default="vae", choices=["lapo", "vae", "vae_bc"], type=str)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--max_episode_steps", default=None, type=int)
    parser.add_argument("--start_noise_scale", default=0.1, type=float)
    parser.add_argument("--goal_noise_scale", default=0.0, type=float)
    parser.add_argument("--vae_lr", default=2e-4, type=float)
    parser.add_argument("--actor_lr", default=2e-4, type=float)
    parser.add_argument("--critic_lr", default=2e-4, type=float)
    parser.add_argument("--tau", default=0.005, type=float)
    parser.add_argument("--discount", default=0.99, type=float)
    parser.add_argument("--expectile", default=0.5, type=float)
    parser.add_argument("--kl_beta", default=1.0, type=float)
    parser.add_argument("--max_latent_action", default=0.675, type=float)
    parser.add_argument("--doubleq_min", default=1.0, type=float)
    args = parser.parse_args()

    env = build_env(args)
    norm_dataset, policy = build_policy(args)

    if not env.fixed_starts:
        raise ValueError("Wrapper did not provide any fixed starts for this env.")

    total_episodes = len(env.fixed_starts) * args.episodes_per_start
    max_episode_steps = args.max_episode_steps or getattr(env, "_max_episode_steps", 1000)

    returns = []
    successes = []
    per_start = {idx: [] for idx in range(len(env.fixed_starts))}

    for episode_idx in range(total_episodes):
        obs = env.reset()
        start_id = getattr(env, "last_reset_start_idx", None)
        start_xy = get_env_start_xy(env)
        goal_xy = get_env_goal(env)

        done = False
        step_count = 0
        episode_return = 0.0
        success = False

        print(
            f"episode {episode_idx}: start_id={start_id}, "
            f"start={start_xy.tolist()}, goal={goal_xy.tolist()}"
        )

        while not done and step_count < max_episode_steps:
            norm_obs = norm_dataset.normalize_state(np.array(obs))
            action, q1, q2 = policy.select_action(norm_obs)
            action = norm_dataset.unnormalize_action(action)
            obs, reward, done, _ = env.step(action)
            episode_return += reward
            step_count += 1

            if is_success(args.env_name, reward, obs, goal_xy):
                success = True

            if args.render:
                env.render()

        returns.append(episode_return)
        successes.append(float(success))
        if start_id is not None:
            per_start[start_id].append(float(success))

        print(
            f"episode {episode_idx} result: return={episode_return:.3f}, "
            f"success={success}, steps={step_count}"
        )

    print("---------------------------------------")
    print(
        f"overall: avg_return={np.mean(returns):.3f}, "
        f"success_rate={np.mean(successes):.3f}, episodes={total_episodes}"
    )
    for start_id, values in per_start.items():
        if values:
            print(
                f"start {start_id}: success_rate={np.mean(values):.3f}, "
                f"episodes={len(values)}"
            )
    print("---------------------------------------")

    env.close()


if __name__ == "__main__":
    main()
