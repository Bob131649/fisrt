#!/usr/bin/env python3

import argparse
import os

import d4rl
import gym
import h5py
import numpy as np
import torch

import algos.algos_v2 as algos
from dataset.d4rl_dataset import D4rlDataset
from fixed_reset_wrapper import FixedResetWrapper

FIXED_START_MODE = "cycle"
SUCCESS_RADIUS = 0.5


def reset_data():
    return {
        "observations": [],
        "actions": [],
        "next_observations": [],
        "rewards": [],
        "terminals": [],
        "timeouts": [],
        "infos/goal": [],
        "infos/qpos": [],
        "infos/qvel": [],
    }


def append_data(data, obs, action, next_obs, reward, done, timeout, goal, env_data):
    data["observations"].append(obs)
    data["actions"].append(action)
    data["next_observations"].append(next_obs)
    data["rewards"].append(reward)
    data["terminals"].append(done)
    data["timeouts"].append(timeout)
    data["infos/goal"].append(goal)
    data["infos/qpos"].append(env_data.qpos.ravel().copy())
    data["infos/qvel"].append(env_data.qvel.ravel().copy())


def npify(data):
    for key in data:
        if key in ["terminals", "timeouts"]:
            dtype = np.bool_
        else:
            dtype = np.float32
        data[key] = np.array(data[key], dtype=dtype)


def get_env_goal(env):
    if "maze2d" in env.spec.id:
        if hasattr(env, "get_target"):
            return env.get_target()
        return getattr(env, "_target", np.zeros(2, dtype=np.float32))

    base_env = env.unwrapped
    if hasattr(base_env, "target_goal") and base_env.target_goal is not None:
        return np.array(base_env.target_goal[:2], dtype=np.float32)
    if hasattr(base_env, "_goal") and base_env._goal is not None:
        return np.array(base_env._goal[:2], dtype=np.float32)
    if hasattr(base_env, "get_target"):
        return np.array(base_env.get_target()[:2], dtype=np.float32)
    return getattr(base_env, "_target", np.zeros(2, dtype=np.float32))


def build_policy(args, env_name):
    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    dataset = d4rl.qlearning_dataset(env)
    if "antmaze" in env_name:
        dataset["rewards"] = dataset["rewards"] * 100
        min_v = 0
        max_v = 100
    else:
        dataset["rewards"] = dataset["rewards"] - dataset["rewards"].min()
        dataset["rewards"] = dataset["rewards"] / dataset["rewards"].max()
        min_v = dataset["rewards"].min() / (1 - args.discount)
        max_v = dataset["rewards"].max() / (1 - args.discount)

    norm_dataset = D4rlDataset(dataset, env_name)
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
    )

    policy.load(args.model_name, args.model_dir)
    policy.eval()
    policy.copy_bn_param()

    if args.no_wrapper:
        wrapped_env = env
    else:
        wrapped_env = FixedResetWrapper(env, env_name=env_name, start_mode=args.start_mode)

    return wrapped_env, norm_dataset, policy


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_name", default="maze2d-large-v1", type=str)
    parser.add_argument("--model_dir", default="./results/Exp0007/maze2d-large-v1", type=str)
    parser.add_argument("--model_name", default="model", type=str)
    parser.add_argument("--output", default="./generated/maze2d-large-v1-point-test.hdf5", type=str)
    parser.add_argument("--num_samples", default=20000, type=int)
    parser.add_argument("--seed", default=789, type=int)
    parser.add_argument("--device", default="cuda", type=str)
    parser.add_argument("--max_episode_steps", default=None, type=int)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--add_noise", action="store_true")
    parser.add_argument("--noise_std", default=None, type=float)
    parser.add_argument("--goal_hold_noise_std", default=0.05, type=float)
    parser.add_argument("--vae_lr", default=2e-4, type=float)
    parser.add_argument("--actor_lr", default=2e-4, type=float)
    parser.add_argument("--critic_lr", default=2e-4, type=float)
    parser.add_argument("--tau", default=0.005, type=float)
    parser.add_argument("--discount", default=0.99, type=float)
    parser.add_argument("--expectile", default=0.9, type=float)
    parser.add_argument("--kl_beta", default=1.0, type=float)
    parser.add_argument("--max_latent_action", default=0.675, type=float)
    parser.add_argument("--doubleq_min", default=1.0, type=float)
    parser.add_argument("--start_mode", default=FIXED_START_MODE, choices=["cycle", "random"], type=str)
    parser.add_argument("--no_wrapper", action="store_true")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)

    env, norm_dataset, policy = build_policy(args, args.env_name)
    env.seed(args.seed)
    env.action_space.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    max_episode_steps = args.max_episode_steps or getattr(env, "_max_episode_steps", 1000)

    data = reset_data()
    obs = env.reset()
    episode_steps = 0
    success_hold_counter = 0
    success_keep_steps_target = None
    success_hits = 0

    for step_idx in range(args.num_samples):
        if success_keep_steps_target is not None:
            action = np.random.randn(*env.action_space.shape) * args.goal_hold_noise_std
        else:
            norm_obs = norm_dataset.normalize_state(np.array(obs))
            action, _, _ = policy.select_action(norm_obs)
            action = norm_dataset.unnormalize_action(action)

        if args.add_noise:
            action = action + np.random.randn(*action.shape) * args.noise_std

        action = np.clip(action, env.action_space.low, env.action_space.high)
        next_obs, reward, done, _ = env.step(action)

        episode_steps += 1
        timeout = episode_steps >= max_episode_steps

        goal = get_env_goal(env)
        position = next_obs[:2]
        goal_distance = np.linalg.norm(position - goal[:2])
        in_success_region = goal_distance <= SUCCESS_RADIUS

        if in_success_region:
            if success_keep_steps_target is None:
                success_keep_steps_target = 5
                success_hits += 1
            success_hold_counter += 1
        else:
            success_hold_counter = 0
            success_keep_steps_target = None

        if success_keep_steps_target is not None and success_hold_counter >= success_keep_steps_target:
            timeout = True

        append_data(
            data,
            obs,
            action,
            next_obs,
            reward,
            done,
            timeout,
            goal,
            env.sim.data,
        )

        if (step_idx + 1) % 10000 == 0:
            print(f"collected {step_idx + 1} transitions, success_hits={success_hits}")

        if done or timeout:
            obs = env.reset()
            episode_steps = 0
            success_hold_counter = 0
            success_keep_steps_target = None
        else:
            obs = next_obs

        if args.render:
            env.render()

    npify(data)
    with h5py.File(args.output, "w") as dataset_file:
        for key, value in data.items():
            dataset_file.create_dataset(key, data=value, compression="gzip")

    print(f"saved dataset to {args.output}")
    print(f"total success hits observed: {success_hits}")


if __name__ == "__main__":
    main()
