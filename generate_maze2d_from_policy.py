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

FIXED_START_MODE = "cycle"

# Hard-code rollout start positions here.
# Coordinates are continuous environment positions.
ENV_FIXED_STARTS = {
    "maze2d-large": [
        np.array([1.0, 1.0], dtype=np.float32),
        np.array([1.0, 8.0], dtype=np.float32),
        np.array([5.0, 4.0], dtype=np.float32),
        np.array([7.0, 1.0], dtype=np.float32),
    ],
    "antmaze-large": [
            (1, 1),
            (4, 1),
            (1, 6),
            (1, 10),
    ],
}

# Success region and randomized keep-step options.
SUCCESS_RADIUS = 0.5
#SUCCESS_KEEP_STEP_CHOICES = [3, 5]


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


def get_fixed_start_locations(env_name):
    if "maze2d-large" in env_name:
        return ENV_FIXED_STARTS["maze2d-large"]
    if "antmaze-large" in env_name:
        return ENV_FIXED_STARTS["antmaze-large"]
    return None


def get_rollout_obs(env):
    if hasattr(env, "_get_obs"):
        return env._get_obs()
    if hasattr(env, "wrapped_env") and hasattr(env.wrapped_env, "_get_obs"):
        return env.wrapped_env._get_obs()
    if hasattr(env, "_wrapped_env") and hasattr(env._wrapped_env, "_get_obs"):
        return env._wrapped_env._get_obs()
    raise AttributeError("Could not recover observation after manually setting the start location.")


def get_env_goal(env):
    # Keep the original maze2d logic unchanged.
    if "maze2d" in env.spec.id:
        if hasattr(env, "get_target"):
            return env.get_target()
        return getattr(env, "_target", np.zeros(2, dtype=np.float32))

    # Add antmaze-specific goal handling without touching maze2d behavior.
    base_env = env.unwrapped
    if hasattr(base_env, "target_goal") and base_env.target_goal is not None:
        return np.array(base_env.target_goal[:2], dtype=np.float32)
    if hasattr(base_env, "_goal") and base_env._goal is not None:
        return np.array(base_env._goal[:2], dtype=np.float32)
    if hasattr(base_env, "get_target"):
        return np.array(base_env.get_target()[:2], dtype=np.float32)
    return getattr(base_env, "_target", np.zeros(2, dtype=np.float32))


def sample_keep_steps():
    return int(5)


def reset_env(env, start_locations=None, start_idx=0, start_mode="cycle"):
    if not start_locations:
        return env.reset(), start_idx

    if start_mode == "random":
        start_idx = np.random.randint(len(start_locations))
    else:
        start_idx = start_idx % len(start_locations)

    start_location = start_locations[start_idx]
    if "maze2d" in env.spec.id:
        # Gym's OrderEnforcing wrapper requires a standard reset() before step().
        env.reset()
        obs = env.reset_to_location(start_location)
    elif "antmaze" in env.spec.id:
        env.reset()
        base_env = env.unwrapped
        start_xy = base_env._rowcol_to_xy(start_location, add_random_noise=False)
        env.set_xy(start_xy)
        obs = get_rollout_obs(env)
    else:
        obs = env.reset()

    if start_mode == "cycle":
        next_start_idx = (start_idx + 1) % len(start_locations)
    else:
        next_start_idx = start_idx

    return obs, next_start_idx


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
    return env, norm_dataset, policy


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_name", default="maze2d-large-v1", type=str)
    parser.add_argument("--model_dir", default="./results/Exp0007/maze2d-large-v1", type=str)
    parser.add_argument("--model_name", default="model", type=str)
    parser.add_argument("--output", default="./generated/maze2d-large-v1-from-policy.hdf5", type=str)
    parser.add_argument("--num_samples", default=int(1e6), type=int)
    parser.add_argument("--seed", default=789, type=int)
    parser.add_argument("--device", default="cuda", type=str)
    parser.add_argument("--max_episode_steps", default=None, type=int)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--add_noise", action="store_true")
    parser.add_argument("--noise_std", default=None, type=float)
    parser.add_argument(
        "--goal_hold_noise_std",
        default=0.05,
        type=float,
        help="Gaussian action noise used during the near-goal hold phase.",
    )
    parser.add_argument("--vae_lr", default=2e-4, type=float)
    parser.add_argument("--actor_lr", default=2e-4, type=float)
    parser.add_argument("--critic_lr", default=2e-4, type=float)
    parser.add_argument("--tau", default=0.005, type=float)
    parser.add_argument("--discount", default=0.99, type=float)
    parser.add_argument("--expectile", default=0.9, type=float)
    parser.add_argument("--kl_beta", default=1.0, type=float)
    parser.add_argument("--max_latent_action", default=0.675, type=float)
    parser.add_argument("--doubleq_min", default=1.0, type=float)
    args = parser.parse_args()

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)

    env, norm_dataset, policy = build_policy(args, args.env_name)
    env.seed(args.seed)
    env.action_space.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    start_locations = get_fixed_start_locations(args.env_name)

    max_episode_steps = args.max_episode_steps or getattr(env, "_max_episode_steps", 1000)

    data = reset_data()
    start_idx = 0
    obs, start_idx = reset_env(env, start_locations, start_idx, FIXED_START_MODE)
    episode_steps = 0
    success_hold_counter = 0
    success_keep_steps_target = None

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
                success_keep_steps_target = sample_keep_steps()
            success_hold_counter += 1
        else:
            success_hold_counter = 0
            success_keep_steps_target = None

        if (
            success_keep_steps_target is not None
            and success_hold_counter >= success_keep_steps_target
        ):
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
            print(f"collected {step_idx + 1} transitions")

        if done or timeout:
            obs, start_idx = reset_env(env, start_locations, start_idx, FIXED_START_MODE)
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


if __name__ == "__main__":
    main()
