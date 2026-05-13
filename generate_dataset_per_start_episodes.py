#!/usr/bin/env python3

import argparse
import os

import d4rl
import gym
import h5py
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import numpy as np
import torch

import algos.algos_v2 as algos
from dataset.d4rl_dataset import D4rlDataset
from fixed_reset_wrapper import FixedResetWrapper

FIXED_START_MODE = "cycle"

# Success region and radomized keep-step options.
SUCCESS_RADIUS = 0.1
SUCCESS_HOLD_RADIUS = 0.5
#SUCCESS_KEEP_STEP_CHOICES = [3, 5]

# Per-start maximum episode-step thresholds for successful trajectories.
# Any successful episode with episode_steps >= threshold will be discarded.
# Use the stable global start_id mapping from FixedResetWrapper presets.
START_STEP_THRESHOLDS = {
    0: 420,
    1: 430,
    2: 280,
    3: 250,
}


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


def extend_data(dst, src):
    for key in dst:
        dst[key].extend(src[key])


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


def clone_and_npify(data):
    copied = {key: list(value) for key, value in data.items()}
    npify(copied)
    return copied


def clone_episode_data(data):
    return {key: list(value) for key, value in data.items()}


def save_hdf5_dataset(data, output_path):
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    np_data = clone_and_npify(data)
    with h5py.File(output_path, "w") as dataset_file:
        for key, value in np_data.items():
            dataset_file.create_dataset(key, data=value, compression="gzip")


def build_snapshot_output_path(output_path, success_count):
    base, ext = os.path.splitext(output_path)
    if not ext:
        ext = ".hdf5"
    return f"{base}_success{success_count}{ext}"


def build_dataset_from_successful_episodes(successful_episodes, num_starts, target_count):
    filtered = reset_data()
    per_start_counts = [0 for _ in range(num_starts)]

    for start_id, episode_data in successful_episodes:
        if start_id is None:
            continue
        if per_start_counts[start_id] >= target_count:
            continue
        extend_data(filtered, episode_data)
        per_start_counts[start_id] += 1
        if min(per_start_counts) >= target_count:
            break

    return filtered, per_start_counts


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


def is_success_by_env_reward(env_name, reward, next_obs, goal):
    if "antmaze" in env_name:
        return reward >= 1.0

    position = next_obs[:2]
    goal_distance = np.linalg.norm(position - goal[:2])
    return goal_distance <= SUCCESS_RADIUS


def get_env_start_xy(env):
    base_env = env.unwrapped
    if hasattr(base_env, "get_xy"):
        return np.array(base_env.get_xy()[:2], dtype=np.float32)
    if hasattr(base_env, "sim") and hasattr(base_env.sim, "data"):
        return np.array(base_env.sim.data.qpos[:2], dtype=np.float32)
    if hasattr(base_env, "physics") and hasattr(base_env.physics, "data"):
        return np.array(base_env.physics.data.qpos[:2], dtype=np.float32)
    return np.array([np.nan, np.nan], dtype=np.float32)


def get_env_position_xy(env, obs):
    del obs
    return get_env_start_xy(env)


def build_wrapped_env(env_name, start_mode="cycle", start_noise_scale=0.5, goal_noise_scale=0):
    base_env = gym.make(env_name)
    return FixedResetWrapper(
        base_env,
        env_name=env_name,
        start_mode=start_mode,
        start_noise_scale=start_noise_scale,
        goal_noise_scale=goal_noise_scale,
    )


def eval_policy(policy, replay_buffer, env_name, eval_episodes=10, plot=False, render=False):
    avg_reward = 0.0
    if plot:
        plt.clf()
    env = build_wrapped_env(env_name)

    for i in range(eval_episodes):
        state, done = env.reset(), False
        start_xy = get_env_start_xy(env)
        goal_xy = get_env_goal(env)
        print(
            f"eval episode {i}: start={start_xy.tolist()}, "
            f"goal={goal_xy.tolist()}, start_id={getattr(env, 'last_reset_start_idx', None)}"
        )
        q1_list, ep_reward = [], []
        while not done:
            state = replay_buffer.normalize_state(np.array(state))
            action, q1, q2 = policy.select_action(state)
            q1_list.append(q1)
            action = replay_buffer.unnormalize_action(action)
            state, reward, done, _ = env.step(action)
            avg_reward += reward
            ep_reward.append(reward)
            if render:
                env.render()
        print('   ---', np.mean(ep_reward), q1_list[0], q1_list[-1], np.mean(q1_list))
        print('---', state[0], state[1], len(ep_reward))

    avg_reward /= eval_episodes
    normalized_score = env.get_normalized_score(avg_reward)
    env.close()

    info = {'AverageReturn': avg_reward, 'NormReturn': normalized_score}
    print("---------------------------------------")
    print(f"Evaluation over {eval_episodes} episodes: {avg_reward:.3f}, {normalized_score:.3f}")
    print("---------------------------------------")
    return info


def sample_keep_steps(env_name):
    if "antmaze" in env_name:
        return 10
    return 20


def get_start_step_threshold(start_id):
    if start_id is None:
        return None
    return START_STEP_THRESHOLDS.get(start_id)


def format_start_progress(success_counts):
    return ", ".join(f"start{idx}={count}" for idx, count in enumerate(success_counts))


def update_active_starts(env, all_fixed_starts, success_counts, target_count):
    if success_counts is None or target_count is None:
        return None

    active_start_ids = [
        start_idx for start_idx, count in enumerate(success_counts) if count < target_count
    ]
    env.set_fixed_starts([all_fixed_starts[start_idx] for start_idx in active_start_ids])
    return active_start_ids


def get_active_start_ids(success_counts, target_count):
    if success_counts is None or target_count is None:
        return None
    return [
        start_idx for start_idx, count in enumerate(success_counts) if count < target_count
    ]


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
    env = FixedResetWrapper(
        env,
        env_name=env_name,
        start_mode=FIXED_START_MODE,
        start_noise_scale=0.5,
        goal_noise_scale=0,
    )
    return env, norm_dataset, policy


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_name", default="maze2d-large-v1", type=str)
    parser.add_argument("--model_dir", default="./results/Exp0007/maze2d-large-v1", type=str)
    parser.add_argument("--model_name", default="model", type=str)
    parser.add_argument("--output", default="./generated/maze2d-large-v1-from-policy.hdf5", type=str)
    parser.add_argument("--num_samples", default=int(1), type=int)
    parser.add_argument("--seed", default=789, type=int)
    parser.add_argument("--device", default="cuda", type=str)
    parser.add_argument("--max_episode_steps", default=None, type=int)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--eval_only", action="store_true")
    parser.add_argument("--eval_episodes", default=10, type=int)
    parser.add_argument("--plot", action="store_true")
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
    parser.add_argument(
        "--success_episodes_per_start",
        default=None,
        type=int,
        help="If set, only successful episodes are saved, and collection stops after each fixed start reaches this many successful episodes.",
    )
    parser.add_argument(
        "--success_episodes_per_start_list",
        nargs="+",
        type=int,
        default=None,
        help="Optional milestones such as 100 200 500. The script keeps collecting until the largest value and exports one hdf5 at each milestone.",
    )
    args = parser.parse_args()

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)

    milestone_counts = None
    final_success_target = args.success_episodes_per_start
    if args.success_episodes_per_start_list:
        milestone_counts = sorted(set(args.success_episodes_per_start_list))
        final_success_target = milestone_counts[-1]
    exported_milestones = set()

    env, norm_dataset, policy = build_policy(args, args.env_name)
    env.seed(args.seed)
    env.action_space.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.eval_only:
        eval_policy(
            policy,
            norm_dataset,
            args.env_name,
            eval_episodes=args.eval_episodes,
            plot=args.plot,
            render=args.render,
        )
        return

    max_episode_steps = args.max_episode_steps or getattr(env, "_max_episode_steps", 1000)

    data = reset_data()
    obs = env.reset()
    episode_steps = 0
    success_hold_counter = 0
    success_keep_steps_target = None
    success_anchor_xy = None
    episode_data = reset_data()
    success_counts = None
    all_fixed_starts = list(env.fixed_starts) if env.fixed_starts else None
    active_start_ids = None
    successful_episodes = []
    if env.fixed_starts and final_success_target is not None:
        success_counts = [0 for _ in env.fixed_starts]
        active_start_ids = update_active_starts(
            env, all_fixed_starts, success_counts, final_success_target
        )
    if success_counts is not None and active_start_ids:
        current_start_id = active_start_ids[env.last_reset_start_idx]
    else:
        current_start_id = None

    step_idx = 0
    while True:
        if final_success_target is None and step_idx >= args.num_samples:
            break
        if (
            final_success_target is not None
            and success_counts is not None
            and min(success_counts) >= final_success_target
        ):
            break

        if success_keep_steps_target is not None:
            action = np.random.randn(*env.action_space.shape) * args.goal_hold_noise_std
        else:
            norm_obs = norm_dataset.normalize_state(np.array(obs))
            action, _, _ = policy.select_action(norm_obs)
            action = norm_dataset.unnormalize_action(action)

        if args.add_noise:
            action = action + np.random.randn(*action.shape) * args.noise_std

        action = np.clip(action, env.action_space.low, env.action_space.high)
        next_obs, reward, env_done, _ = env.step(action)
        done = env_done

        episode_steps += 1
        timeout = episode_steps >= max_episode_steps

        goal = get_env_goal(env)
        current_xy = get_env_position_xy(env, next_obs)
        goal_distance = np.linalg.norm(current_xy[:2] - goal[:2])

        in_success_region = is_success_by_env_reward(
            args.env_name, reward, next_obs, goal
        )

        if in_success_region:
            if success_keep_steps_target is None:
                success_keep_steps_target = sample_keep_steps(args.env_name)
                success_hold_counter = 1
                success_anchor_xy = current_xy.copy()
                done = False
            else:
                success_hold_counter += 1
                done = False
        elif success_keep_steps_target is not None:
            success_hold_counter += 1
            done = False
        else:
            success_hold_counter = 0
            success_keep_steps_target = None

        if success_keep_steps_target is not None:
            reward = 1.0

        episode_success = False
        step_threshold = None
        if (
            success_keep_steps_target is not None
            and success_hold_counter >= success_keep_steps_target
        ):
            timeout = True
            done = True
            if success_anchor_xy is not None:
                final_bias = np.linalg.norm(current_xy[:2] - success_anchor_xy[:2])
                episode_success = final_bias <= SUCCESS_HOLD_RADIUS
            if episode_success:
                step_threshold = get_start_step_threshold(current_start_id)
                if step_threshold is not None and episode_steps >= step_threshold:
                    episode_success = False

        append_data(
            episode_data,
            obs,
            action,
            next_obs,
            reward,
            done,
            timeout,
            goal,
            env.sim.data,
        )

        step_idx += 1
        if (step_idx + 1) % 10000 == 0:
            print(f"collected {step_idx + 1} transitions")

        if done or timeout:
            should_save_episode = final_success_target is None or episode_success
            if not should_save_episode and current_start_id is not None:
                final_bias = None
                if success_anchor_xy is not None:
                    final_bias = np.linalg.norm(current_xy[:2] - success_anchor_xy[:2])
                discard_reason = "unsuccessful"
                if step_threshold is not None and episode_steps >= step_threshold:
                    discard_reason = f"step_threshold_exceeded(threshold={step_threshold})"
                print(
                    f"discarded {discard_reason} episode "
                    f"from start {current_start_id}: "
                    f"steps={episode_steps}, last_reward={reward}, "
                    f"final_bias={final_bias}, "
                    f"progress=({format_start_progress(success_counts)})"
                )
            if should_save_episode:
                if (
                    final_success_target is None
                    or current_start_id is None
                    or success_counts[current_start_id] < final_success_target
                ):
                    extend_data(data, episode_data)
                    if final_success_target is not None and current_start_id is not None:
                        if episode_success:
                            reward_hit_count = int(
                                np.sum(np.asarray(episode_data["rewards"], dtype=np.float32) == 1.0)
                            )
                            successful_episodes.append(
                                (current_start_id, clone_episode_data(episode_data))
                            )
                            success_counts[current_start_id] += 1
                            print(
                                "saved successful episode "
                                f"for start {current_start_id}: "
                                f"{success_counts[current_start_id]}/{final_success_target} "
                                f"({format_start_progress(success_counts)}), "
                                f"reward_hit_count={reward_hit_count}"
                            )
                            if milestone_counts is not None:
                                current_min_success = min(success_counts)
                                for milestone in milestone_counts:
                                    if (
                                        milestone <= current_min_success
                                        and milestone not in exported_milestones
                                    ):
                                        milestone_data, milestone_per_start = (
                                            build_dataset_from_successful_episodes(
                                                successful_episodes,
                                                len(all_fixed_starts),
                                                milestone,
                                            )
                                        )
                                        milestone_output = build_snapshot_output_path(
                                            args.output, milestone
                                        )
                                        save_hdf5_dataset(milestone_data, milestone_output)
                                        exported_milestones.add(milestone)
                                        print(
                                            "exported milestone dataset "
                                            f"({milestone} per start, actual={milestone_per_start}) "
                                            f"to {milestone_output}"
                                        )

                            next_active_start_ids = get_active_start_ids(
                                success_counts, final_success_target
                            )
                            if next_active_start_ids != active_start_ids:
                                active_start_ids = update_active_starts(
                                    env,
                                    all_fixed_starts,
                                    success_counts,
                                    final_success_target,
                                )

            episode_data = reset_data()
            if success_counts is not None and not active_start_ids:
                break
            obs = env.reset()
            if success_counts is not None and active_start_ids:
                current_start_id = active_start_ids[env.last_reset_start_idx]
            episode_steps = 0
            success_hold_counter = 0
            success_keep_steps_target = None
            success_anchor_xy = None
        else:
            obs = next_obs

        if args.render:
            env.render()

    if final_success_target is not None and all_fixed_starts is not None:
        final_data, final_per_start = build_dataset_from_successful_episodes(
            successful_episodes,
            len(all_fixed_starts),
            final_success_target,
        )
        save_hdf5_dataset(final_data, args.output)
    else:
        final_per_start = None
        save_hdf5_dataset(data, args.output)

    print(f"saved dataset to {args.output}")
    if success_counts is not None:
        print(f"successful episodes per start: {format_start_progress(success_counts)}")
    if final_per_start is not None:
        print(
            "saved dataset per-start counts: "
            + ", ".join(f"start{idx}={count}" for idx, count in enumerate(final_per_start))
        )


if __name__ == "__main__":
    main()
