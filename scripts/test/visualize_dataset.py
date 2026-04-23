import argparse
import os
import time

import d4rl
import gym
import h5py
import numpy as np
from mujoco_py.generated import const


def compute_episode_ranges(terminals, timeouts):
    """Use terminals/timeouts only to split trajectories."""
    done = np.logical_or(terminals.astype(bool), timeouts.astype(bool))
    done_indices = np.where(done)[0]

    if len(done_indices) == 0:
        return [(0, len(terminals))]

    episode_starts = np.concatenate([[0], done_indices[:-1] + 1])
    episode_ranges = [(int(s), int(e) + 1) for s, e in zip(episode_starts, done_indices)]

    last_end = done_indices[-1] + 1
    if last_end < len(terminals):
        episode_ranges.append((int(last_end), len(terminals)))

    return episode_ranges


def get_hdf5_keys(h5file):
    keys = []

    def visitor(name, item):
        if isinstance(item, h5py.Dataset):
            keys.append(name)

    h5file.visititems(visitor)
    return keys


def load_dataset_from_hdf5(dataset_path):
    data_dict = {}
    with h5py.File(dataset_path, "r") as dataset_file:
        for key in get_hdf5_keys(dataset_file):
            try:
                data_dict[key] = dataset_file[key][:]
            except ValueError:
                data_dict[key] = dataset_file[key][()]
    return data_dict


def add_circle_marker(viewer, point, rgba, radius, height, label="", xy_offset=(0.0, 0.0)):
    viewer.add_marker(
        pos=np.array([point[0] + xy_offset[0], point[1] + xy_offset[1], height]),
        size=np.array([radius, radius, radius]),
        rgba=np.array(rgba),
        type=const.GEOM_SPHERE,
        label=label,
    )


def infer_episode_goal(dataset, observations, rewards, start_idx, end_idx, base_env):
    """
    Goal logic:
    1. Prefer dataset['infos/goal'] if present.
    2. Else, if success exists, use first reward==1 position.
    3. Else, use the last position of this segment as a fallback.
    """
    if "infos/goal" in dataset:
        return dataset["infos/goal"][start_idx][:2], "infos/goal"

    success_indices = np.where(rewards[start_idx:end_idx] == 1)[0]
    if len(success_indices) > 0:
        first_hit = start_idx + success_indices[0]
        return observations[first_hit, :2], "first_reward_hit"

    if hasattr(base_env, "target_goal"):
        return np.array(base_env.target_goal[:2]), "env.target_goal"

    return observations[end_idx - 1, :2], "last_observation"


def infer_episode_success(rewards, start_idx, end_idx):
    """
    Success is separate from trajectory splitting.
    For AntMaze sparse reward, reward==1 indicates success.
    """
    return bool(np.any(rewards[start_idx:end_idx] == 1))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--env_name",
        type=str,
        default=None,
        help="Optional display name for this dataset run. Does not need to be a registered Gym env.",
    )
    parser.add_argument(
        "--render_env_name",
        type=str,
        default="maze2d-large-v1",
        help="Registered Gym env used only for rendering/replay state.",
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        required=True,
        help="Local HDF5 dataset path to visualize.",
    )
    parser.add_argument(
        "--episode_id",
        type=int,
        default=0,
        help="Episode index to start from.",
    )
    parser.add_argument(
        "--show_start",
        action="store_true",
        help="Render start marker.",
    )
    parser.add_argument(
        "--show_goal",
        action="store_true",
        help="Render goal marker.",
    )
    parser.add_argument(
        "--loop_forever",
        action="store_true",
        help="Play episodes continuously without exiting.",
    )
    parser.add_argument(
        "--pause_between_episodes",
        type=float,
        default=0.5,
        help="Seconds to pause between episodes.",
    )
    args = parser.parse_args()

    dataset_name = args.env_name or os.path.splitext(os.path.basename(args.dataset_path))[0]
    env = gym.make(args.render_env_name)
    dataset = load_dataset_from_hdf5(args.dataset_path)

    required_keys = [
        "infos/qpos",
        "infos/qvel",
        "observations",
        "terminals",
        "timeouts",
        "rewards",
        "actions",
    ]
    for key in required_keys:
        if key not in dataset:
            raise ValueError(f"Dataset is missing key: {key}")

    qpos = dataset["infos/qpos"]
    qvel = dataset["infos/qvel"]
    observations = dataset["observations"]
    rewards = dataset["rewards"]
    actions = dataset["actions"]
    terminals = dataset["terminals"]
    timeouts = dataset["timeouts"]

    episode_ranges = compute_episode_ranges(terminals, timeouts)
    num_episodes = len(episode_ranges)

    if args.episode_id < 0 or args.episode_id >= num_episodes:
        raise ValueError(
            f"episode_id={args.episode_id} is out of range. "
            f"Valid range: [0, {num_episodes - 1}]"
        )

    print(f"Loaded {qpos.shape[0]} transitions")
    print(f"Dataset name: {dataset_name}")
    print(f"Render env: {args.render_env_name}")
    print(f"Observations shape: {observations.shape}")
    print(f"Rewards shape: {rewards.shape}")
    print(f"Actions shape: {actions.shape}")
    print(f"Total episodes (split by terminals/timeouts): {num_episodes}")
    print(f"Start playing from episode {args.episode_id}")

    base_env = env.unwrapped
    marker_offset = (1.0, 1.0) if "maze2d" in args.render_env_name else (0.0, 0.0)
    env.reset()
    env.render()

    current_episode = args.episode_id

    while True:
        start_idx, end_idx = episode_ranges[current_episode]
        episode_len = end_idx - start_idx

        episode_start = observations[start_idx, :2]
        episode_goal, goal_source = infer_episode_goal(
            dataset, observations, rewards, start_idx, end_idx, base_env
        )
        episode_success = infer_episode_success(rewards, start_idx, end_idx)
        episode_return = float(np.sum(rewards[start_idx:end_idx]))

        print(
            f"\nEpisode {current_episode}: "
            f"indices [{start_idx}, {end_idx}), "
            f"length={episode_len}, return={episode_return:.4f}, success={episode_success}"
        )
        print(f"  start xy: {episode_start}")
        print(f"  goal  xy: {episode_goal}  (source={goal_source})")
        print(
            f"  terminal end={bool(terminals[end_idx - 1])}, "
            f"timeout end={bool(timeouts[end_idx - 1])}"
        )

        for t in range(start_idx, end_idx):
            base_env.set_state(qpos[t], qvel[t])

            if base_env.viewer is not None:
                if hasattr(base_env.viewer, "_markers"):
                    base_env.viewer._markers[:] = []

                # Keep camera roughly centered on current ant position
                # if hasattr(base_env.viewer, "cam"):
                #     base_env.viewer.cam.lookat[0] = observations[t, 0]
                #     base_env.viewer.cam.lookat[1] = observations[t, 1]

                if args.show_start:
                    add_circle_marker(
                        base_env.viewer,
                        episode_start,
                        rgba=[0.2, 0.4, 1.0, 0.95],
                        radius=0.2,
                        height=0.2,
                        label="start",
                        xy_offset=marker_offset,
                    )

                if args.show_goal and episode_goal is not None:
                    add_circle_marker(
                        base_env.viewer,
                        episode_goal,
                        rgba=[1.0, 0.2, 0.2, 0.95],
                        radius=0.2,
                        height=0.2,
                        label="goal",
                        xy_offset=marker_offset,
                    )

            env.render()

        time.sleep(args.pause_between_episodes)

        current_episode += 1
        if current_episode >= num_episodes:
            if args.loop_forever:
                current_episode = 0
            else:
                break
