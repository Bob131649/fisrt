import argparse

import d4rl
import gym
import numpy as np
from mujoco_py.generated import const


def add_circle_marker(viewer, point, rgba, radius, height, label="", xy_offset=(0.0, 0.0)):
    viewer.add_marker(
        pos=np.array([point[0] + xy_offset[0], point[1] + xy_offset[1], height]),
        size=np.array([radius, radius, radius]),
        rgba=np.array(rgba),
        type=const.GEOM_SPHERE,
        label=label,
    )


def parse_start_position(start_position):
    parts = [part.strip() for part in start_position.split(",")]
    if len(parts) != 2:
        raise ValueError("start_position must be formatted as 'x,y', for example '1,7'.")
    return np.array([float(parts[0]), float(parts[1])], dtype=np.float32)


def get_marker_offset(env_name):
    if "maze2d" in env_name:
        return (1.0, 1.0)
    return (0.0, 0.0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Visualize a specified start position on a maze environment."
    )
    parser.add_argument(
        "--env_name",
        type=str,
        required=True,
        help="Registered Gym env name, e.g. maze2d-large-v1 or antmaze-large-diverse-v2.",
    )
    parser.add_argument(
        "--start_position",
        type=str,
        required=True,
        help="Start position formatted as 'x,y', e.g. '1,7'.",
    )
    parser.add_argument(
        "--height",
        type=float,
        default=0.2,
        help="Marker height in the rendered scene.",
    )
    parser.add_argument(
        "--radius",
        type=float,
        default=0.2,
        help="Marker radius.",
    )
    args = parser.parse_args()

    env = gym.make(args.env_name)
    start_position = parse_start_position(args.start_position)
    marker_offset = get_marker_offset(args.env_name)

    env.reset()
    env.render()
    base_env = env.unwrapped

    if base_env.viewer is not None and hasattr(base_env.viewer, "_markers"):
        base_env.viewer._markers[:] = []

    if "maze2d" in args.env_name and hasattr(base_env, "reset_to_location"):
        base_env.reset_to_location(start_position)
    elif hasattr(base_env, "set_xy"):
        base_env.set_xy(start_position)

    while True:
        if base_env.viewer is not None and hasattr(base_env.viewer, "_markers"):
            base_env.viewer._markers[:] = []
            add_circle_marker(
                base_env.viewer,
                start_position,
                rgba=[1.0, 0.0, 0.0, 0.95],
                radius=args.radius,
                height=args.height,
                label="start",
                xy_offset=marker_offset,
            )
        env.render()
