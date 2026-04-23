import argparse

import d4rl
import gym
import numpy as np
from mujoco_py.generated import const


def add_circle_marker(viewer, point, rgba, radius, height, label=""):
    viewer.add_marker(
        pos=np.array([point[0], point[1], height]),
        size=np.array([radius, radius, radius]),
        rgba=np.array(rgba),
        type=const.GEOM_SPHERE,
        label=label,
    )


def parse_rowcol(rowcol_text):
    parts = [part.strip() for part in rowcol_text.split(",")]
    if len(parts) != 2:
        raise ValueError("start_position must be formatted as 'row,col', for example '1,7'.")
    return int(parts[0]), int(parts[1])


def get_antmaze_xy(base_env, rowcol, add_random_noise=False):
    if not hasattr(base_env, "_rowcol_to_xy"):
        raise AttributeError("The loaded environment does not expose _rowcol_to_xy().")
    return np.array(
        base_env._rowcol_to_xy(rowcol, add_random_noise=add_random_noise),
        dtype=np.float32,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Visualize a specified AntMaze start cell by converting row/col into the environment xy coordinates."
    )
    parser.add_argument(
        "--env_name",
        type=str,
        required=True,
        help="Registered AntMaze env name, e.g. antmaze-large-play-v2.",
    )
    parser.add_argument(
        "--start_position",
        type=str,
        required=True,
        help="Start maze cell formatted as 'row,col', e.g. '1,7'.",
    )
    parser.add_argument(
        "--height",
        type=float,
        default=0.8,
        help="Marker height in the rendered scene.",
    )
    parser.add_argument(
        "--radius",
        type=float,
        default=0.6,
        help="Marker radius.",
    )
    parser.add_argument(
        "--use_env_noise",
        action="store_true",
        help="Use the environment's built-in row/col to xy random noise.",
    )
    args = parser.parse_args()

    env = gym.make(args.env_name)
    rowcol = parse_rowcol(args.start_position)

    env.reset()
    env.render()
    base_env = env.unwrapped

    if base_env.viewer is not None and hasattr(base_env.viewer, "_markers"):
        base_env.viewer._markers[:] = []

    start_xy = get_antmaze_xy(base_env, rowcol, add_random_noise=args.use_env_noise)
    if hasattr(base_env, "set_xy"):
        base_env.set_xy(start_xy)
    else:
        raise AttributeError("The loaded environment does not expose set_xy().")

    print(f"start row/col: {rowcol}")
    print(f"start xy: ({start_xy[0]:.4f}, {start_xy[1]:.4f})")
    if hasattr(base_env, "target_goal"):
        print(f"target goal: ({base_env.target_goal[0]:.4f}, {base_env.target_goal[1]:.4f})")

    while True:
        if base_env.viewer is not None and hasattr(base_env.viewer, "_markers"):
            base_env.viewer._markers[:] = []
            add_circle_marker(
                base_env.viewer,
                start_xy,
                rgba=[1.0, 0.0, 0.0, 0.95],
                radius=args.radius,
                height=args.height,
                label="start",
            )
        env.render()
