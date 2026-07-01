import gym
import numpy as np


ENV_RESET_PRESETS = {
    "maze2d-large": {
        "fixed_starts": [
            [1.0, 1.0],
            [1.0, 8.0],
            [5.0, 4.0],
            [7.0, 1.0],
        ],
        "fixed_goal": [7.0, 9.0],
        "start_format": "xy",
        "goal_format": "xy",
    },
    "antmaze-large": {
        "fixed_starts": [
            [1.0, 1.0],
            [3.0, 1.0],
            [1.0, 6.0],
            [1.0, 10.0],
        ],
        "fixed_goal": [33.0, 25.0],
        "start_format": "rowcol",
        "goal_format": "xy",
    },
}


def _to_point_array(point):
    point_array = np.asarray(point, dtype=np.float32).reshape(-1)
    if point_array.shape[0] < 2:
        raise ValueError(f"Point must have at least 2 coordinates, got {point}.")
    return point_array


def _infer_preset_key(env_name):
    if env_name is None:
        return None
    if "maze2d" in env_name:
        return "maze2d-large"
    if "antmaze" in env_name:
        return "antmaze-large"
    return None


class FixedResetWrapper(gym.Wrapper):
    """
    Force reset() to use user-provided start/goal locations.

    Supports:
    - Maze2D: fixed goal via set_target(), fixed start via reset_to_location()
    - AntMaze: fixed goal via set_target()/set_target_goal(), fixed start via set_xy()

    For AntMaze starts, `start_format` can be:
    - "xy": start points are continuous xy coordinates
    - "rowcol": start points are maze row/col coordinates
    - "auto": infer integer 2-tuples as row/col, otherwise treat as xy
    """

    def __init__(
        self,
        env,
        env_name=None,
        fixed_starts=None,
        fixed_goal=None,
        start_mode="cycle",
        start_format=None,
        goal_format=None,
        start_noise_scale=0.1,
        goal_noise_scale=0.1,
    ):
        super().__init__(env)
        self.fixed_starts = None
        self.fixed_goal = None
        self.start_mode = start_mode
        self.env_name = env_name or getattr(getattr(env, "spec", None), "id", "")
        self.start_format = start_format or "auto"
        self.goal_format = goal_format or "auto"
        self.start_noise_scale = float(start_noise_scale)
        self.goal_noise_scale = float(goal_noise_scale)
        self._next_start_idx = 0
        self.last_reset_start_idx = None

        preset_key = _infer_preset_key(self.env_name)
        preset = ENV_RESET_PRESETS.get(preset_key)
        if preset is not None:
            if fixed_starts is None:
                fixed_starts = preset.get("fixed_starts")
            if fixed_goal is None:
                fixed_goal = preset.get("fixed_goal")
            if start_format is None:
                self.start_format = preset.get("start_format", "auto")
            if goal_format is None:
                self.goal_format = preset.get("goal_format", "auto")

        if fixed_starts is not None:
            self.set_fixed_starts(fixed_starts)
        if fixed_goal is not None:
            self.set_fixed_goal(fixed_goal)

    def set_fixed_starts(self, fixed_starts):
        starts = []
        for start in fixed_starts:
            starts.append(_to_point_array(start))
        self.fixed_starts = starts or None
        self._next_start_idx = 0

    def set_fixed_goal(self, fixed_goal):
        self.fixed_goal = _to_point_array(fixed_goal)

    def clear_fixed_starts(self):
        self.fixed_starts = None
        self._next_start_idx = 0
        self.last_reset_start_idx = None

    def clear_fixed_goal(self):
        self.fixed_goal = None

    def reset(self, **kwargs):
        if not self.fixed_starts:
            obs = self.env.reset(**kwargs)
            self._apply_fixed_goal()
            return self._get_obs_after_goal_update(default_obs=obs)

        start = self._select_start()
        env_name = self.env_name or getattr(getattr(self.env, "spec", None), "id", "")

        if "maze2d" in env_name:
            self.env.reset(**kwargs)
            self._apply_fixed_goal()
            return self._reset_maze2d_start(start)

        if "antmaze" in env_name:
            self.env.reset(**kwargs)
            self._apply_fixed_goal()
            self._apply_antmaze_start(start)
            return self._get_obs()

        raise NotImplementedError(
            f"FixedResetWrapper does not support env '{env_name}' yet."
        )

    def _apply_fixed_goal(self):
        if self.fixed_goal is None:
            return

        goal = self._resolve_goal(self.fixed_goal)

        if hasattr(self.env, "set_target"):
            self.env.set_target(goal[:2])
            return
        if hasattr(self.env, "set_target_goal"):
            self.env.set_target_goal(goal[:2])
            return

        base_env = self.env.unwrapped
        if hasattr(base_env, "set_target"):
            base_env.set_target(goal[:2])
            return
        if hasattr(base_env, "set_target_goal"):
            base_env.set_target_goal(goal[:2])
            return

        raise AttributeError("Environment does not provide a way to set a fixed goal.")

    def _select_start(self):
        if self.start_mode == "random":
            start_idx = np.random.randint(len(self.fixed_starts))
        else:
            start_idx = self._next_start_idx % len(self.fixed_starts)
            self._next_start_idx = (self._next_start_idx + 1) % len(self.fixed_starts)
        self.last_reset_start_idx = start_idx
        return self.fixed_starts[start_idx]

    def _apply_antmaze_start(self, start):
        base_env = self.env.unwrapped
        start_xy = self._resolve_antmaze_start(start, base_env)
        self.env.set_xy(start_xy)

    def _resolve_antmaze_start(self, start, base_env):
        if self.start_format == "xy":
            return self._add_xy_noise(start[:2], self.start_noise_scale)

        if self.start_format == "rowcol":
            if not hasattr(base_env, "_rowcol_to_xy"):
                raise AttributeError("AntMaze env does not expose _rowcol_to_xy().")
            rowcol = tuple(int(value) for value in start[:2])
            start_xy = base_env._rowcol_to_xy(rowcol, add_random_noise=False)
            return self._add_xy_noise(start_xy[:2], self.start_noise_scale)

        if self.start_format == "auto":
            rounded = np.round(start[:2])
            if np.allclose(start[:2], rounded) and hasattr(base_env, "_rowcol_to_xy"):
                rowcol = tuple(int(value) for value in rounded)
                start_xy = base_env._rowcol_to_xy(rowcol, add_random_noise=False)
                return self._add_xy_noise(start_xy[:2], self.start_noise_scale)
            return self._add_xy_noise(start[:2], self.start_noise_scale)

        raise ValueError(f"Unknown start_format: {self.start_format}")

    def _resolve_goal(self, goal):
        env_name = self.env_name or getattr(getattr(self.env, "spec", None), "id", "")
        if "antmaze" not in env_name:
            return self._add_xy_noise(goal[:2], self.goal_noise_scale)

        base_env = self.env.unwrapped
        if self.goal_format == "xy":
            return self._add_xy_noise(goal[:2], self.goal_noise_scale)

        if self.goal_format == "rowcol":
            if not hasattr(base_env, "_rowcol_to_xy"):
                raise AttributeError("AntMaze env does not expose _rowcol_to_xy().")
            rowcol = tuple(int(value) for value in goal[:2])
            goal_xy = base_env._rowcol_to_xy(rowcol, add_random_noise=False)
            return self._add_xy_noise(goal_xy[:2], self.goal_noise_scale)

        if self.goal_format == "auto":
            rounded = np.round(goal[:2])
            if np.allclose(goal[:2], rounded) and hasattr(base_env, "_rowcol_to_xy"):
                rowcol = tuple(int(value) for value in rounded)
                goal_xy = base_env._rowcol_to_xy(rowcol, add_random_noise=False)
                return self._add_xy_noise(goal_xy[:2], self.goal_noise_scale)
            return self._add_xy_noise(goal[:2], self.goal_noise_scale)

        raise ValueError(f"Unknown goal_format: {self.goal_format}")

    def _reset_maze2d_start(self, start):
        if self.start_noise_scale > 0:
            return self.env.reset_to_location(start[:2])

        if not hasattr(self.env.unwrapped, "set_state"):
            return self.env.reset_to_location(start[:2])

        self.env.unwrapped.sim.reset()
        reset_location = np.array(start[:2], dtype=self.env.observation_space.dtype)
        qpos = reset_location.copy()
        qvel = self.env.unwrapped.init_qvel + self.env.unwrapped.np_random.randn(self.env.unwrapped.model.nv) * 0.1
        self.env.unwrapped.set_state(qpos, qvel)
        return self.env.unwrapped._get_obs()

    def _add_xy_noise(self, xy, noise_scale):
        xy = np.asarray(xy, dtype=np.float32).copy()
        if noise_scale <= 0:
            return xy
        noise = np.random.uniform(low=-noise_scale, high=noise_scale, size=xy.shape)
        return xy + noise.astype(np.float32)

    def _get_obs(self):
        if hasattr(self.env, "_get_obs"):
            return self.env._get_obs()
        if hasattr(self.env, "wrapped_env") and hasattr(self.env.wrapped_env, "_get_obs"):
            return self.env.wrapped_env._get_obs()
        if hasattr(self.env, "_wrapped_env") and hasattr(self.env._wrapped_env, "_get_obs"):
            return self.env._wrapped_env._get_obs()
        if hasattr(self.env, "unwrapped") and hasattr(self.env.unwrapped, "_get_obs"):
            return self.env.unwrapped._get_obs()
        raise AttributeError("Could not recover observation after applying fixed reset.")

    def _get_obs_after_goal_update(self, default_obs=None):
        try:
            return self._get_obs()
        except AttributeError:
            if default_obs is not None:
                return default_obs
            raise

def build_wrapped_env(env_name, start_mode="cycle", start_noise_scale=0.1, goal_noise_scale=0):
    base_env = gym.make(env_name)
    return FixedResetWrapper(
        base_env,
        env_name=env_name,
        start_mode=start_mode,
        start_noise_scale=start_noise_scale,
        goal_noise_scale=goal_noise_scale,
    )


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
