import signal

import numpy as np
import pyrealsense2 as rs

from deploy.config import CAMERA, ROBOT


class RealSenseCamera:
    def __init__(self, config=CAMERA):
        self.pipeline = rs.pipeline()
        rs_config = rs.config()
        if config.serial:
            rs_config.enable_device(config.serial)
        rs_config.enable_stream(
            rs.stream.color,
            config.width,
            config.height,
            rs.format.rgb8,
            config.fps,
        )
        self.rs_config = rs_config

    def __enter__(self):
        self.pipeline.start(self.rs_config)
        return self

    def __exit__(self, exc_type, exc, tb):
        try:
            self.pipeline.stop()
        except KeyboardInterrupt:
            print("[cleanup] camera stop interrupted; continuing shutdown")

    def read_rgb(self):
        frame = self.pipeline.wait_for_frames(1000).get_color_frame()
        if not frame:
            raise RuntimeError("no color frame")
        return np.asanyarray(frame.get_data()).copy()


class FinishCleanupBeforeCtrlC:
    def __init__(self):
        self._previous_handler = None
        self._caught = None

    def _handler(self, sig, frame):
        self._caught = (sig, frame)
        print("[cleanup] Ctrl-C received, finishing save before exit")

    def __enter__(self):
        self._previous_handler = signal.getsignal(signal.SIGINT)
        signal.signal(signal.SIGINT, self._handler)
        return self

    def __exit__(self, exc_type, exc, tb):
        signal.signal(signal.SIGINT, self._previous_handler)
        if self._caught is not None:
            raise KeyboardInterrupt


def make_franka_arm(config=ROBOT):
    from robot_arms.arms import FrankaLeft, FrankaRight
    from robot_arms.franka.franka_basic import Franka

    arm_cls = {"left": FrankaLeft, "right": FrankaRight, "basic": Franka}[config.arm]
    return arm_cls(
        config.ip,
        config.relative_dynamics_factor,
        control_mode=config.control_mode,
    )


def read_robot_state(arm):
    trans, quat, _ = arm.get_ee_pose(frame="global")
    quat = np.asarray(quat, dtype=np.float32)
    quat /= max(np.linalg.norm(quat), 1e-6)
    return np.r_[trans, quat, arm.get_gripper_width()].astype(np.float32)


def make_fake_state(open_width):
    return np.r_[np.zeros(3), [0, 0, 0, 1], open_width].astype(np.float32)


def move_robot_to_target_state(arm, target_state, action_delta, config=ROBOT, open_width=0.04, closed_width=0.0, deadband=0.001):
    target_state = np.asarray(target_state, dtype=np.float32)
    translation = target_state[:3]
    quat = target_state[3:7] / max(np.linalg.norm(target_state[3:7]), 1e-6)
    gripper_width = float(np.clip(target_state[7], closed_width, open_width))
    gripper_delta = float(action_delta[7])

    arm.set_ee_pose(translation, quat, asynchronous=config.async_motion, frame="global")
    if gripper_delta < -deadband:
        print("[gripper] close")
        arm.close_gripper(asynchronous=False)
    elif gripper_delta > deadband:
        print("[gripper] open")
        arm.open_gripper(asynchronous=False)
    return translation, gripper_width, gripper_delta
