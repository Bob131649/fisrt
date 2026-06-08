import argparse
import json
import select
import sys
import termios
import threading
import time
import tty
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import h5py
import numpy as np


REPO_ROOT = Path(__file__).resolve().parent
ROBOT_ARMS_SRC = REPO_ROOT / "robot_arms" / "src"
if str(ROBOT_ARMS_SRC) not in sys.path:
    sys.path.insert(0, str(ROBOT_ARMS_SRC))

from robot_arms.arms import FrankaLeft, FrankaRight  # noqa: E402
from robot_arms.franka.franka_basic import Franka  # noqa: E402


STAGE_IDLE = 0
STAGE_INIT = 1
STAGE_MOVE_TO_GRASP = 2
STAGE_GRASP = 3
STAGE_MOVE_TO_PLACE = 4
STAGE_RELEASE = 5
STAGE_FINISHED = 6


@dataclass
class Keypoint:
    name: str
    translation: np.ndarray
    quaternion: np.ndarray
    frame: str = "global"


class Keyboard:
    def __enter__(self):
        self.fd = sys.stdin.fileno()
        self.old_settings = termios.tcgetattr(self.fd)
        tty.setcbreak(self.fd)
        return self

    def __exit__(self, exc_type, exc, tb):
        termios.tcsetattr(self.fd, termios.TCSADRAIN, self.old_settings)

    def read_key(self) -> Optional[str]:
        readable, _, _ = select.select([sys.stdin], [], [], 0.0)
        if not readable:
            return None
        return sys.stdin.read(1).lower()


class SharedAction:
    def __init__(self):
        self.lock = threading.Lock()
        self.stage = STAGE_IDLE
        self.target_translation = np.zeros(3, dtype=np.float64)
        self.target_quaternion = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
        self.target_gripper_width = 0.04

    def set(self, stage, keypoint: Optional[Keypoint], gripper_width: float):
        with self.lock:
            self.stage = stage
            if keypoint is not None:
                self.target_translation = keypoint.translation.copy()
                self.target_quaternion = keypoint.quaternion.copy()
            self.target_gripper_width = float(gripper_width)

    def snapshot(self):
        with self.lock:
            return (
                self.stage,
                self.target_translation.copy(),
                self.target_quaternion.copy(),
                self.target_gripper_width,
            )


class EpisodeBuffer:
    def __init__(self):
        self.rows: List[Dict[str, np.ndarray]] = []
        self.started_at = time.time()
        self.ended_at = None

    def append(self, row):
        self.rows.append(row)

    def __len__(self):
        return len(self.rows)


def load_keypoints(path: Path) -> List[Keypoint]:
    with path.open("r", encoding="utf-8") as f:
        raw = json.load(f)

    if isinstance(raw, dict):
        raw_keypoints = raw.get("keypoints", raw.get("waypoints"))
    else:
        raw_keypoints = raw

    if not isinstance(raw_keypoints, list) or len(raw_keypoints) != 3:
        raise ValueError("keypoint JSON must contain exactly 3 keypoints: init, grasp, place")

    default_names = ["init", "grasp", "place"]
    keypoints = []
    for idx, item in enumerate(raw_keypoints):
        translation = np.asarray(item["translation"], dtype=np.float64)
        quaternion = np.asarray(item["quaternion"], dtype=np.float64)
        if translation.shape != (3,):
            raise ValueError(f"keypoint {idx} translation must have shape (3,)")
        if quaternion.shape != (4,):
            raise ValueError(f"keypoint {idx} quaternion must have shape (4,)")
        keypoints.append(
            Keypoint(
                name=item.get("name", default_names[idx]),
                translation=translation,
                quaternion=quaternion,
                frame=item.get("frame", "global"),
            )
        )
    return keypoints


def make_arm(args):
    arm_classes = {
        "left": FrankaLeft,
        "right": FrankaRight,
        "basic": Franka,
    }
    return arm_classes[args.arm](
        robot_ip=args.robot_ip,
        relative_dynamics_factor=args.relative_dynamics_factor,
    )


def move_to_keypoint(arm, keypoint: Keypoint, action: SharedAction, stage: int, gripper_width: float):
    action.set(stage, keypoint, gripper_width)
    arm.set_ee_pose(
        keypoint.translation,
        keypoint.quaternion,
        asynchronous=False,
        frame=keypoint.frame,
    )


def initialize_at_first_keypoint(arm, keypoints: List[Keypoint], args, action: SharedAction):
    init = keypoints[0]
    print(f"[init] hard impedance, moving to {init.name}")
    arm.set_hard()
    action.set(STAGE_INIT, init, args.open_width)
    arm.open_gripper(asynchronous=False)
    move_to_keypoint(arm, init, action, STAGE_INIT, args.open_width)
    arm.set_soft()
    print("[init] soft impedance, ready")


def collect_sample(arm, action: SharedAction, episode_idx: int):
    trans, quat, rpy = arm.get_ee_pose(frame="global")
    stage, target_trans, target_quat, target_gripper = action.snapshot()
    return {
        "timestamp": np.asarray(time.time(), dtype=np.float64),
        "episode_index": np.asarray(episode_idx, dtype=np.int64),
        "translation": np.asarray(trans, dtype=np.float64),
        "rotation": np.asarray(quat, dtype=np.float64),
        "rpy": np.asarray(rpy, dtype=np.float64),
        "joint_q": np.asarray(arm.get_joint_pose(), dtype=np.float64),
        "joint_dq": np.asarray(arm.get_joint_vel(), dtype=np.float64),
        "gripper_w": np.asarray(arm.get_gripper_width(), dtype=np.float64),
        "ee_force": np.asarray(arm.get_ee_force(), dtype=np.float64),
        "stage": np.asarray(stage, dtype=np.int64),
        "target_translation": np.asarray(target_trans, dtype=np.float64),
        "target_rotation": np.asarray(target_quat, dtype=np.float64),
        "target_gripper_w": np.asarray(target_gripper, dtype=np.float64),
        "reward": np.asarray(False, dtype=np.bool_),
        "done": np.asarray(False, dtype=np.bool_),
        "timeout": np.asarray(False, dtype=np.bool_),
    }


def recorder_loop(arm, action, buffer, stop_event, sample_period, episode_idx):
    next_t = time.monotonic()
    while not stop_event.is_set():
        try:
            buffer.append(collect_sample(arm, action, episode_idx))
        except Exception as exc:
            print(f"[record] sample failed: {exc}")
        next_t += sample_period
        sleep_s = next_t - time.monotonic()
        if sleep_s > 0:
            stop_event.wait(sleep_s)


def execute_open_loop(arm, keypoints, args, action, stop_event, sequence_done):
    try:
        arm.set_soft()
        move_to_keypoint(arm, keypoints[1], action, STAGE_MOVE_TO_GRASP, args.open_width)
        if stop_event.is_set():
            return

        action.set(STAGE_GRASP, keypoints[1], args.closed_width)
        arm.close_gripper(asynchronous=False)
        if stop_event.is_set():
            return

        move_to_keypoint(arm, keypoints[2], action, STAGE_MOVE_TO_PLACE, args.closed_width)
        if stop_event.is_set():
            return

        action.set(STAGE_RELEASE, keypoints[2], args.open_width)
        arm.open_gripper(asynchronous=False)
        action.set(STAGE_FINISHED, keypoints[2], args.open_width)
        print("[episode] open-loop sequence finished; press f to save")
    except Exception as exc:
        print(f"[episode] execution failed: {exc}")
        stop_event.set()
    finally:
        sequence_done.set()


def stack_rows(rows, name):
    return np.stack([row[name] for row in rows], axis=0)


def next_dataset_path(output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    indices = []
    for path in output_dir.glob("dataset_*.hdf5"):
        try:
            indices.append(int(path.stem.split("_")[-1]))
        except ValueError:
            continue
    next_idx = max(indices) + 1 if indices else 0
    return output_dir / f"dataset_{next_idx:06d}.hdf5"


def finish_episode(buffer: EpisodeBuffer, timeout: bool):
    if len(buffer) == 0:
        print("[episode] no samples collected; skip")
        return None

    buffer.rows[-1]["done"] = np.asarray(True, dtype=np.bool_)
    buffer.rows[-1]["timeout"] = np.asarray(timeout, dtype=np.bool_)
    buffer.ended_at = time.time()
    return {
        "buffer": buffer,
        "timeout": bool(timeout),
    }


def write_episode_group(parent, episode_data, episode_idx):
    buffer = episode_data["buffer"]
    rows = buffer.rows

    group = parent.create_group(f"episode_{episode_idx:06d}")
    group.attrs["episode_index"] = episode_idx
    group.attrs["created_at_unix"] = buffer.started_at
    group.attrs["ended_at_unix"] = buffer.ended_at
    group.attrs["timeout"] = bool(episode_data["timeout"])

    obs = group.create_group("observations")
    obs.create_dataset("translation", data=stack_rows(rows, "translation"), compression="gzip")
    obs.create_dataset("rotation", data=stack_rows(rows, "rotation"), compression="gzip")
    obs.create_dataset("rpy", data=stack_rows(rows, "rpy"), compression="gzip")
    obs.create_dataset("joint_q", data=stack_rows(rows, "joint_q"), compression="gzip")
    obs.create_dataset("joint_dq", data=stack_rows(rows, "joint_dq"), compression="gzip")
    obs.create_dataset("gripper_w", data=stack_rows(rows, "gripper_w"), compression="gzip")
    obs.create_dataset("ee_force", data=stack_rows(rows, "ee_force"), compression="gzip")

    actions = group.create_group("actions")
    actions.create_dataset("stage", data=stack_rows(rows, "stage"), compression="gzip")
    actions.create_dataset("target_translation", data=stack_rows(rows, "target_translation"), compression="gzip")
    actions.create_dataset("target_rotation", data=stack_rows(rows, "target_rotation"), compression="gzip")
    actions.create_dataset("target_gripper_w", data=stack_rows(rows, "target_gripper_w"), compression="gzip")

    group.create_dataset("timestamp", data=stack_rows(rows, "timestamp"), compression="gzip")
    group.create_dataset("episode_index", data=stack_rows(rows, "episode_index"), compression="gzip")
    group.create_dataset("reward", data=stack_rows(rows, "reward"), compression="gzip")
    group.create_dataset("done", data=stack_rows(rows, "done"), compression="gzip")
    group.create_dataset("timeout", data=stack_rows(rows, "timeout"), compression="gzip")

    # Backward-compatible names inside each episode group.
    group.create_dataset("translation", data=stack_rows(rows, "translation"), compression="gzip")
    group.create_dataset("rotation", data=stack_rows(rows, "rotation"), compression="gzip")
    group.create_dataset("gripper_w", data=stack_rows(rows, "gripper_w"), compression="gzip")


def save_dataset(episodes, path: Path, keypoints, args):
    if len(episodes) == 0:
        print("[save] no finished episodes; skip")
        return

    with h5py.File(path, "w") as h5:
        h5.attrs["created_at_unix"] = episodes[0]["buffer"].started_at
        h5.attrs["ended_at_unix"] = time.time()
        h5.attrs["sample_rate_hz"] = args.rate
        h5.attrs["soft_impedance"] = "franka_basic.set_soft default"
        h5.attrs["hard_impedance"] = "franka_basic.set_hard default"
        h5.attrs["robot_ip"] = args.robot_ip
        h5.attrs["arm"] = args.arm
        h5.attrs["num_episodes"] = len(episodes)

        kp_group = h5.create_group("keypoints")
        for idx, kp in enumerate(keypoints):
            group = kp_group.create_group(f"{idx}_{kp.name}")
            group.create_dataset("translation", data=kp.translation)
            group.create_dataset("quaternion", data=kp.quaternion)
            group.attrs["frame"] = kp.frame

        episodes_group = h5.create_group("episodes")
        for episode_idx, episode_data in enumerate(episodes):
            write_episode_group(episodes_group, episode_data, episode_idx)

    print(f"[save] {len(episodes)} episodes -> {path}")


def stop_current_episode(arm, episode, timeout=True):
    if episode is None:
        return
    episode["stop_event"].set()
    if timeout:
        try:
            arm.stop_motion()
        except Exception as exc:
            print(f"[stop] robot stop failed: {exc}")
    episode["record_thread"].join()
    episode["exec_thread"].join()


def start_episode(arm, keypoints, args, action, episode_idx):
    stop_event = threading.Event()
    sequence_done = threading.Event()
    buffer = EpisodeBuffer()
    sample_period = 1.0 / args.rate

    record_thread = threading.Thread(
        target=recorder_loop,
        args=(arm, action, buffer, stop_event, sample_period, episode_idx),
        daemon=True,
    )
    exec_thread = threading.Thread(
        target=execute_open_loop,
        args=(arm, keypoints, args, action, stop_event, sequence_done),
        daemon=True,
    )
    record_thread.start()
    exec_thread.start()
    print("[episode] started: s ignored while active, f finishes timeout=1 and starts next")
    return {
        "stop_event": stop_event,
        "sequence_done": sequence_done,
        "record_thread": record_thread,
        "exec_thread": exec_thread,
        "buffer": buffer,
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Collect Franka open-loop keypoint episodes into HDF5."
    )
    parser.add_argument("--robot-ip", default="192.168.31.11")
    parser.add_argument("--arm", choices=["left", "right", "basic"], default="left")
    parser.add_argument("--keypoints", required=True, type=Path)
    parser.add_argument("--output-dir", default=Path("dataset/franka"), type=Path)
    parser.add_argument("--rate", default=20.0, type=float)
    parser.add_argument("--relative-dynamics-factor", default=0.04, type=float)
    parser.add_argument("--open-width", default=0.04, type=float)
    parser.add_argument("--closed-width", default=0.0, type=float)
    return parser.parse_args()


def main():
    args = parse_args()
    keypoints = load_keypoints(args.keypoints)
    output_dir = args.output_dir.resolve()
    action = SharedAction()
    arm = make_arm(args)
    episode = None
    episodes = []

    print("controls: s=start first episode, f=finish timeout=1 and start next, k=reinitialize, q=save dataset and quit")
    initialize_at_first_keypoint(arm, keypoints, args, action)

    with Keyboard() as keyboard:
        try:
            while True:
                key = keyboard.read_key()
                if key is None:
                    time.sleep(0.02)
                    continue

                if key == "s":
                    if episode is not None:
                        print("[episode] already active")
                        continue
                    episode_idx = len(episodes)
                    episode = start_episode(arm, keypoints, args, action, episode_idx)

                elif key == "f":
                    if episode is None:
                        print("[episode] no active episode")
                        continue
                    stop_current_episode(arm, episode, timeout=True)
                    finished = finish_episode(episode["buffer"], timeout=True)
                    if finished is not None:
                        episodes.append(finished)
                    episode = None
                    initialize_at_first_keypoint(arm, keypoints, args, action)
                    episode = start_episode(arm, keypoints, args, action, len(episodes))

                elif key == "k":
                    if episode is not None:
                        stop_current_episode(arm, episode, timeout=True)
                        finished = finish_episode(episode["buffer"], timeout=True)
                        if finished is not None:
                            episodes.append(finished)
                        episode = None
                    initialize_at_first_keypoint(arm, keypoints, args, action)

                elif key == "q":
                    if episode is not None:
                        stop_current_episode(arm, episode, timeout=False)
                        finished = finish_episode(episode["buffer"], timeout=False)
                        if finished is not None:
                            episodes.append(finished)
                        episode = None
                    save_dataset(episodes, next_dataset_path(output_dir), keypoints, args)
                    break
        finally:
            try:
                arm.set_hard()
            except Exception:
                pass


if __name__ == "__main__":
    main()
