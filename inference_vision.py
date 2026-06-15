#!/usr/bin/env python3

import argparse
import json
import time
from pathlib import Path

import algos.algos_v2_vision as algos
import numpy as np
import pyrealsense2 as rs
import torch
import cv2



def load_variant(model_dir: Path):
    path = model_dir / "variant.json"
    if path.exists():
        return json.loads(path.read_text())
    return {}


class Camera:
    def __init__(self, serial, width, height, fps):
        self.pipeline = rs.pipeline()
        cfg = rs.config()
        if serial:
            cfg.enable_device(serial)
        cfg.enable_stream(rs.stream.color, width, height, rs.format.rgb8, fps)
        self.cfg = cfg

    def __enter__(self):
        self.pipeline.start(self.cfg)
        return self

    def __exit__(self, exc_type, exc, tb):
        self.pipeline.stop()

    def rgb(self):
        frame = self.pipeline.wait_for_frames(1000).get_color_frame()
        if not frame:
            raise RuntimeError("no color frame")
        return np.asanyarray(frame.get_data()).copy()


def make_arm(args):
    from robot_arms.arms import FrankaLeft, FrankaRight
    from robot_arms.franka.franka_basic import Franka

    cls = {"left": FrankaLeft, "right": FrankaRight, "basic": Franka}[args.arm]
    return cls(args.robot_ip, args.relative_dynamics_factor, control_mode=args.control_mode)


def resize(rgb, image_size):
    if cv2 is None:
        raise ImportError("opencv-python is required for image resizing")
    return cv2.resize(rgb, (image_size[1], image_size[0]), interpolation=cv2.INTER_AREA)


def state_from_robot(arm):
    trans, quat, _ = arm.get_ee_pose(frame="global")
    quat = np.asarray(quat, dtype=np.float32)
    quat /= max(np.linalg.norm(quat), 1e-6)
    return np.r_[trans, quat, arm.get_gripper_width()].astype(np.float32)


def norm(x, mean, std):
    return ((x - mean) / (std + 1e-6)).astype(np.float32)


def denorm(x, mean, std):
    return (x * (std + 1e-6) + mean).astype(np.float32)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model-dir", required=True, type=Path)
    p.add_argument("--model-name", default="model")
    p.add_argument("--load-best", action="store_true")
    p.add_argument("--execute", action="store_true")

    p.add_argument("--robot-ip", default="192.168.31.12")
    p.add_argument("--arm", choices=["left", "right", "basic"], default="basic")
    p.add_argument("--control-mode", choices=["curobo", "armlib"], default="armlib")
    p.add_argument("--relative-dynamics-factor", type=float, default=0.03)

    p.add_argument("--camera-serial", default="230422272989")
    p.add_argument("--camera-width", type=int, default=640)
    p.add_argument("--camera-height", type=int, default=480)
    p.add_argument("--camera-fps", type=int, default=30)

    p.add_argument("--rate", type=float, default=5.0)
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--image-size", nargs=2, type=int, default=None)
    p.add_argument("--action-mode", choices=["absolute_eef", "relative_eef"], default=None)
    p.add_argument("--device", default=None)
    p.add_argument("--vae", action="store_true")

    p.add_argument("--open-width", type=float, default=0.04)
    p.add_argument("--closed-width", type=float, default=0.0)
    p.add_argument("--gripper-deadband", type=float, default=0.004)
    return p.parse_args()


def main():
    args = parse_args()
    variant = load_variant(args.model_dir)
    cli_action_mode = args.action_mode
    cli_image_size = args.image_size

    stats_meta = json.loads((args.model_dir / "normalization_stats.json").read_text())
    state_mean = np.asarray(stats_meta["state_mean"], dtype=np.float32)
    state_std = np.asarray(stats_meta["state_std"], dtype=np.float32)
    action_mean = np.asarray(stats_meta["action_mean"], dtype=np.float32)
    action_std = np.asarray(stats_meta["action_std"], dtype=np.float32)
    args.image_size = tuple(cli_image_size or stats_meta.get("image_size") or variant.get("image_size") or (224, 224))
    args.action_mode = cli_action_mode or stats_meta.get("action_mode") or variant.get("action_mode") or "absolute_eef"
    print(f"[stats] loaded normalization_stats.json from {args.model_dir}")
    device = args.device or variant.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    action_dim = int(variant.get("action_dim", 8))

    policy = algos.Latent(
        8,
        action_dim,
        action_dim * 2,
        0.0,
        100.0,
        device=device,
        image_shape=(3, args.image_size[0], args.image_size[1]),
        robomimic_feature_dim=int(variant.get("robomimic_feature_dim", 256)),
        robomimic_crop_shape=variant.get("robomimic_crop_shape"),
        robomimic_backbone_class=variant.get("robomimic_backbone_class", "ResNet18Conv"),
        robomimic_pool_class=variant.get("robomimic_pool_class", "SpatialSoftmax"),
        max_latent_action=float(variant.get("max_latent_action", 0.675)),
        expectile=float(variant.get("expectile", 0.9)),
        kl_beta=float(variant.get("kl_beta", 1.0)),
        doubleq_min=float(variant.get("doubleq_min", 1.0)),
        vae=bool(variant.get("vae", args.vae)),
    )
    policy.load(args.model_name, str(args.model_dir), load_best=args.load_best)
    policy.eval()

    arm = make_arm(args) if args.execute else None
    print(f"[ready] execute={args.execute}, action_mode={args.action_mode}. Press Ctrl-C to stop.")

    with Camera(args.camera_serial, args.camera_width, args.camera_height, args.camera_fps) as cam:
        for step in range(args.steps):
            tic = time.time()
            rgb = cam.rgb()
            if args.execute:
                state = state_from_robot(arm)
            else:
                state = np.r_[np.zeros(3), [0, 0, 0, 1], args.open_width].astype(np.float32)

            image = resize(rgb, args.image_size)
            action_n, q1, q2 = policy.select_action(norm(state, state_mean, state_std), image=image)
            action = denorm(action_n, action_mean, action_std)
            target = action[:8] if args.action_mode == "absolute_eef" else state + action[:8]
            trans = target[:3]
            quat = target[3:7] / max(np.linalg.norm(target[3:7]), 1e-6)
            grip = float(np.clip(target[7], args.closed_width, args.open_width))

            print(f"{step:04d} q=({q1:.3f},{q2:.3f}) trans={trans.round(4)} grip={grip:.3f}")
            if args.execute:
                arm.set_ee_pose(trans, quat, asynchronous=False, frame="global")
                if abs(state[7] - grip) > args.gripper_deadband:
                    arm.set_gripper_opening(grip, asynchronous=False)

            sleep = 1.0 / args.rate - (time.time() - tic)
            if sleep > 0:
                time.sleep(sleep)


if __name__ == "__main__":
    main()
