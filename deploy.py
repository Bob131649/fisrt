#!/usr/bin/env python3

import argparse
import json
import time
from pathlib import Path

import numpy as np

from deploy.config import (
    ACTION_SMOOTHING,
    CLOSED_GRIPPER_WIDTH,
    DEFAULT_RATE_HZ,
    DEFAULT_STEPS,
    FALLBACK_IMAGE_SIZE,
    GRIPPER_DEADBAND,
    OPEN_GRIPPER_WIDTH,
    ROBOT,
    TRAIN_RGB_CROP,
)
from deploy.policy import load_latent_policy
from deploy.debug_view import (
    DeployRecorder,
    action_reconstruction_mse,
    close_debug_windows,
    encoder_attention_overlay,
    render_preview,
    show_debug_window,
)
from deploy.keyboard import SpaceKeyController
from deploy.robot_io import (
    FinishCleanupBeforeCtrlC,
    RealSenseCamera,
    make_fake_state,
    make_franka_arm,
    move_robot_to_target_state,
    read_robot_state,
)
from deploy.vision import crop_and_resize_rgb, normalize, smooth_action, stats_crop_to_deploy_crop, unnormalize


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--model-name", default="model")
    parser.add_argument("--load-best", action="store_true")
    parser.add_argument("--device", default=None)
    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    parser.add_argument("--rate", type=float, default=DEFAULT_RATE_HZ)
    parser.add_argument("--dry-run", action="store_true", help="Do not connect to or move the real robot.")
    parser.add_argument("--show-debug", action="store_true", help="Show RGB, policy input, attention, and recon MSE.")
    parser.add_argument("--output", default="", help="Optional mp4 path or output directory for debug video.")
    return parser.parse_args()


def maybe_debug_frame(args, recorder, policy, policy_state, rgb, image, crop, action_n, step):
    if not (args.show_debug or recorder.enabled):
        return None, None

    recon_mse = action_reconstruction_mse(policy, policy_state, image, action_n)
    attention = None
    if args.show_debug and not getattr(args, "disable_attention", False):
        try:
            attention = encoder_attention_overlay(policy, policy_state, image)
        except Exception as exc:
            print(f"[debug] encoder attention disabled: {exc}")
            args.disable_attention = True

    frame = render_preview(rgb, image, crop, recorder.recon_values + [recon_mse], step, attention)
    recorder.record(frame, recon_mse)
    if args.show_debug and show_debug_window(frame):
        print("[debug] stopped by user")
        return recon_mse, True
    return recon_mse, False


def main():
    args = parse_args()
    variant_path = args.model_dir / "variant.json"
    variant = json.loads(variant_path.read_text()) if variant_path.exists() else {}
    stats = json.loads((args.model_dir / "normalization_stats.json").read_text())
    image_size = tuple(stats.get("image_size") or variant.get("image_size") or FALLBACK_IMAGE_SIZE)
    if stats.get("rgb_crop") is not None:
        crop = stats_crop_to_deploy_crop(stats["rgb_crop"])
    else:
        crop = tuple(variant.get("rgb_crop") or TRAIN_RGB_CROP)
    policy, device = load_latent_policy(args, variant, stats, image_size)
    print(f"[load] using checkpoint: {args.model_name}")

    state_mean = np.asarray(stats["state_mean"], dtype=np.float32)
    state_std = np.asarray(stats["state_std"], dtype=np.float32)
    action_mean = np.asarray(stats["action_mean"], dtype=np.float32)
    action_std = np.asarray(stats["action_std"], dtype=np.float32)
    normalize_state = bool(stats.get("normalize_proprio", variant.get("normalize_proprio", True)))
    normalize_action = bool(stats.get("normalize_action", variant.get("normalize_action", True)))

    recorder = DeployRecorder(args.output, fps=args.rate)
    recorder.print_targets()
    execute_robot = not args.dry_run
    arm = make_franka_arm() if execute_robot else None
    previous_action_n = None

    print(
        f"[ready] execute={execute_robot}, device={device}, image_size={image_size}, "
        f"crop={crop}, robot_ip={ROBOT.ip}, rate={args.rate}Hz. Ctrl-C stops deploy."
    )

    camera = RealSenseCamera()
    with SpaceKeyController() as keyboard:
        keyboard.wait_for_space()
        stop_by_space = False
        with camera:
            try:
                for step in range(args.steps):
                    if keyboard.stop_requested():
                        print("[keyboard] stopped by Space")
                        break
                    tic = time.time()
                    rgb = camera.read_rgb()
                    robot_state = read_robot_state(arm) if execute_robot else make_fake_state(OPEN_GRIPPER_WIDTH)
                    image = crop_and_resize_rgb(rgb, image_size, crop).astype(np.uint8)
                    policy_state = normalize(robot_state, state_mean, state_std) if normalize_state else robot_state

                    action_n, q1, q2 = policy.select_action(policy_state, image=image)
                    action_n = smooth_action(action_n, previous_action_n, ACTION_SMOOTHING)
                    previous_action_n = action_n.copy()

                    recon_mse, stop = maybe_debug_frame(args, recorder, policy, policy_state, rgb, image, crop, action_n, step)
                    action = unnormalize(action_n, action_mean, action_std) if normalize_action else action_n.astype(np.float32)
                    target_state = robot_state + action[:8]
                    translation = target_state[:3]
                    gripper_width = float(np.clip(target_state[7], CLOSED_GRIPPER_WIDTH, OPEN_GRIPPER_WIDTH))
                    gripper_delta = float(action[7])

                    print(
                        f"{step:04d} q=({q1:.3f},{q2:.3f}) trans={translation.round(4)} "
                        f"grip={gripper_width:.3f} state_grip={robot_state[7]:.3f} "
                        f"delta_grip={gripper_delta:.4f}"
                        + (f" recon_mse={recon_mse:.6f}" if recon_mse is not None else "")
                    )

                    if execute_robot:
                        move_robot_to_target_state(
                            arm,
                            target_state,
                            action,
                            open_width=OPEN_GRIPPER_WIDTH,
                            closed_width=CLOSED_GRIPPER_WIDTH,
                            deadband=GRIPPER_DEADBAND,
                        )
                    if stop:
                        break

                    stop_by_space = keyboard.sleep_until(tic + 1.0 / args.rate)
                    if stop_by_space:
                        print("[keyboard] stopped by Space")
                        break
            finally:
                with FinishCleanupBeforeCtrlC():
                    close_debug_windows()
                    recorder.save()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("[exit] deploy stopped by Ctrl-C")
