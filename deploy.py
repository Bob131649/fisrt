#!/usr/bin/env python3

import argparse
import json
import time
from pathlib import Path

import algos.algos_v2_vision as algos
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pyrealsense2 as rs
import torch
import torch.nn.functional as F
import cv2


TRAIN_CROP_TOP = 0
TRAIN_CROP_LEFT = 90
TRAIN_CROP_HEIGHT = 360
TRAIN_CROP_WIDTH = 480


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


def preprocess_rgb(rgb, image_size, crop):
    if cv2 is None:
        raise ImportError("opencv-python is required for image resizing")
    top, left, height, width = crop
    bottom = top + height
    right = left + width
    frame_h, frame_w = rgb.shape[:2]
    if top < 0 or left < 0 or bottom > frame_h or right > frame_w:
        raise ValueError(
            "RGB frame is smaller than the train crop window: "
            f"frame={frame_w}x{frame_h}, crop left/top/width/height="
            f"{left}/{top}/{width}/{height}"
        )
    cropped = rgb[top:bottom, left:right]
    return cv2.resize(cropped, (image_size[1], image_size[0]), interpolation=cv2.INTER_LINEAR)


def render_rgb_preview(rgb, processed, crop):
    top, left, height, width = crop
    preview_rgb = rgb.copy()
    cv2.rectangle(preview_rgb, (left, top), (left + width, top + height), (0, 255, 0), 2)

    preview_bgr = cv2.cvtColor(preview_rgb, cv2.COLOR_RGB2BGR)
    processed_bgr = cv2.cvtColor(processed, cv2.COLOR_RGB2BGR)
    processed_bgr = cv2.resize(
        processed_bgr,
        (preview_bgr.shape[1], preview_bgr.shape[0]),
        interpolation=cv2.INTER_NEAREST,
    )

    canvas = np.hstack([preview_bgr, processed_bgr])
    cv2.putText(canvas, "raw rgb + crop box", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.putText(
        canvas,
        "policy input after crop+resize",
        (preview_bgr.shape[1] + 12, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return canvas


def state_from_robot(arm):
    trans, quat, _ = arm.get_ee_pose(frame="global")
    quat = np.asarray(quat, dtype=np.float32)
    quat /= max(np.linalg.norm(quat), 1e-6)
    return np.r_[trans, quat, arm.get_gripper_width()].astype(np.float32)


def norm(x, mean, std):
    return ((x - mean) / (std + 1e-6)).astype(np.float32)


def denorm(x, mean, std):
    return (x * (std + 1e-6) + mean).astype(np.float32)


def quaternion_distance_threshold(degrees):
    return 1.0 - np.cos(np.deg2rad(degrees) * 0.5)


def pose_error(target_state, current_state):
    d_error = np.linalg.norm(target_state[:3] - current_state[:3])
    q_error = 1.0 - abs(np.dot(target_state[3:7], current_state[3:7]))
    return d_error, q_error


def resolve_device(cli_device, variant_device):
    if cli_device:
        return cli_device
    if variant_device == "cuda" and not torch.cuda.is_available():
        return "cpu"
    return variant_device or ("cuda" if torch.cuda.is_available() else "cpu")


def unique_path(path: Path):
    """Return a non-existing path by appending _001, _002, ... if needed."""
    path = Path(path)
    if not path.exists():
        return path
    for idx in range(1, 10000):
        candidate = path.with_name(f"{path.stem}_{idx:03d}{path.suffix}")
        if not candidate.exists():
            return candidate
    raise RuntimeError(f"could not find an unused output path for {path}")


def resolve_output_paths(output):
    """Resolve --output into a video path and a matching recon-MSE plot path."""
    if not output:
        return None, None
    output_path = Path(output).expanduser()
    if output_path.suffix:
        video_path = unique_path(output_path)
        plot_path = unique_path(video_path.with_name(f"{video_path.stem}_recon_mse.png"))
    else:
        output_path.mkdir(parents=True, exist_ok=True)
        run_id = time.strftime("%Y%m%d_%H%M%S")
        video_path = unique_path(output_path / f"deploy_{run_id}.mp4")
        plot_path = unique_path(output_path / f"deploy_{run_id}_recon_mse.png")
    video_path.parent.mkdir(parents=True, exist_ok=True)
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    return video_path, plot_path


def draw_frame_counter(frame_bgr, frame_idx):
    """Draw the deploy step id in the top-left corner of a BGR frame."""
    text = f"frame {frame_idx}"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.55
    thickness = 1
    text_size, baseline = cv2.getTextSize(text, font, font_scale, thickness)
    pad = 6
    x, y = 10, 24
    overlay = frame_bgr.copy()
    cv2.rectangle(
        overlay,
        (x - pad, y - text_size[1] - pad),
        (x + text_size[0] + pad, y + baseline + pad),
        (0, 0, 0),
        -1,
    )
    cv2.addWeighted(overlay, 0.55, frame_bgr, 0.45, 0, frame_bgr)
    cv2.putText(frame_bgr, text, (x, y), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)


def make_record_frame(rgb, processed, crop, frame_idx):
    """Build one saved deploy visualization frame without touching the live preview window."""
    frame = render_rgb_preview(rgb, processed, crop)
    draw_frame_counter(frame, frame_idx)
    return frame


def save_video(frames, video_path, fps):
    """Write recorded deploy frames to an mp4 file."""
    if not frames or video_path is None:
        return
    height, width = frames[0].shape[:2]
    writer = cv2.VideoWriter(
        str(video_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        float(fps),
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"failed to open video writer: {video_path}")
    for frame in frames:
        writer.write(frame)
    writer.release()


def save_recon_plot(recon_values, plot_path):
    """Save a line plot of per-step VAE action reconstruction MSE."""
    if not recon_values or plot_path is None:
        return
    values = np.asarray(recon_values, dtype=np.float32)
    steps = np.arange(len(values))
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(steps, values, linewidth=2, color="tab:blue")
    ax.set_xlabel("Deploy Step")
    ax.set_ylabel("Recon MSE per action dim")
    ax.set_title(f"Deploy VAE recon MSE (mean={values.mean():.6f})")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(plot_path, dpi=180)
    plt.close(fig)


def compute_action_recon_mse(policy, policy_state, image, action_n):
    """Compute VAE recon MSE for the normalized action selected during deploy."""
    if policy.actor_vae is None:
        return float("nan")
    with torch.no_grad():
        state_t = torch.as_tensor(policy_state.reshape(1, -1), dtype=torch.float32, device=policy.device)
        image_t = torch.as_tensor(image, dtype=torch.float32, device=policy.device)
        if image_t.ndim == 3:
            image_t = image_t.unsqueeze(0)
        if image_t.shape[-1] == 3:
            image_t = image_t.permute(0, 3, 1, 2)
        if image_t.max() > 1:
            image_t = image_t / 255.0
        action_t = torch.as_tensor(action_n.reshape(1, -1), dtype=torch.float32, device=policy.device)
        feature = policy._obs_feature(state_t, image=image_t)
        recons_action, _, _ = policy.actor_vae(feature, action_t)
        recon_mse = F.mse_loss(recons_action, action_t, reduction="none").mean(dim=1)
    return float(recon_mse.item())


class DeployRecorder:
    """Optional deploy artifact recorder used by --output."""

    def __init__(self, output, fps):
        self.video_path, self.recon_plot_path = resolve_output_paths(output)
        self.fps = fps
        self.frames = []
        self.recon_values = []

    @property
    def enabled(self):
        return self.video_path is not None

    def print_targets(self):
        """Print where output artifacts will be written."""
        if not self.enabled:
            return
        print(f"[output] deploy video: {self.video_path}")
        print(f"[output] recon MSE plot: {self.recon_plot_path}")

    def record(self, step, rgb, processed, crop, recon_mse):
        """Record one deploy step for later video and loss-curve saving."""
        if not self.enabled:
            return
        self.recon_values.append(recon_mse)
        self.frames.append(make_record_frame(rgb, processed, crop, step))

    def measure_and_record(self, step, policy, policy_state, rgb, processed, crop, action_n):
        """Compute recon MSE and record deploy artifacts when --output is enabled."""
        if not self.enabled:
            return None
        recon_mse = compute_action_recon_mse(policy, policy_state, processed, action_n)
        self.record(step, rgb, processed, crop, recon_mse)
        return recon_mse

    def save(self):
        """Save the recorded video and recon-MSE plot at shutdown."""
        if not self.enabled:
            return
        save_video(self.frames, self.video_path, fps=self.fps)
        save_recon_plot(self.recon_values, self.recon_plot_path)
        print(f"[saved] deploy video: {self.video_path}")
        print(f"[saved] recon MSE plot: {self.recon_plot_path}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model-dir", required=True, type=Path)
    p.add_argument("--model-name", default="model")
    p.add_argument("--load-best", action="store_true")
    p.add_argument("--execute", action="store_true", default= True)

    p.add_argument("--robot-ip", default="192.168.31.12")
    p.add_argument("--arm", choices=["left", "right", "basic"], default="basic")
    p.add_argument("--control-mode", choices=["curobo", "armlib"], default="armlib")
    p.add_argument("--relative-dynamics-factor", type=float, default=0.04)

    p.add_argument("--camera-serial", default="230422272989")
    p.add_argument("--camera-width", type=int, default=640)
    p.add_argument("--camera-height", type=int, default=480)
    p.add_argument("--camera-fps", type=int, default=30)

    p.add_argument("--rate", type=float, default=5.0)
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--image-size", nargs=2, type=int, default=None)
    p.add_argument(
        "--crop",
        nargs=4,
        type=int,
        default=None,
        metavar=("TOP", "LEFT", "HEIGHT", "WIDTH"),
        help="RGB crop before resize; defaults to train_vision crop.",
    )
    p.add_argument("--device", default=None)
    p.add_argument("--vae", action="store_true")
    p.add_argument("--show-rgb", action="store_true")
    p.add_argument("--visualization", action="store_true")
    p.add_argument(
        "--output",
        default="",
        help="Optional output mp4 path or directory. Saves deploy visualization video and recon MSE plot.",
    )

    p.add_argument("--open-width", type=float, default=0.04)
    p.add_argument("--closed-width", type=float, default=0.0)
    p.add_argument("--d-threshold", type=float, default=0.01)
    p.add_argument("--q-threshold", type=float, default=quaternion_distance_threshold(1.0))
    p.add_argument("--max-ext-count", type=int, default=10)
    return p.parse_args()


def main():
    args = parse_args()
    variant = load_variant(args.model_dir)
    cli_image_size = args.image_size
    recorder = DeployRecorder(args.output, fps=args.rate)

    stats_meta = json.loads((args.model_dir / "normalization_stats.json").read_text())
    state_mean = np.asarray(stats_meta["state_mean"], dtype=np.float32)
    state_std = np.asarray(stats_meta["state_std"], dtype=np.float32)
    action_mean = np.asarray(stats_meta["action_mean"], dtype=np.float32)
    action_std = np.asarray(stats_meta["action_std"], dtype=np.float32)
    args.image_size = tuple(cli_image_size or stats_meta.get("image_size") or variant.get("image_size") or (224, 224))
    args.crop = tuple(args.crop or stats_meta.get("rgb_crop") or variant.get("rgb_crop") or (
        TRAIN_CROP_TOP,
        TRAIN_CROP_LEFT,
        TRAIN_CROP_HEIGHT,
        TRAIN_CROP_WIDTH,
    ))
    print(f"[stats] loaded normalization_stats.json from {args.model_dir}")
    device = resolve_device(args.device, variant.get("device"))
    state_dim = int(stats_meta.get("state_dim", variant.get("state_dim", len(state_mean))))
    action_dim = int(stats_meta.get("action_dim", variant.get("action_dim", len(action_mean))))
    normalize_proprio = bool(stats_meta.get("normalize_proprio", variant.get("normalize_proprio", True)))
    normalize_action = bool(stats_meta.get("normalize_action", variant.get("normalize_action", True)))

    if len(state_mean) != state_dim or len(state_std) != state_dim:
        raise ValueError(f"state stats dim mismatch: state_dim={state_dim}, mean={len(state_mean)}, std={len(state_std)}")
    if len(action_mean) != action_dim or len(action_std) != action_dim:
        raise ValueError(
            f"action stats dim mismatch: action_dim={action_dim}, mean={len(action_mean)}, std={len(action_std)}"
        )

    policy = algos.Latent(
        state_dim,
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
        encoder_mode=variant.get("encoder_mode", "robomimic"),
        concat_mode=variant.get("concat_mode", "default"),
        zipper_backbone=variant.get("zipper_backbone", "resnet18"),
        zipper_normalize_image=bool(variant.get("zipper_normalize_image", True)),
        max_latent_action=float(variant.get("max_latent_action", 0.675)),
        expectile=float(variant.get("expectile", 0.9)),
        kl_beta=float(variant.get("kl_beta", 1.0)),
        doubleq_min=float(variant.get("doubleq_min", 1.0)),
        vae=bool(variant.get("vae", args.vae)),
    )
    policy.load(args.model_name, str(args.model_dir), load_best=args.load_best)
    policy.eval()

    arm = make_arm(args) if args.execute else None
    print(
        f"[ready] execute={args.execute}, action_type=threshold_delta_eef, "
        f"device={device}, image_size={args.image_size}, crop={args.crop}. Press Ctrl-C to stop."
    )
    show_rgb = args.show_rgb or args.visualization

    if show_rgb:
        print("[preview] showing raw RGB and processed policy input. Press 'q' in the window to stop.")
    recorder.print_targets()

    if args.execute:
        current_state = state_from_robot(arm)
    else:
        current_state = np.r_[np.zeros(3), [0, 0, 0, 1], args.open_width].astype(np.float32)
    target_state = current_state.copy()
    ext_count = 0

    with Camera(args.camera_serial, args.camera_width, args.camera_height, args.camera_fps) as cam:
        try:
            for step in range(args.steps):
                tic = time.time()
                if args.execute:
                    state = state_from_robot(arm)
                else:
                    state = target_state.copy()

                d_error, q_error = pose_error(target_state, state)
                should_select = (d_error < args.d_threshold and q_error < args.q_threshold) or ext_count > args.max_ext_count
                print(
                    f"---------- {ext_count} d={d_error:.5f} q={q_error:.6f} "
                    f"grip_target={target_state[7]:.3f} "
                    f"ready={d_error < args.d_threshold},{q_error < args.q_threshold}"
                )

                if should_select:
                    if args.execute:
                        arm.set_gripper_opening(float(target_state[7]), asynchronous=True)

                    rgb = cam.rgb()
                    image = preprocess_rgb(rgb, args.image_size, args.crop)
                    if show_rgb:
                        preview = render_rgb_preview(rgb, image, args.crop)
                        cv2.imshow("deploy_rgb_preview", preview)
                        key = cv2.waitKey(1) & 0xFF
                        if key == ord("q"):
                            print("[preview] stopped by user")
                            break

                    policy_state = norm(state, state_mean, state_std) if normalize_proprio else state.astype(np.float32)
                    action_n, q1, q2 = policy.select_action(policy_state, image=image)
                    recon_mse = recorder.measure_and_record(step, policy, policy_state, rgb, image, args.crop, action_n)
                    action = denorm(action_n, action_mean, action_std) if normalize_action else action_n.astype(np.float32)
                    target_state = state + action[:8]
                    target_state[3:7] /= max(np.linalg.norm(target_state[3:7]), 1e-6)
                    target_state[7] = float(np.clip(target_state[7], args.closed_width, args.open_width))

                    print(
                        f"{step:04d} q=({q1:.3f},{q2:.3f}) trans={target_state[:3].round(4)} "
                        f"grip={target_state[7]:.3f} state_grip={state[7]:.3f} "
                        f"action_grip_delta={float(action[7]):.4f}"
                        + (f" recon_mse={recon_mse:.6f}" if recon_mse is not None else "")
                    )
                    ext_count = 0

                if args.execute:
                    arm.set_ee_pose(target_state[:3], target_state[3:7], asynchronous=True, frame="global")

                sleep = 1.0 / args.rate - (time.time() - tic)
                if sleep > 0:
                    time.sleep(sleep)
                ext_count += 1
        finally:
            if show_rgb:
                cv2.destroyAllWindows()
            recorder.save()


if __name__ == "__main__":
    main()
