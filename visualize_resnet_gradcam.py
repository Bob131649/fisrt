#!/usr/bin/env python3
import argparse
import json
import os
import pickle
from typing import Dict, Optional, Tuple

import cv2
import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from networks.image_encoder_nets import ResnetEncoder


IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def find_default_h5(data_root: str) -> str:
    for root, _, files in os.walk(data_root):
        for name in sorted(files):
            if name.endswith(".h5") or name.endswith(".hdf5"):
                return os.path.join(root, name)
    raise FileNotFoundError(f"No h5 file found under {data_root}")


def load_crop_width(stats_path: Optional[str], default_width: int = 200) -> int:
    if not stats_path or not os.path.exists(stats_path):
        return default_width

    try:
        with open(stats_path, "rb") as f:
            stats = pickle.load(f)
        return int(stats.get("cell_width", default_width))
    except Exception as exc:
        print(f"Warning: failed to load stats from {stats_path}: {exc}")
        return default_width


def load_model_config(model_dir: str) -> Dict:
    variant_path = os.path.join(model_dir, "variant.json")
    if not os.path.exists(variant_path):
        return {"img_feature_dim": 8}

    with open(variant_path, "r") as f:
        return json.load(f)


def preprocess_image(rgb: np.ndarray, crop_width: int) -> Tuple[np.ndarray, torch.Tensor]:
    center_x = rgb.shape[1] // 2
    cropped = rgb[:300, center_x - crop_width:center_x + crop_width, :]
    resized = cv2.resize(cropped, (224, 224), interpolation=cv2.INTER_LINEAR)

    image_float = resized.astype(np.float32) / 255.0
    normalized = (image_float - IMAGENET_MEAN) / IMAGENET_STD
    tensor = torch.from_numpy(normalized).permute(2, 0, 1).unsqueeze(0)
    return image_float, tensor


def overlay_heatmap(image: np.ndarray, heatmap: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    heatmap_uint8 = np.uint8(255 * heatmap)
    heatmap_bgr = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    heatmap_rgb = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    blended = np.clip((1.0 - alpha) * image + alpha * heatmap_rgb, 0.0, 1.0)
    return blended


def make_panel(original: np.ndarray, gradcam: np.ndarray, activation: np.ndarray) -> np.ndarray:
    overlay = overlay_heatmap(original, gradcam)
    activation_overlay = overlay_heatmap(original, activation)

    gradcam_rgb = cv2.applyColorMap(np.uint8(255 * gradcam), cv2.COLORMAP_JET)
    gradcam_rgb = cv2.cvtColor(gradcam_rgb, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

    top = np.concatenate([original, gradcam_rgb], axis=1)
    bottom = np.concatenate([overlay, activation_overlay], axis=1)
    panel = np.concatenate([top, bottom], axis=0)
    return np.uint8(np.clip(panel * 255.0, 0, 255))


def compute_gradcam(encoder: ResnetEncoder, image_tensor: torch.Tensor, target_dim: Optional[int]) -> Tuple[np.ndarray, np.ndarray]:
    activations = {}
    gradients = {}

    def forward_hook(_, __, output):
        activations["value"] = output

    def backward_hook(_, grad_input, grad_output):
        del grad_input
        gradients["value"] = grad_output[0]

    target_layer = encoder.feature_extractor[-1]
    handle_fwd = target_layer.register_forward_hook(forward_hook)
    handle_bwd = target_layer.register_full_backward_hook(backward_hook)

    try:
        encoder.zero_grad(set_to_none=True)
        embedding = encoder(image_tensor)

        if target_dim is None:
            score = embedding.pow(2).sum()
        else:
            score = embedding[:, target_dim].sum()

        score.backward()

        acts = activations["value"][0]
        grads = gradients["value"][0]

        weights = grads.mean(dim=(1, 2), keepdim=True)
        cam = torch.relu((weights * acts).sum(dim=0))
        cam = cam.detach().cpu().numpy()
        cam = cv2.resize(cam, (224, 224), interpolation=cv2.INTER_LINEAR)
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)

        mean_activation = acts.detach().mean(dim=0).cpu().numpy()
        mean_activation = mean_activation - mean_activation.min()
        mean_activation = mean_activation / (mean_activation.max() + 1e-8)
        mean_activation = cv2.resize(mean_activation, (224, 224), interpolation=cv2.INTER_LINEAR)
        return cam, mean_activation
    finally:
        handle_fwd.remove()
        handle_bwd.remove()


def save_figure(original: np.ndarray, gradcam: np.ndarray, activation: np.ndarray, save_path: str) -> None:
    overlay = overlay_heatmap(original, gradcam)
    activation_overlay = overlay_heatmap(original, activation)

    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    axes[0, 0].imshow(original)
    axes[0, 0].set_title("Input")
    axes[0, 1].imshow(gradcam, cmap="jet")
    axes[0, 1].set_title("Grad-CAM")
    axes[1, 0].imshow(overlay)
    axes[1, 0].set_title("Grad-CAM Overlay")
    axes[1, 1].imshow(activation_overlay)
    axes[1, 1].set_title("Mean Feature Activation")

    for ax in axes.flat:
        ax.axis("off")

    fig.tight_layout()
    fig.savefig(save_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def maybe_open_writer(save_path: Optional[str], frame_size: Tuple[int, int], fps: int):
    if not save_path:
        return None

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(save_path, fourcc, fps, frame_size)
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open video writer for {save_path}")
    return writer


def play_h5(
    encoder: ResnetEncoder,
    data_file: str,
    crop_width: int,
    target_dim: Optional[int],
    device: str,
    fps: int,
    start_frame: int,
    max_frames: int,
    save_video_path: Optional[str],
    show_window: bool,
) -> None:
    writer = None
    window_name = "GradCAM Playback"
    paused = True

    try:
        with h5py.File(data_file, "r") as f:
            total_frames = len(f["rgb"])
            start = int(np.clip(start_frame, 0, total_frames - 1))
            end = total_frames if max_frames <= 0 else min(total_frames, start + max_frames)
            frame_idx = start
            last_panel_bgr = None

            while frame_idx < end:
                raw_rgb = f["rgb"][frame_idx]
                original, image_tensor = preprocess_image(raw_rgb, crop_width)
                image_tensor = image_tensor.to(device)
                image_tensor.requires_grad_(True)

                with torch.enable_grad():
                    gradcam, activation = compute_gradcam(encoder, image_tensor, target_dim)

                panel_rgb = make_panel(original, gradcam, activation)
                panel_bgr = cv2.cvtColor(panel_rgb, cv2.COLOR_RGB2BGR)
                last_panel_bgr = panel_bgr

                if writer is None and save_video_path:
                    height, width = panel_bgr.shape[:2]
                    writer = maybe_open_writer(save_video_path, (width, height), fps)

                if writer is not None and not paused:
                    writer.write(panel_bgr)

                if show_window:
                    while True:
                        display_frame = last_panel_bgr.copy()
                        status = "Paused" if paused else "Playing"
                        help_text = "Space: pause/resume | m: prev | n: next | q/ESC: quit"
                        cv2.putText(display_frame, f"{status}  frame {frame_idx}/{end - 1}", (10, 25),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
                        cv2.putText(display_frame, help_text, (10, display_frame.shape[0] - 12),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
                        cv2.imshow(window_name, display_frame)

                        wait_ms = 0 if paused else max(1, int(1000 / max(fps, 1)))
                        key = cv2.waitKey(wait_ms) & 0xFF

                        if key == 255 and not paused:
                            frame_idx += 1
                            break
                        if key == ord(" "):
                            paused = not paused
                            continue
                        if key == ord("m"):
                            frame_idx = max(start, frame_idx - 1)
                            paused = True
                            break
                        if key == ord("n"):
                            frame_idx = min(end - 1, frame_idx + 1)
                            paused = True
                            break
                        if key == 27 or key == ord("q"):
                            return
                        if key == 255 and paused:
                            continue
                else:
                    frame_idx += 1

                if frame_idx % 10 == 0:
                    print(f"Processed frame {frame_idx}/{end - 1}")

            if show_window and last_panel_bgr is not None:
                while True:
                    display_frame = last_panel_bgr.copy()
                    cv2.putText(display_frame, "Finished  Press q/ESC to quit", (10, 25),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
                    cv2.putText(display_frame, "Playback reached the last frame", (10, display_frame.shape[0] - 12),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
                    cv2.imshow(window_name, display_frame)
                    key = cv2.waitKey(0) & 0xFF
                    if key == 27 or key == ord("q") or key == ord(" "):
                        return
    finally:
        if writer is not None:
            writer.release()
        if show_window:
            cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description="Visualize what a trained ResNet encoder attends to with Grad-CAM.")
    parser.add_argument("--model_dir", default="results/Exp0680/datr", type=str)
    parser.add_argument("--model_name", default="model_vae.pth", type=str)
    parser.add_argument("--stats_path", default=None, type=str)
    parser.add_argument("--data_file", default=None, type=str)
    parser.add_argument("--data_root", default="dataset", type=str)
    parser.add_argument("--frame_idx", default=0, type=int)
    parser.add_argument("--target_dim", default=None, type=int)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", type=str)
    parser.add_argument("--output_dir", default="viz/gradcam", type=str)
    parser.add_argument("--play_h5", action="store_true")
    parser.add_argument("--show_window", action="store_true")
    parser.add_argument("--fps", default=8, type=int)
    parser.add_argument("--start_frame", default=0, type=int)
    parser.add_argument("--max_frames", default=-1, type=int)
    parser.add_argument("--save_video", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    data_file = args.data_file or find_default_h5(args.data_root)
    stats_path = args.stats_path or os.path.join(args.model_dir, "stats.pkl")
    crop_width = load_crop_width(stats_path)

    model_cfg = load_model_config(args.model_dir)
    img_feature_dim = int(model_cfg.get("img_feature_dim", 8))

    encoder = ResnetEncoder(backbone="resnet18", output_dim=img_feature_dim, pretrained=False)
    checkpoint_path = args.model_name
    if not os.path.isabs(checkpoint_path):
        checkpoint_path = os.path.join(args.model_dir, checkpoint_path)

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    encoder.load_state_dict(checkpoint["img_encoder"])
    encoder.eval()
    encoder.to(args.device)

    with h5py.File(data_file, "r") as f:
        total_frames = len(f["rgb"])
        frame_idx = int(np.clip(args.frame_idx, 0, total_frames - 1))
        raw_rgb = f["rgb"][frame_idx]

    stem = os.path.splitext(os.path.basename(data_file))[0]
    target_tag = "norm" if args.target_dim is None else f"dim{args.target_dim}"

    if args.play_h5:
        video_path = None
        if args.save_video:
            video_path = os.path.join(args.output_dir, f"{stem}_{target_tag}_playback.mp4")
        play_h5(
            encoder=encoder,
            data_file=data_file,
            crop_width=crop_width,
            target_dim=args.target_dim,
            device=args.device,
            fps=args.fps,
            start_frame=args.start_frame,
            max_frames=args.max_frames,
            save_video_path=video_path,
            show_window=args.show_window,
        )
        print(f"Data file: {data_file}")
        print(f"Crop width: {crop_width}")
        print(f"Target: {target_tag}")
        if video_path:
            print(f"Saved playback video to: {video_path}")
        return

    original, image_tensor = preprocess_image(raw_rgb, crop_width)
    image_tensor = image_tensor.to(args.device)
    image_tensor.requires_grad_(True)

    with torch.enable_grad():
        gradcam, activation = compute_gradcam(encoder, image_tensor, args.target_dim)

    save_path = os.path.join(args.output_dir, f"{stem}_frame{frame_idx:03d}_{target_tag}.png")
    save_figure(original, gradcam, activation, save_path)

    np.save(os.path.join(args.output_dir, f"{stem}_frame{frame_idx:03d}_{target_tag}_gradcam.npy"), gradcam)
    np.save(os.path.join(args.output_dir, f"{stem}_frame{frame_idx:03d}_{target_tag}_activation.npy"), activation)

    print(f"Saved visualization to: {save_path}")
    print(f"Data file: {data_file}")
    print(f"Frame index: {frame_idx}")
    print(f"Crop width: {crop_width}")
    print(f"Target: {target_tag}")


if __name__ == "__main__":
    main()
