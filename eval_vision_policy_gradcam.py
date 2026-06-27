#!/usr/bin/env python3

import argparse
import json
import os
from typing import Optional, Tuple

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data._utils.collate import default_collate

import algos.algos_v2_vision as algos
from dataset.franka_dataset import FrankaImageDataset
from eval_vision_policy_losses import (
    apply_normalization_stats,
    build_episode_indices,
    chain_threshold_episode_indices,
    load_json_if_exists,
    str2bool,
)


def image_tensor_to_numpy(image_tensor: torch.Tensor) -> np.ndarray:
    image = image_tensor.detach().cpu().float()
    if image.max() > 1:
        image = image / 255.0
    image = image.permute(1, 2, 0).numpy()
    return np.clip(image, 0.0, 1.0)


def normalize_heatmap(heatmap: np.ndarray) -> np.ndarray:
    heatmap = np.asarray(heatmap, dtype=np.float32)
    heatmap = np.maximum(heatmap, 0.0)
    heatmap = heatmap - heatmap.min()
    denom = heatmap.max()
    if denom > 1e-8:
        heatmap = heatmap / denom
    return heatmap


def overlay_heatmap(image: np.ndarray, heatmap: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    heatmap_uint8 = np.uint8(255 * normalize_heatmap(heatmap))
    heatmap_bgr = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    heatmap_rgb = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return np.clip((1.0 - alpha) * image + alpha * heatmap_rgb, 0.0, 1.0)


def apply_erase_to_batch(batch, erase_rect, fill_mode: str = "mean"):
    erased = {
        key: value.clone() if torch.is_tensor(value) else value
        for key, value in batch.items()
    }
    image = erased["image"].clone()
    _, _, height, width = image.shape
    x1, y1, x2, y2 = erase_rect
    x1 = max(0, min(width, int(x1)))
    x2 = max(0, min(width, int(x2)))
    y1 = max(0, min(height, int(y1)))
    y2 = max(0, min(height, int(y2)))
    if x2 <= x1 or y2 <= y1:
        raise ValueError(f"invalid erase_rect after clipping: {(x1, y1, x2, y2)}")

    patch = image[:, :, y1:y2, x1:x2]
    if fill_mode == "zero":
        fill = torch.zeros_like(patch)
    elif fill_mode == "random":
        if image.dtype == torch.uint8:
            fill = torch.randint(0, 256, patch.shape, dtype=image.dtype, device=image.device)
        else:
            fill = torch.rand_like(patch)
            if image.max() > 1:
                fill = fill * 255.0
    else:
        fill = image.float().mean(dim=(2, 3), keepdim=True).to(dtype=image.dtype)
    image[:, :, y1:y2, x1:x2] = fill
    erased["image"] = image
    return erased


def add_title_bar_rgb(image: np.ndarray, title: str, title_height: int = 34) -> np.ndarray:
    image_u8 = np.uint8(np.clip(image * 255.0, 0, 255))
    title_bar = np.full((title_height, image_u8.shape[1], 3), 255, dtype=np.uint8)
    text_size, _ = cv2.getTextSize(title, cv2.FONT_HERSHEY_SIMPLEX, 0.68, 2)
    text_x = max(0, (image_u8.shape[1] - text_size[0]) // 2)
    text_y = (title_height + text_size[1]) // 2 - 2
    cv2.putText(
        title_bar,
        title,
        (text_x, text_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.68,
        (0, 0, 0),
        2,
        cv2.LINE_AA,
    )
    return np.concatenate([title_bar, image_u8], axis=0).astype(np.float32) / 255.0


def make_panel(
    original: np.ndarray,
    gradcam: np.ndarray,
    activation: np.ndarray,
    lines=None,
    alpha: float = 0.45,
) -> np.ndarray:
    overlay = overlay_heatmap(original, gradcam, alpha=alpha)
    activation_overlay = overlay_heatmap(original, activation, alpha=alpha)
    gradcam_rgb = cv2.applyColorMap(np.uint8(255 * normalize_heatmap(gradcam)), cv2.COLORMAP_JET)
    gradcam_rgb = cv2.cvtColor(gradcam_rgb, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

    top = np.concatenate(
        [
            add_title_bar_rgb(original, "Policy Input"),
            add_title_bar_rgb(gradcam_rgb, "ResNet Grad-CAM"),
        ],
        axis=1,
    )
    bottom = np.concatenate(
        [
            add_title_bar_rgb(overlay, "Grad-CAM Overlay"),
            add_title_bar_rgb(activation_overlay, "Mean Feature Activation"),
        ],
        axis=1,
    )
    panel = np.concatenate([top, bottom], axis=0)
    panel_bgr = cv2.cvtColor(np.uint8(np.clip(panel * 255.0, 0, 255)), cv2.COLOR_RGB2BGR)

    if lines:
        y = 24
        for line in lines:
            cv2.putText(
                panel_bgr,
                str(line),
                (10, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                panel_bgr,
                str(line),
                (10, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 0, 0),
                1,
                cv2.LINE_AA,
            )
            y += 20
    return panel_bgr


def save_figure(original: np.ndarray, gradcam: np.ndarray, activation: np.ndarray, save_path: str) -> None:
    overlay = overlay_heatmap(original, gradcam)
    activation_overlay = overlay_heatmap(original, activation)

    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    axes[0, 0].imshow(original)
    axes[0, 0].set_title("Policy Input")
    axes[0, 1].imshow(gradcam, cmap="jet")
    axes[0, 1].set_title("ResNet Grad-CAM")
    axes[1, 0].imshow(overlay)
    axes[1, 0].set_title("Grad-CAM Overlay")
    axes[1, 1].imshow(activation_overlay)
    axes[1, 1].set_title("Mean Feature Activation")
    for ax in axes.flat:
        ax.axis("off")
    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    fig.savefig(save_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def build_policy_and_dataset(args):
    model_dir = os.path.expanduser(args.model_dir)
    variant = load_json_if_exists(os.path.join(model_dir, "variant.json"))
    stats = load_json_if_exists(os.path.join(model_dir, "normalization_stats.json"))

    image_size = tuple(args.image_size or stats.get("image_size") or variant.get("image_size", [224, 224]))
    normalize_proprio = (
        args.normalize_proprio
        if args.normalize_proprio is not None
        else bool(stats.get("normalize_proprio", variant.get("normalize_proprio", True)))
    )
    normalize_action = (
        args.normalize_action
        if args.normalize_action is not None
        else bool(stats.get("normalize_action", variant.get("normalize_action", True)))
    )

    dataset = FrankaImageDataset(
        args.dataset_path,
        image_size=image_size,
        reward_scale=args.reward_scale,
        success_reward=args.success_reward,
        normalize_proprio=normalize_proprio,
        normalize_action=normalize_action,
    )
    apply_normalization_stats(dataset, stats)

    state_dim = int(dataset.state_dim)
    action_dim = int(dataset.action_dim)
    latent_dim = int(action_dim * 2)
    image_shape = (3, image_size[0], image_size[1])

    policy = algos.Latent(
        state_dim,
        action_dim,
        latent_dim,
        min_v=0.0,
        max_v=float(max(dataset.rewards.max(), args.reward_scale * args.success_reward, 1.0)),
        device=args.device,
        discount=float(args.discount if args.discount is not None else variant.get("discount", 0.99)),
        tau=float(args.tau if args.tau is not None else variant.get("tau", 0.005)),
        vae_lr=float(args.vae_lr if args.vae_lr is not None else variant.get("vae_lr", 2e-4)),
        actor_lr=float(args.actor_lr if args.actor_lr is not None else variant.get("actor_lr", 2e-4)),
        critic_lr=float(args.critic_lr if args.critic_lr is not None else variant.get("critic_lr", 2e-4)),
        obs_encoder_lr=args.obs_encoder_lr if args.obs_encoder_lr is not None else variant.get("obs_encoder_lr", None),
        max_latent_action=float(
            args.max_latent_action
            if args.max_latent_action is not None
            else variant.get("max_latent_action", 0.675)
        ),
        expectile=float(args.expectile if args.expectile is not None else variant.get("expectile", 0.9)),
        kl_beta=float(args.kl_beta if args.kl_beta is not None else variant.get("kl_beta", 1.0)),
        doubleq_min=float(args.doubleq_min if args.doubleq_min is not None else variant.get("doubleq_min", 1.0)),
        image_shape=image_shape,
        robomimic_feature_dim=int(
            args.robomimic_feature_dim
            if args.robomimic_feature_dim is not None
            else variant.get("robomimic_feature_dim", 256)
        ),
        robomimic_crop_shape=tuple(args.robomimic_crop_shape)
        if args.robomimic_crop_shape is not None
        else tuple(variant["robomimic_crop_shape"])
        if variant.get("robomimic_crop_shape") is not None
        else None,
        robomimic_backbone_class=args.robomimic_backbone_class
        or variant.get("robomimic_backbone_class", "ResNet18Conv"),
        robomimic_pool_class=args.robomimic_pool_class
        or variant.get("robomimic_pool_class", "SpatialSoftmax"),
        encoder_mode=variant.get("encoder_mode", "robomimic"),
        concat_mode=variant.get("concat_mode", "default"),
        zipper_backbone=variant.get("zipper_backbone", "resnet18"),
        zipper_normalize_image=bool(variant.get("zipper_normalize_image", True)),
        vae=bool(variant.get("vae", False)),
    )
    policy.load(args.checkpoint_name, model_dir, load_best=args.load_best)
    policy.eval()
    return policy, dataset, model_dir


def get_resnet_gradcam_layer(policy):
    image_encoder = getattr(policy.obs_encoder, "image_encoder", None)
    feature_extractor = getattr(image_encoder, "feature_extractor", None)
    if feature_extractor is None:
        raise RuntimeError(
            "This Grad-CAM script expects eval policy to use the zipper ResNet+GAP encoder "
            "(obs_encoder.image_encoder.feature_extractor)."
        )
    return feature_extractor[-1]


def make_gradcam_score(policy, feature, action, target: str, target_dim: Optional[int]):
    if target == "feature_dim":
        if target_dim is None:
            raise ValueError("--target_dim is required when --target feature_dim")
        return feature[:, int(target_dim)].sum(), {}

    if target == "feature_norm":
        return feature.pow(2).sum(), {}

    policy_action = None
    if target in ("action_norm", "action_dim", "q_min", "q_mean", "q1", "q2"):
        if policy.ope_mode:
            policy_action = policy.target_policy.select_action_tensor(feature)
        else:
            latent_action = None if policy.vae else policy.actor(feature)
            policy_action = policy.actor_vae.decode(feature, z=latent_action)

        q1, q2 = policy.critic(feature, policy_action)
        metrics = {
            "policy_action_norm": float(policy_action.detach().pow(2).sum(dim=1).sqrt().cpu()[0]),
            "q1": float(q1.detach().cpu()[0]),
            "q2": float(q2.detach().cpu()[0]),
        }
        if target == "action_dim":
            if target_dim is None:
                raise ValueError("--target_dim is required when --target action_dim")
            score = policy_action[:, int(target_dim)].sum()
        elif target == "q_min":
            score = torch.min(q1, q2).mean()
        elif target == "q_mean":
            score = 0.5 * (q1 + q2).mean()
        elif target == "q1":
            score = q1.mean()
        elif target == "q2":
            score = q2.mean()
        else:
            score = policy_action.pow(2).sum(dim=1).mean()
        return score, metrics

    if policy.actor_vae is None:
        raise RuntimeError(f"--target {target} needs actor_vae, but this policy has no actor_vae.")

    recons_action, mu, log_var = policy.actor_vae(feature, action)
    recon_per_dim = F.mse_loss(recons_action, action, reduction="none")
    kl_per_dim = -0.5 * (1 + log_var - mu.pow(2) - log_var.exp())
    metrics = {
        "recon_mean": float(recon_per_dim.mean().detach().cpu()),
        "recon_sum": float(recon_per_dim.sum(dim=1).mean().detach().cpu()),
        "kl": float(kl_per_dim.sum(dim=1).mean().detach().cpu()),
    }

    if target == "recon_sum":
        score = recon_per_dim.sum(dim=1).mean()
    elif target == "kl":
        score = kl_per_dim.sum(dim=1).mean()
    elif target == "total":
        score = recon_per_dim.mean(dim=1).mean() + policy.kl_beta * kl_per_dim.sum(dim=1).mean()
    else:
        score = recon_per_dim.mean(dim=1).mean()
    return score, metrics


def compute_policy_gradcam(policy, batch, target: str, target_dim: Optional[int]) -> Tuple[np.ndarray, np.ndarray, dict]:
    target_layer = get_resnet_gradcam_layer(policy)
    activations = {}
    gradients = {}

    def forward_hook(_, __, output):
        activations["value"] = output

    def backward_hook(_, grad_input, grad_output):
        del grad_input
        gradients["value"] = grad_output[0]

    handle_fwd = target_layer.register_forward_hook(forward_hook)
    handle_bwd = target_layer.register_full_backward_hook(backward_hook)

    policy.zero_grad(set_to_none=True)
    try:
        image = batch["image"].to(policy.device)
        if image.dtype == torch.uint8:
            image = image.float() / 255.0
        else:
            image = image.float()
            if image.max() > 1:
                image = image / 255.0
        image.requires_grad_(True)

        state = batch["state"].to(policy.device).float()
        action = batch["action"].to(policy.device).float()
        feature = policy._obs_feature(state, image=image)
        score, metrics = make_gradcam_score(policy, feature, action, target, target_dim)
        score.backward()

        acts = activations["value"][0]
        grads = gradients["value"][0]
        weights = grads.mean(dim=(1, 2), keepdim=True)
        cam = torch.relu((weights * acts).sum(dim=0)).detach().cpu().numpy()
        cam = cv2.resize(cam, (int(image.shape[-1]), int(image.shape[-2])), interpolation=cv2.INTER_LINEAR)
        cam = normalize_heatmap(cam)

        mean_activation = acts.detach().mean(dim=0).cpu().numpy()
        mean_activation = cv2.resize(
            mean_activation,
            (int(image.shape[-1]), int(image.shape[-2])),
            interpolation=cv2.INTER_LINEAR,
        )
        mean_activation = normalize_heatmap(mean_activation)
        metrics.update(
            {
                "score": float(score.detach().cpu()),
                "feature_dim": int(feature.shape[1]),
                "feature_l2": float(feature.detach().pow(2).sum(dim=1).sqrt().cpu()[0]),
            }
        )
        return cam, mean_activation, metrics
    finally:
        handle_fwd.remove()
        handle_bwd.remove()


def pick_episode(dataset, episode_id, seed, chain_threshold):
    episodes = build_episode_indices(dataset)
    if not episodes:
        raise RuntimeError("No episode with at least 2 transitions was found in the dataset.")
    rng = np.random.default_rng(seed)
    if episode_id is None:
        episode_id = int(rng.integers(len(episodes)))
    if not 0 <= episode_id < len(episodes):
        raise IndexError(f"episode_id must be in [0, {len(episodes) - 1}], got {episode_id}.")
    episode_indices = episodes[episode_id]
    before_len = len(episode_indices)
    if chain_threshold:
        episode_indices = chain_threshold_episode_indices(dataset, episode_indices)
    return episode_id, episode_indices, before_len, len(episodes)


def save_video(frames, output_path, fps):
    if not frames:
        return
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    h, w = frames[0].shape[:2]
    writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), float(fps), (w, h))
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open video writer: {output_path}")
    for frame in frames:
        writer.write(frame)
    writer.release()


def show_window(frames, fps, window_scale=2.0, window_name="Eval Vision Policy GradCAM"):
    if not frames:
        return
    if not os.environ.get("DISPLAY"):
        print("DISPLAY is not set; skipped interactive playback window.")
        return

    idx = 0
    paused = True
    delay_ms = max(1, int(1000.0 / max(float(fps), 1e-6)))
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    while True:
        frame = frames[idx].copy()
        status = "Paused" if paused else "Playing"
        help_text = "Space: pause/play | m: prev | n: next | q/ESC: quit"
        cv2.putText(
            frame,
            f"{status}  frame {idx + 1}/{len(frames)}",
            (10, frame.shape[0] - 36),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            help_text,
            (10, frame.shape[0] - 12),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.50,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        display_frame = frame
        if window_scale != 1.0:
            display_frame = cv2.resize(
                frame,
                (
                    int(round(frame.shape[1] * float(window_scale))),
                    int(round(frame.shape[0] * float(window_scale))),
                ),
                interpolation=cv2.INTER_NEAREST,
            )
        cv2.imshow(window_name, display_frame)
        key = cv2.waitKey(0 if paused else delay_ms) & 0xFF

        if key in (27, ord("q")):
            break
        if key == ord(" "):
            paused = not paused
            continue
        if key == ord("m"):
            idx = max(0, idx - 1)
            paused = True
            continue
        if key == ord("n"):
            idx = min(len(frames) - 1, idx + 1)
            paused = True
            continue
        if not paused:
            idx = min(len(frames) - 1, idx + 1)
            if idx == len(frames) - 1:
                paused = True

    cv2.destroyWindow(window_name)


def main():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--model_dir", required=True, type=str)
    parser.add_argument("--checkpoint_name", default="model", type=str)
    parser.add_argument("--load_best", nargs="?", const=True, default=False, type=str2bool)
    parser.add_argument("--dataset_path", required=True, nargs="+", type=str)
    parser.add_argument("--output_dir", default="", type=str)
    parser.add_argument("--episode_id", default=None, type=int)
    parser.add_argument("--step_id", default=0, type=str)
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--batch_size", default=128, type=int)
    parser.add_argument("--device", default="cuda", type=str)
    parser.add_argument(
        "--target",
        default="action_norm",
        choices=[
            "action_norm",
            "action_dim",
            "q_min",
            "q_mean",
            "q1",
            "q2",
            "recon_mean",
            "recon_sum",
            "kl",
            "total",
            "feature_norm",
            "feature_dim",
        ],
        type=str,
        help="Backward target for Grad-CAM. action_norm follows the loaded policy's deploy action path.",
    )
    parser.add_argument("--target_dim", default=None, type=int)
    parser.add_argument("--overlay_alpha", default=0.45, type=float)
    parser.add_argument("--fps", default=10.0, type=float)
    parser.add_argument(
        "--erase_compare",
        action="store_true",
        help="Also run Grad-CAM after occluding --erase_rect and log clean-vs-erased metric deltas.",
    )
    parser.add_argument(
        "--erase_apply",
        action="store_true",
        help="Apply --erase_rect to the main Grad-CAM input and main video instead of only comparing.",
    )
    parser.add_argument(
        "--erase_rect",
        nargs=4,
        type=int,
        default=(120, 35, 180, 130),
        metavar=("X1", "Y1", "X2", "Y2"),
        help="Policy-input pixel rectangle to occlude when --erase_compare is set.",
    )
    parser.add_argument(
        "--erase_fill",
        default="mean",
        choices=["mean", "zero", "random"],
        type=str,
        help="Fill value for --erase_rect when --erase_compare is set.",
    )
    parser.add_argument(
        "--show_window",
        action="store_true",
        help="Open an interactive OpenCV playback window. Space pauses/plays, m/n move frames.",
    )
    parser.add_argument(
        "--window_scale",
        default=2.0,
        type=float,
        help="Scale factor for the interactive OpenCV window.",
    )
    parser.add_argument(
        "--save_images",
        action="store_true",
        help="Save per-frame PNG summaries. Disabled by default.",
    )
    parser.add_argument(
        "--save_npy",
        action="store_true",
        help="Save per-frame Grad-CAM and activation .npy arrays. Disabled by default.",
    )
    parser.add_argument(
        "--chain_threshold_transitions",
        nargs="?",
        const=True,
        default=True,
        type=str2bool,
    )

    parser.add_argument("--image_size", nargs=2, default=None, type=int)
    parser.add_argument("--reward_scale", default=100.0, type=float)
    parser.add_argument("--success_reward", default=1.0, type=float)
    parser.add_argument("--normalize_proprio", nargs="?", const=True, default=None, type=str2bool)
    parser.add_argument("--normalize_action", nargs="?", const=True, default=None, type=str2bool)
    parser.add_argument("--robomimic_feature_dim", default=None, type=int)
    parser.add_argument("--robomimic_crop_shape", nargs=2, default=None, type=int)
    parser.add_argument("--robomimic_backbone_class", default=None, type=str)
    parser.add_argument("--robomimic_pool_class", default=None, type=str)
    parser.add_argument("--max_latent_action", default=None, type=float)
    parser.add_argument("--discount", default=None, type=float)
    parser.add_argument("--tau", default=None, type=float)
    parser.add_argument("--vae_lr", default=None, type=float)
    parser.add_argument("--actor_lr", default=None, type=float)
    parser.add_argument("--critic_lr", default=None, type=float)
    parser.add_argument("--obs_encoder_lr", default=None, type=float)
    parser.add_argument("--expectile", default=None, type=float)
    parser.add_argument("--kl_beta", default=None, type=float)
    parser.add_argument("--doubleq_min", default=None, type=float)
    args = parser.parse_args()

    policy, dataset, model_dir = build_policy_and_dataset(args)
    output_dir = args.output_dir or os.path.join(model_dir, "eval_policy_gradcam")
    os.makedirs(output_dir, exist_ok=True)

    episode_id, episode_indices, before_len, num_episodes = pick_episode(
        dataset,
        args.episode_id,
        args.seed,
        args.chain_threshold_transitions,
    )
    if not episode_indices:
        raise RuntimeError("Selected episode is empty after threshold chaining.")

    all_steps = str(args.step_id).lower() == "all"
    if all_steps:
        selected = list(enumerate(episode_indices))
    else:
        step_id = int(args.step_id)
        if not 0 <= step_id < len(episode_indices):
            raise IndexError(f"step_id must be in [0, {len(episode_indices) - 1}], got {step_id}.")
        selected = [(step_id, episode_indices[step_id])]

    frames = []
    erased_frames = []
    rows = []
    first_png_path = None
    for step_id, dataset_idx in selected:
        sample = dataset.get_item(int(dataset_idx), augment=False)
        batch = default_collate([sample])
        batch_for_main = (
            apply_erase_to_batch(batch, args.erase_rect, fill_mode=args.erase_fill)
            if args.erase_apply
            else batch
        )
        gradcam, activation, metrics = compute_policy_gradcam(
            policy,
            batch_for_main,
            args.target,
            args.target_dim,
        )
        erased_metrics = None
        if args.erase_compare:
            erased_batch = (
                batch_for_main
                if args.erase_apply
                else apply_erase_to_batch(batch, args.erase_rect, fill_mode=args.erase_fill)
            )
            erased_gradcam, erased_activation, erased_metrics = compute_policy_gradcam(
                policy,
                erased_batch,
                args.target,
                args.target_dim,
            )
        original = image_tensor_to_numpy(batch_for_main["image"][0])
        erased_original = (
            image_tensor_to_numpy(erased_batch["image"][0])
            if args.erase_compare
            else None
        )
        raw_idx = int(dataset.indices[int(dataset_idx)])
        next_raw_idx = int(dataset.next_indices[int(dataset_idx)])
        target_tag = args.target if args.target_dim is None else f"{args.target}_dim{args.target_dim}"
        stem = f"episode_{episode_id:04d}_step_{step_id:04d}_raw_{raw_idx:06d}_{target_tag}"

        png_path = os.path.join(output_dir, f"{stem}.png")
        if args.save_images:
            save_figure(original, gradcam, activation, png_path)
        if args.save_npy:
            np.save(os.path.join(output_dir, f"{stem}_gradcam.npy"), gradcam)
            np.save(os.path.join(output_dir, f"{stem}_activation.npy"), activation)
            if args.erase_compare:
                np.save(os.path.join(output_dir, f"{stem}_erase_gradcam.npy"), erased_gradcam)
                np.save(os.path.join(output_dir, f"{stem}_erase_activation.npy"), erased_activation)
        if args.save_images and first_png_path is None:
            first_png_path = png_path

        lines = [
            f"episode {episode_id} step {step_id + 1}/{len(episode_indices)} raw {raw_idx}->{next_raw_idx}",
            f"target {target_tag} score {metrics['score']:.4f}",
        ]
        if "recon_mean" in metrics:
            lines.append(f"recon {metrics['recon_mean']:.4f} kl {metrics['kl']:.4f}")
        if "policy_action_norm" in metrics:
            lines.append(
                f"action_norm {metrics['policy_action_norm']:.4f} "
                f"q1 {metrics['q1']:.4f} q2 {metrics['q2']:.4f}"
            )
        frames.append(make_panel(original, gradcam, activation, lines=lines, alpha=args.overlay_alpha))
        if args.erase_compare:
            erased_lines = [
                f"episode {episode_id} step {step_id + 1}/{len(episode_indices)} raw {raw_idx}->{next_raw_idx}",
                f"erase {tuple(args.erase_rect)} fill {args.erase_fill}",
                f"target {target_tag} score {erased_metrics['score']:.4f}",
                f"delta score {erased_metrics['score'] - metrics['score']:+.4f}",
            ]
            if "policy_action_norm" in erased_metrics:
                erased_lines.append(
                    f"action_norm {erased_metrics['policy_action_norm']:.4f} "
                    f"delta {erased_metrics['policy_action_norm'] - metrics['policy_action_norm']:+.4f}"
                )
            if "recon_mean" in erased_metrics:
                erased_lines.append(
                    f"recon {erased_metrics['recon_mean']:.4f} "
                    f"delta {erased_metrics['recon_mean'] - metrics['recon_mean']:+.4f}"
                )
            erased_frames.append(
                make_panel(
                    erased_original,
                    erased_gradcam,
                    erased_activation,
                    lines=erased_lines,
                    alpha=args.overlay_alpha,
                )
            )
        rows.append(
            [
                step_id,
                int(dataset_idx),
                raw_idx,
                next_raw_idx,
                metrics.get("score", np.nan),
                metrics.get("recon_mean", np.nan),
                metrics.get("recon_sum", np.nan),
                metrics.get("kl", np.nan),
                metrics.get("feature_l2", np.nan),
                erased_metrics.get("score", np.nan) if erased_metrics is not None else np.nan,
                (
                    erased_metrics.get("score", np.nan) - metrics.get("score", np.nan)
                    if erased_metrics is not None
                    else np.nan
                ),
                (
                    erased_metrics.get("policy_action_norm", np.nan)
                    if erased_metrics is not None
                    else np.nan
                ),
                (
                    erased_metrics.get("policy_action_norm", np.nan)
                    - metrics.get("policy_action_norm", np.nan)
                    if erased_metrics is not None
                    else np.nan
                ),
                erased_metrics.get("recon_mean", np.nan) if erased_metrics is not None else np.nan,
                (
                    erased_metrics.get("recon_mean", np.nan) - metrics.get("recon_mean", np.nan)
                    if erased_metrics is not None
                    else np.nan
                ),
                erased_metrics.get("feature_l2", np.nan) if erased_metrics is not None else np.nan,
                (
                    erased_metrics.get("feature_l2", np.nan) - metrics.get("feature_l2", np.nan)
                    if erased_metrics is not None
                    else np.nan
                ),
            ]
        )
        if all_steps and ((step_id + 1) % 20 == 0 or step_id + 1 == len(episode_indices)):
            print(f"computed gradcam frames: {step_id + 1}/{len(episode_indices)}")

    csv_path = os.path.join(output_dir, f"episode_{episode_id:04d}_{args.target}_gradcam.csv")
    np.savetxt(
        csv_path,
        np.asarray(rows, dtype=np.float32),
        delimiter=",",
        header=(
            "step,dataset_idx,raw_frame,next_raw_frame,score,recon_mean,recon_sum,kl,feature_l2,"
            "erase_score,erase_score_delta,erase_action_norm,erase_action_norm_delta,"
            "erase_recon_mean,erase_recon_mean_delta,erase_feature_l2,erase_feature_l2_delta"
        ),
        comments="",
    )

    video_path = ""
    if all_steps:
        video_path = os.path.join(output_dir, f"episode_{episode_id:04d}_{args.target}_gradcam.mp4")
        save_video(frames, video_path, fps=args.fps)
        if args.erase_compare:
            erase_video_path = os.path.join(
                output_dir,
                f"episode_{episode_id:04d}_{args.target}_erase_gradcam.mp4",
            )
            save_video(erased_frames, erase_video_path, fps=args.fps)

    if args.show_window:
        show_window(frames, fps=args.fps, window_scale=args.window_scale)

    print(f"episodes found: {num_episodes}")
    print(
        f"selected episode: {episode_id}, transitions: {len(episode_indices)} "
        f"(before chain filter: {before_len})"
    )
    print(f"chain threshold transitions: {args.chain_threshold_transitions}")
    print(f"target: {args.target}, target_dim: {args.target_dim}")
    print(f"erase compare: {args.erase_compare}")
    print(f"erase apply: {args.erase_apply}")
    if args.erase_compare or args.erase_apply:
        print(f"erase rect: {tuple(args.erase_rect)}, fill: {args.erase_fill}")
    if args.save_images:
        print(f"saved first png: {first_png_path}")
    print(f"save images: {args.save_images}")
    print(f"save npy: {args.save_npy}")
    print(f"saved csv: {csv_path}")
    if video_path:
        print(f"saved video: {video_path}")
        if args.erase_compare:
            print(f"saved erased video: {erase_video_path}")


if __name__ == "__main__":
    main()
