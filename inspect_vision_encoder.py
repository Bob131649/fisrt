#!/usr/bin/env python3

import argparse
import json
import os

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
    chain_threshold_episode_indices,
    load_json_if_exists,
)
from vision_eval.loss_plot import build_episode_indices, describe_episode

cv2 = None


def import_cv2():
    global cv2
    if cv2 is None:
        import cv2 as cv2_module

        cv2 = cv2_module
    return cv2


def str2bool(value):
    if isinstance(value, bool):
        return value
    value = value.lower()
    if value in ("yes", "true", "t", "1", "y"):
        return True
    if value in ("no", "false", "f", "0", "n"):
        return False
    raise argparse.ArgumentTypeError("Boolean value expected.")


def parse_step_id(value):
    if value is None:
        return None
    if isinstance(value, int):
        return value
    if str(value).lower() == "all":
        return "all"
    return int(value)


def save_image_tensor(image_tensor, output_path):
    plt.imsave(output_path, image_tensor_to_numpy(image_tensor))


def save_feature_plot(feature, output_path, title):
    feature = np.asarray(feature, dtype=np.float32).reshape(-1)
    fig, axes = plt.subplots(2, 1, figsize=(12, 6), constrained_layout=True)

    axes[0].plot(np.arange(len(feature)), feature, linewidth=1.4)
    axes[0].set_title(title)
    axes[0].set_xlabel("feature dim")
    axes[0].set_ylabel("value")
    axes[0].grid(True, alpha=0.25)

    axes[1].imshow(feature.reshape(1, -1), aspect="auto", cmap="viridis")
    axes[1].set_yticks([])
    axes[1].set_xlabel("feature dim")

    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def unique_path(path):
    path = os.path.abspath(path)
    if not os.path.exists(path):
        return path
    root, ext = os.path.splitext(path)
    for idx in range(1, 10000):
        candidate = f"{root}_{idx:03d}{ext}"
        if not os.path.exists(candidate):
            return candidate
    raise RuntimeError(f"could not find an unused output path for {path}")


def format_vec(values, precision=6):
    values = np.asarray(values, dtype=np.float32).reshape(-1)
    return "[" + ", ".join(f"{value:+.{precision}f}" for value in values) + "]"


def get_transition_actions(dataset, dataset_idx):
    normalized_action = np.asarray(dataset.actions[int(dataset_idx)], dtype=np.float32).reshape(-1)
    if hasattr(dataset, "raw_actions"):
        raw_action = np.asarray(dataset.raw_actions[int(dataset_idx)], dtype=np.float32).reshape(-1)
    elif getattr(dataset, "normalize_action_flag", False):
        raw_action = dataset.unnormalize_action(normalized_action)
    else:
        raw_action = normalized_action.copy()
    return raw_action, normalized_action


def action_to_raw(dataset, action):
    action = np.asarray(action, dtype=np.float32).reshape(-1)
    if getattr(dataset, "normalize_action_flag", False):
        return dataset.unnormalize_action(action)
    return action.copy()


def print_transition_action(dataset, dataset_idx, prefix="dataset action"):
    raw_action, normalized_action = get_transition_actions(dataset, dataset_idx)
    raw_state = np.asarray(dataset.raw_states[int(dataset_idx)], dtype=np.float32).reshape(-1)
    raw_next_state = np.asarray(dataset.raw_next_states[int(dataset_idx)], dtype=np.float32).reshape(-1)
    raw_delta_from_state = raw_next_state - raw_state

    print(f"{prefix} raw 8d:", format_vec(raw_action))
    print(
        f"{prefix} raw split:",
        f"pos={format_vec(raw_action[:3])}",
        f"quat={format_vec(raw_action[3:7])}",
        f"gripper={raw_action[7]:+.6f}",
    )
    print(f"{prefix} normalized 8d:", format_vec(normalized_action))
    print(
        f"{prefix} normalized split:",
        f"pos={format_vec(normalized_action[:3])}",
        f"quat={format_vec(normalized_action[3:7])}",
        f"gripper={normalized_action[7]:+.6f}",
    )
    print(
        f"{prefix} raw state delta check:",
        f"pos={format_vec(raw_delta_from_state[:3])}",
        f"quat={format_vec(raw_delta_from_state[3:7])}",
        f"gripper={raw_delta_from_state[7]:+.6f}",
    )


def print_transition_action_short(dataset, dataset_idx, step_label):
    raw_action, normalized_action = get_transition_actions(dataset, dataset_idx)
    print(
        f"{step_label} action:",
        f"raw_pos={format_vec(raw_action[:3])}",
        f"raw_gripper={raw_action[7]:+.6f}",
        f"norm_pos={format_vec(normalized_action[:3])}",
        f"norm_gripper={normalized_action[7]:+.6f}",
    )


def print_recon_action_diff(dataset, target_norm, recon_norm, prefix="vae recon action"):
    target_norm = np.asarray(target_norm, dtype=np.float32).reshape(-1)
    recon_norm = np.asarray(recon_norm, dtype=np.float32).reshape(-1)
    diff_norm = recon_norm - target_norm
    target_raw = action_to_raw(dataset, target_norm)
    recon_raw = action_to_raw(dataset, recon_norm)
    diff_raw = recon_raw - target_raw
    mse_per_dim = diff_norm ** 2

    print(f"{prefix} target normalized 8d:", format_vec(target_norm))
    print(f"{prefix} recon normalized 8d:", format_vec(recon_norm))
    print(f"{prefix} diff normalized 8d:", format_vec(diff_norm))
    print(
        f"{prefix} diff normalized split:",
        f"pos={format_vec(diff_norm[:3])}",
        f"quat={format_vec(diff_norm[3:7])}",
        f"gripper={diff_norm[7]:+.6f}",
    )
    print(f"{prefix} target raw 8d:", format_vec(target_raw))
    print(f"{prefix} recon raw 8d:", format_vec(recon_raw))
    print(f"{prefix} diff raw 8d:", format_vec(diff_raw))
    print(
        f"{prefix} diff raw split:",
        f"pos={format_vec(diff_raw[:3])}",
        f"quat={format_vec(diff_raw[3:7])}",
        f"gripper={diff_raw[7]:+.6f}",
    )
    print(f"{prefix} mse per dim normalized:", format_vec(mse_per_dim))
    print(
        f"{prefix} mse summary:",
        f"mean={mse_per_dim.mean():.6f}",
        f"sum={mse_per_dim.sum():.6f}",
        f"pos_mean={mse_per_dim[:3].mean():.6f}",
        f"gripper={mse_per_dim[7]:.6f}",
    )


def print_recon_action_diff_short(dataset, target_norm, recon_norm, step_label):
    target_norm = np.asarray(target_norm, dtype=np.float32).reshape(-1)
    recon_norm = np.asarray(recon_norm, dtype=np.float32).reshape(-1)
    diff_norm = recon_norm - target_norm
    diff_raw = action_to_raw(dataset, recon_norm) - action_to_raw(dataset, target_norm)
    print(
        f"{step_label} recon diff:",
        f"norm_pos={format_vec(diff_norm[:3])}",
        f"norm_gripper={diff_norm[7]:+.6f}",
        f"raw_pos={format_vec(diff_raw[:3])}",
        f"raw_gripper={diff_raw[7]:+.6f}",
        f"mse={np.mean(diff_norm ** 2):.6f}",
    )


def image_tensor_to_numpy(image_tensor):
    image = image_tensor.detach().cpu().float()
    if image.max() > 1:
        image = image / 255.0
    image = image.permute(1, 2, 0).numpy()
    return np.clip(image, 0.0, 1.0)


def normalize_heatmap(heatmap):
    heatmap = np.asarray(heatmap, dtype=np.float32)
    heatmap = np.maximum(heatmap, 0.0)
    heatmap = heatmap - heatmap.min()
    denom = heatmap.max()
    if denom > 1e-12:
        heatmap = heatmap / denom
    return heatmap


def save_heatmap_and_overlay(image_tensor, heatmap, heatmap_path, overlay_path, title, alpha=0.45):
    image = image_tensor_to_numpy(image_tensor)
    heatmap = normalize_heatmap(heatmap)
    cmap = plt.get_cmap("inferno")
    colored_heatmap = cmap(heatmap)[..., :3]
    overlay = np.clip((1.0 - alpha) * image + alpha * colored_heatmap, 0.0, 1.0)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(heatmap, cmap="inferno")
    ax.set_title(title)
    ax.axis("off")
    fig.tight_layout(pad=0.1)
    fig.savefig(heatmap_path, dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(overlay)
    ax.set_title(title)
    ax.axis("off")
    fig.tight_layout(pad=0.1)
    fig.savefig(overlay_path, dpi=180)
    plt.close(fig)


def build_overlay_arrays(image_tensor, heatmap, alpha=0.45):
    image = image_tensor_to_numpy(image_tensor)
    heatmap = normalize_heatmap(heatmap)
    if heatmap.shape[:2] != image.shape[:2]:
        cv2_mod = import_cv2()
        heatmap = cv2_mod.resize(
            heatmap,
            (image.shape[1], image.shape[0]),
            interpolation=cv2_mod.INTER_CUBIC,
        )
        heatmap = normalize_heatmap(heatmap)
    cmap = plt.get_cmap("inferno")
    colored_heatmap = cmap(heatmap)[..., :3]
    overlay = np.clip((1.0 - alpha) * image + alpha * colored_heatmap, 0.0, 1.0)
    return image, colored_heatmap, overlay


def _first_tensor(value):
    if torch.is_tensor(value):
        return value
    if isinstance(value, (tuple, list)):
        for item in value:
            found = _first_tensor(item)
            if found is not None:
                return found
    if isinstance(value, dict):
        for item in value.values():
            found = _first_tensor(item)
            if found is not None:
                return found
    return None


def _module_temperature(module):
    temperature = getattr(module, "temperature", 1.0)
    if torch.is_tensor(temperature):
        return temperature.detach()
    return torch.tensor(float(temperature))


def _spatial_softmax_attention(module, feature):
    if feature.ndim != 4:
        return None

    with torch.no_grad():
        feature = feature.detach()
        nets = getattr(module, "nets", None)
        if nets is not None:
            feature = nets(feature)

        batch_size, channels, height, width = feature.shape
        temperature = _module_temperature(module).to(device=feature.device, dtype=feature.dtype)
        attention = F.softmax(feature.reshape(batch_size * channels, height * width) / temperature, dim=-1)
        return attention.reshape(batch_size, channels, height, width)


def capture_spatial_softmax_outputs(policy):
    outputs = []
    handles = []

    def hook_fn(module, inputs, output):
        tensor = _first_tensor(output)
        attention = None
        if inputs:
            input_tensor = _first_tensor(inputs[0])
            if input_tensor is not None:
                attention = _spatial_softmax_attention(module, input_tensor)
        if tensor is not None:
            outputs.append(
                {
                    "coords": tensor.detach().cpu(),
                    "attention": attention.detach().cpu() if attention is not None else None,
                    "module": module.__class__.__name__,
                }
            )

    for module in policy.obs_encoder.modules():
        if "SpatialSoftmax" in module.__class__.__name__:
            handles.append(module.register_forward_hook(hook_fn))
    return outputs, handles


def remove_hooks(handles):
    for handle in handles:
        handle.remove()


def spatial_output_to_keypoints(outputs, image_shape, coord_range="minus_one_one"):
    if not outputs:
        return np.zeros((0, 2), dtype=np.float32)

    tensor = outputs[-1]["coords"] if isinstance(outputs[-1], dict) else outputs[-1]
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    coords = tensor[0].float().numpy()

    if coords.ndim == 1:
        if coords.size < 2:
            return np.zeros((0, 2), dtype=np.float32)
        coords = coords[: (coords.size // 2) * 2].reshape(-1, 2)
    elif coords.ndim >= 2:
        coords = coords.reshape(-1, coords.shape[-1])
        if coords.shape[1] > 2:
            coords = coords[:, :2]

    if coords.size == 0:
        return np.zeros((0, 2), dtype=np.float32)

    height, width = image_shape[-2], image_shape[-1]
    x = coords[:, 0].copy()
    y = coords[:, 1].copy()

    if coord_range == "zero_one":
        x = x * (width - 1)
        y = y * (height - 1)
    elif coord_range == "minus_one_one":
        x = (x + 1.0) * 0.5 * (width - 1)
        y = (y + 1.0) * 0.5 * (height - 1)
    elif np.nanmin(coords) >= -0.1 and np.nanmax(coords) <= 1.1:
        x = x * (width - 1)
        y = y * (height - 1)
    elif np.nanmin(coords) >= -1.1 and np.nanmax(coords) <= 1.1:
        x = (x + 1.0) * 0.5 * (width - 1)
        y = (y + 1.0) * 0.5 * (height - 1)

    keypoints = np.stack([x, y], axis=1)
    finite = np.isfinite(keypoints).all(axis=1)
    return keypoints[finite].astype(np.float32)


def spatial_outputs_to_weight_map(outputs, image_shape, agg="max"):
    attention = None
    for item in reversed(outputs):
        if isinstance(item, dict) and item.get("attention") is not None:
            attention = item["attention"]
            break
    if attention is None:
        return np.zeros(tuple(image_shape[-2:]), dtype=np.float32)

    maps = attention[0].float()
    if agg == "mean":
        weight_map = maps.mean(dim=0)
    elif agg == "sum":
        weight_map = maps.sum(dim=0)
    else:
        weight_map = maps.max(dim=0).values

    weight_map = weight_map.numpy()
    if weight_map.shape != tuple(image_shape[-2:]):
        cv2_mod = import_cv2()
        weight_map = cv2_mod.resize(
            weight_map,
            (int(image_shape[-1]), int(image_shape[-2])),
            interpolation=cv2_mod.INTER_CUBIC,
        )
    return normalize_heatmap(weight_map)


def find_average_pool_gradcam_layer(policy):
    image_encoder = getattr(policy.obs_encoder, "image_encoder", None)
    if image_encoder is None:
        return None
    return getattr(image_encoder, "feature_extractor", None)


def compute_recon_gradcam_saliency(policy, batch, loss_mode="recon_mean"):
    layer = find_average_pool_gradcam_layer(policy)
    if layer is None:
        raise RuntimeError(
            "Could not find obs_encoder.image_encoder.feature_extractor for Grad-CAM. "
            "Use a zipper / average-pooling encoder, or use --saliency_backend spatial."
        )

    captured = {}

    def hook_fn(module, inputs, output):
        captured["activation"] = output
        output.retain_grad()

    handle = layer.register_forward_hook(hook_fn)
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
        recons_action, mu, log_var = policy.actor_vae(feature, action)
        recon_per_dim = F.mse_loss(recons_action, action, reduction="none")

        if loss_mode == "recon_sum":
            loss = recon_per_dim.sum(dim=1).mean()
        elif loss_mode == "kl":
            kl_per_dim = -0.5 * (1 + log_var - mu.pow(2) - log_var.exp())
            loss = kl_per_dim.sum(dim=1).mean()
        elif loss_mode == "total":
            kl_per_dim = -0.5 * (1 + log_var - mu.pow(2) - log_var.exp())
            loss = recon_per_dim.mean(dim=1).mean() + policy.kl_beta * kl_per_dim.sum(dim=1).mean()
        else:
            loss = recon_per_dim.mean(dim=1).mean()

        loss.backward()
    finally:
        handle.remove()

    activation = captured.get("activation")
    if activation is None or activation.grad is None:
        raise RuntimeError("Grad-CAM layer did not capture activations with gradients.")
    weights = activation.grad.mean(dim=(2, 3), keepdim=True)
    heatmap = torch.relu((weights * activation).sum(dim=1))[0].detach().cpu().numpy()
    if heatmap.shape != tuple(image.shape[-2:]):
        cv2_mod = import_cv2()
        heatmap = cv2_mod.resize(
            heatmap,
            (int(image.shape[-1]), int(image.shape[-2])),
            interpolation=cv2_mod.INTER_CUBIC,
        )
    heatmap = normalize_heatmap(heatmap)

    return {
        "backend": "gradcam",
        "heatmap": heatmap,
        "keypoints": np.zeros((0, 2), dtype=np.float32),
        "feature": feature.detach().cpu().numpy()[0],
        "target_action_norm": action.detach().cpu().numpy()[0],
        "recon_action_norm": recons_action.detach().cpu().numpy()[0],
        "recon_mean": float(recon_per_dim.mean().detach().cpu()),
        "recon_sum": float(recon_per_dim.sum(dim=1).mean().detach().cpu()),
        "kl_raw": float((-0.5 * (1 + log_var - mu.pow(2) - log_var.exp())).sum(dim=1).mean().detach().cpu()),
        "saliency_loss": float(loss.detach().cpu()),
    }


def compute_recon_saliency(
    policy,
    batch,
    loss_mode="recon_mean",
    spatial_coord_range="minus_one_one",
    spatial_weight_agg="max",
    saliency_backend="auto",
):
    if policy.actor_vae is None:
        raise RuntimeError("Saliency needs actor_vae, but this policy has no actor_vae.")

    if saliency_backend == "gradcam":
        return compute_recon_gradcam_saliency(policy, batch, loss_mode=loss_mode)

    image = batch["image"].to(policy.device)
    if image.dtype == torch.uint8:
        image = image.float() / 255.0
    else:
        image = image.float()
        if image.max() > 1:
            image = image / 255.0

    state = batch["state"].to(policy.device).float()
    action = batch["action"].to(policy.device).float()

    spatial_outputs, hook_handles = capture_spatial_softmax_outputs(policy)
    if saliency_backend == "spatial" and not hook_handles:
        raise RuntimeError("No SpatialSoftmax module found for --saliency_backend spatial.")
    if saliency_backend == "auto" and not hook_handles:
        return compute_recon_gradcam_saliency(policy, batch, loss_mode=loss_mode)

    policy.zero_grad(set_to_none=True)
    try:
        with torch.no_grad():
            feature = policy._obs_feature(state, image=image)
    finally:
        remove_hooks(hook_handles)
    with torch.no_grad():
        recons_action, mu, log_var = policy.actor_vae(feature, action)
        recon_per_dim = F.mse_loss(recons_action, action, reduction="none")

        if loss_mode == "recon_sum":
            loss = recon_per_dim.sum(dim=1).mean()
        elif loss_mode == "kl":
            kl_per_dim = -0.5 * (1 + log_var - mu.pow(2) - log_var.exp())
            loss = kl_per_dim.sum(dim=1).mean()
        elif loss_mode == "total":
            kl_per_dim = -0.5 * (1 + log_var - mu.pow(2) - log_var.exp())
            loss = recon_per_dim.mean(dim=1).mean() + policy.kl_beta * kl_per_dim.sum(dim=1).mean()
        else:
            loss = recon_per_dim.mean(dim=1).mean()

    heatmap = spatial_outputs_to_weight_map(spatial_outputs, image.shape, agg=spatial_weight_agg)

    return {
        "backend": "spatial",
        "heatmap": heatmap,
        "keypoints": spatial_output_to_keypoints(spatial_outputs, image.shape, spatial_coord_range),
        "feature": feature.detach().cpu().numpy()[0],
        "target_action_norm": action.detach().cpu().numpy()[0],
        "recon_action_norm": recons_action.detach().cpu().numpy()[0],
        "recon_mean": float(recon_per_dim.mean().detach().cpu()),
        "recon_sum": float(recon_per_dim.sum(dim=1).mean().detach().cpu()),
        "kl_raw": float((-0.5 * (1 + log_var - mu.pow(2) - log_var.exp())).sum(dim=1).mean().detach().cpu()),
        "saliency_loss": float(loss.detach().cpu()),
    }


def draw_panel_label(frame_bgr, lines, origin=(8, 18)):
    cv2_mod = import_cv2()
    x, y = origin
    font = cv2_mod.FONT_HERSHEY_SIMPLEX
    font_scale = 0.42
    thickness = 1
    line_h = 16
    pad = 5
    text_w = 0
    for line in lines:
        size, _ = cv2_mod.getTextSize(line, font, font_scale, thickness)
        text_w = max(text_w, size[0])
    overlay = frame_bgr.copy()
    box_h = line_h * len(lines) + pad * 2
    cv2_mod.rectangle(
        overlay,
        (x - pad, y - line_h + 2),
        (x + text_w + pad, y - line_h + 2 + box_h),
        (0, 0, 0),
        -1,
    )
    cv2_mod.addWeighted(overlay, 0.55, frame_bgr, 0.45, 0, frame_bgr)
    for idx, line in enumerate(lines):
        cv2_mod.putText(
            frame_bgr,
            line,
            (x, y + idx * line_h),
            font,
            font_scale,
            (255, 255, 255),
            thickness,
            cv2_mod.LINE_AA,
        )


def rgb_float_to_bgr_uint8(image):
    cv2_mod = import_cv2()
    image_u8 = (np.clip(image, 0.0, 1.0) * 255.0).astype(np.uint8)
    return cv2_mod.cvtColor(image_u8, cv2_mod.COLOR_RGB2BGR)


def draw_keypoints(frame_bgr, keypoints):
    cv2_mod = import_cv2()
    if keypoints is None:
        return
    for idx, point in enumerate(keypoints):
        x, y = int(round(float(point[0]))), int(round(float(point[1])))
        if x < 0 or y < 0 or x >= frame_bgr.shape[1] or y >= frame_bgr.shape[0]:
            continue
        color = (0, 255, 255) if idx % 2 == 0 else (255, 80, 255)
        cv2_mod.circle(frame_bgr, (x, y), 4, color, -1, lineType=cv2_mod.LINE_AA)
        cv2_mod.circle(frame_bgr, (x, y), 6, (0, 0, 0), 1, lineType=cv2_mod.LINE_AA)


def compose_saliency_triptych(image_tensor, heatmap, keypoints, lines, alpha=0.45):
    image, heatmap_rgb, overlay = build_overlay_arrays(image_tensor, heatmap, alpha=alpha)
    keypoint_panel = rgb_float_to_bgr_uint8(image)
    draw_keypoints(keypoint_panel, keypoints)
    panels = [
        ("image", rgb_float_to_bgr_uint8(image)),
        ("keypoints", keypoint_panel),
        ("highlight overlay", rgb_float_to_bgr_uint8(overlay)),
        ("highlight", rgb_float_to_bgr_uint8(heatmap_rgb)),
    ]
    for name, panel in panels:
        draw_panel_label(panel, [name] + lines)
    return np.concatenate([panel for _, panel in panels], axis=1)


def save_keypoint_overlay(image_tensor, keypoints, output_path, title):
    frame = rgb_float_to_bgr_uint8(image_tensor_to_numpy(image_tensor))
    draw_keypoints(frame, keypoints)
    draw_panel_label(frame, [title, f"keypoints {len(keypoints)}"])
    cv2_mod = import_cv2()
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    cv2_mod.imwrite(output_path, frame)


def save_triptych_png(image_tensor, heatmap, keypoints, output_path, lines, alpha=0.45):
    frame = compose_saliency_triptych(
        image_tensor,
        heatmap,
        keypoints,
        lines=lines,
        alpha=alpha,
    )
    cv2_mod = import_cv2()
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    cv2_mod.imwrite(output_path, frame)


def parse_erase_rect(values):
    if values is None:
        return None
    if len(values) != 4:
        raise argparse.ArgumentTypeError("--erase_rect needs exactly four values: X1 Y1 X2 Y2")
    return tuple(int(value) for value in values)


def apply_erase_to_batch(batch, erase_rect, fill_mode="mean", enabled=True):
    if not enabled or erase_rect is None:
        return batch
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

    if fill_mode == "zero":
        fill = torch.zeros_like(image[:, :, y1:y2, x1:x2])
    elif fill_mode == "random":
        if image.dtype == torch.uint8:
            fill = torch.randint(
                0,
                256,
                image[:, :, y1:y2, x1:x2].shape,
                dtype=image.dtype,
                device=image.device,
            )
        else:
            fill = torch.rand_like(image[:, :, y1:y2, x1:x2])
            if image.max() > 1:
                fill = fill * 255.0
    else:
        fill = image.float().mean(dim=(2, 3), keepdim=True).to(dtype=image.dtype)
    image[:, :, y1:y2, x1:x2] = fill
    erased["image"] = image
    return erased


def erase_image_tensor(image_tensor, erase_rect, fill_mode="mean", enabled=True):
    if not enabled or erase_rect is None:
        return image_tensor
    batch = {"image": image_tensor.unsqueeze(0)}
    return apply_erase_to_batch(batch, erase_rect, fill_mode=fill_mode, enabled=enabled)["image"][0]


def print_erase_comparison(dataset, clean_saliency, erased_saliency):
    clean_recon = np.asarray(clean_saliency["recon_action_norm"], dtype=np.float32)
    erased_recon = np.asarray(erased_saliency["recon_action_norm"], dtype=np.float32)
    diff_norm = erased_recon - clean_recon
    diff_raw = action_to_raw(dataset, erased_recon) - action_to_raw(dataset, clean_recon)
    print(
        "erase delta:",
        f"recon_mean={erased_saliency['recon_mean'] - clean_saliency['recon_mean']:+.6f}",
        f"recon_sum={erased_saliency['recon_sum'] - clean_saliency['recon_sum']:+.6f}",
        f"kl_raw={erased_saliency['kl_raw'] - clean_saliency['kl_raw']:+.6f}",
        f"action_norm_l2={np.linalg.norm(diff_norm):.6f}",
        f"action_raw_l2={np.linalg.norm(diff_raw):.6f}",
    )
    print("erase action norm diff:", format_vec(diff_norm))
    print("erase action raw diff:", format_vec(diff_raw))


def scale_frame(frame, display_scale):
    if display_scale == 1.0:
        return frame
    cv2_mod = import_cv2()
    h, w = frame.shape[:2]
    return cv2_mod.resize(
        frame,
        (int(round(w * display_scale)), int(round(h * display_scale))),
        interpolation=cv2_mod.INTER_NEAREST,
    )


def show_episode_frames(frames, fps, display_scale, window_name="Vision Saliency Episode"):
    if not frames:
        return
    if not os.environ.get("DISPLAY"):
        print("DISPLAY is not set; skipped interactive playback.")
        return

    cv2_mod = import_cv2()
    idx = 0
    paused = False
    delay_ms = max(1, int(1000.0 / max(float(fps), 1e-6)))
    while idx < len(frames):
        cv2_mod.imshow(window_name, scale_frame(frames[idx], display_scale))
        key = cv2_mod.waitKey(0 if paused else delay_ms) & 0xFF
        if key in (27, ord("q")):
            break
        if key in (ord("p"), 32):
            paused = not paused
            continue
        if key in (ord("m"), ord("b"), 81):
            idx = max(0, idx - 1)
            paused = True
            continue
        if key in (ord("n"), 83):
            idx += 1
            paused = True
            continue
        if not paused:
            idx += 1
    cv2_mod.destroyAllWindows()


def save_video(frames, output_path, fps):
    if not frames:
        return
    cv2_mod = import_cv2()
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    h, w = frames[0].shape[:2]
    writer = cv2_mod.VideoWriter(
        output_path,
        cv2_mod.VideoWriter_fourcc(*"mp4v"),
        float(fps),
        (w, h),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open video writer: {output_path}")
    for frame in frames:
        writer.write(frame)
    writer.release()


def find_dataset_episode(episodes, dataset, dataset_id, dataset_episode):
    matched = []
    for episode in episodes:
        first_idx = int(episode[0])
        if int(dataset.segment_ids[first_idx]) == int(dataset_id):
            matched.append(episode)

    if not matched:
        raise IndexError(f"No episodes found for dataset_id={dataset_id}.")
    if not 0 <= dataset_episode < len(matched):
        raise IndexError(
            f"dataset_episode must be in [0, {len(matched) - 1}] for dataset_id={dataset_id}, "
            f"got {dataset_episode}."
        )
    return matched[dataset_episode], len(matched)


def pick_episode_indices(dataset, episode_id, seed, dataset_id, dataset_episode, threshold=False):
    episodes = build_episode_indices(dataset)
    if not episodes:
        raise RuntimeError("No episode-like contiguous transition sequence found.")

    rng = np.random.default_rng(seed)
    if dataset_id is not None or dataset_episode is not None:
        if dataset_id is None or dataset_episode is None:
            raise ValueError("--dataset_id and --dataset_episode must be used together.")
        episode, num_source_episodes = find_dataset_episode(episodes, dataset, dataset_id, dataset_episode)
        global_episode_id = int(episodes.index(episode))
        original_episode_len = len(episode)
        if threshold:
            episode = chain_threshold_episode_indices(dataset, episode)
        if not episode:
            raise RuntimeError("Threshold filtering produced an empty episode.")
        return episode, {
            "dataset_id": int(dataset_id),
            "dataset_episode": int(dataset_episode),
            "source_episode_count": int(num_source_episodes),
            "episode_len": len(episode),
            "before_threshold_len": int(original_episode_len),
            "threshold": bool(threshold),
            "global_episode_id": global_episode_id,
        }

    if episode_id is None:
        episode_id = int(rng.integers(len(episodes)))
    if not 0 <= episode_id < len(episodes):
        raise IndexError(f"episode_id must be in [0, {len(episodes) - 1}], got {episode_id}")
    episode = episodes[episode_id]
    original_episode_len = len(episode)
    if threshold:
        episode = chain_threshold_episode_indices(dataset, episode)
    if not episode:
        raise RuntimeError("Threshold filtering produced an empty episode.")
    return episode, {
        "episode_id": int(episode_id),
        "episode_len": len(episode),
        "before_threshold_len": int(original_episode_len),
        "threshold": bool(threshold),
    }


def describe_selected_sample(dataset, dataset_idx):
    episodes = build_episode_indices(dataset)
    for global_episode_id, episode in enumerate(episodes):
        if int(dataset_idx) not in episode:
            continue
        step_id = episode.index(int(dataset_idx))
        info = describe_episode(dataset, episode)
        info.update(
            {
                "global_episode_id": int(global_episode_id),
                "step_id": int(step_id),
                "episode_len": int(len(episode)),
            }
        )
        return info

    segment_id = int(dataset.segment_ids[int(dataset_idx)])
    dataset_path = None
    if hasattr(dataset, "dataset_paths") and segment_id < len(dataset.dataset_paths):
        dataset_path = dataset.dataset_paths[segment_id]
    return {
        "segment_id": segment_id,
        "dataset_path": dataset_path,
        "dataset_name": os.path.basename(dataset_path) if dataset_path else f"segment_{segment_id}",
        "episode_id_in_dataset": None,
        "raw_start": int(dataset.indices[int(dataset_idx)]),
        "raw_end": int(dataset.next_indices[int(dataset_idx)]),
        "global_episode_id": None,
        "step_id": None,
        "episode_len": None,
    }


def pick_dataset_index(dataset, episode_id, step_id, sample_idx, seed, dataset_id, dataset_episode, threshold=False):
    if sample_idx is not None:
        return int(sample_idx), None

    episodes = build_episode_indices(dataset)
    if not episodes:
        raise RuntimeError("No episode-like contiguous transition sequence found.")

    rng = np.random.default_rng(seed)
    if dataset_id is not None or dataset_episode is not None:
        if dataset_id is None or dataset_episode is None:
            raise ValueError("--dataset_id and --dataset_episode must be used together.")
        episode, num_source_episodes = find_dataset_episode(episodes, dataset, dataset_id, dataset_episode)
        global_episode_id = int(episodes.index(episode))
        original_episode_len = len(episode)
        if threshold:
            episode = chain_threshold_episode_indices(dataset, episode)
        if not episode:
            raise RuntimeError("Threshold filtering produced an empty episode.")
        if step_id is None:
            step_id = int(rng.integers(len(episode)))
        if not 0 <= step_id < len(episode):
            raise IndexError(f"step_id must be in [0, {len(episode) - 1}], got {step_id}")
        return int(episode[step_id]), {
            "dataset_id": int(dataset_id),
            "dataset_episode": int(dataset_episode),
            "source_episode_count": int(num_source_episodes),
            "step_id": int(step_id),
            "episode_len": len(episode),
            "before_threshold_len": int(original_episode_len),
            "threshold": bool(threshold),
            "global_episode_id": global_episode_id,
        }

    if episode_id is None:
        episode_id = int(rng.integers(len(episodes)))
    if not 0 <= episode_id < len(episodes):
        raise IndexError(f"episode_id must be in [0, {len(episodes) - 1}], got {episode_id}")

    episode = episodes[episode_id]
    original_episode_len = len(episode)
    if threshold:
        episode = chain_threshold_episode_indices(dataset, episode)
    if not episode:
        raise RuntimeError("Threshold filtering produced an empty episode.")
    if step_id is None:
        step_id = int(rng.integers(len(episode)))
    if not 0 <= step_id < len(episode):
        raise IndexError(f"step_id must be in [0, {len(episode) - 1}], got {step_id}")

    return int(episode[step_id]), {
        "episode_id": episode_id,
        "step_id": step_id,
        "episode_len": len(episode),
        "before_threshold_len": int(original_episode_len),
        "threshold": bool(threshold),
    }


def inspect_episode_all_steps(policy, dataset, episode_indices, sample_info, output_dir, args):
    frames = []
    rows = []
    prefix = (
        f"dataset_{sample_info['segment_id']}_episode_{sample_info['episode_id_in_dataset']}"
        f"_global_{sample_info['global_episode_id']}"
    )
    video_path = args.video_path or os.path.join(
        output_dir,
        f"{prefix}_{args.saliency_loss}_saliency.mp4",
    )

    for local_step, dataset_idx in enumerate(episode_indices):
        sample = dataset.get_item(int(dataset_idx), augment=False)
        batch = default_collate([sample])
        batch_for_saliency = apply_erase_to_batch(
            batch,
            args.erase_rect,
            fill_mode=args.erase_fill,
            enabled=args.erase,
        )
        saliency = compute_recon_saliency(
            policy,
            batch_for_saliency,
            loss_mode=args.saliency_loss,
            spatial_coord_range=args.spatial_coord_range,
            spatial_weight_agg=args.spatial_weight_agg,
            saliency_backend=args.saliency_backend,
        )
        raw_idx = int(dataset.indices[int(dataset_idx)])
        next_raw_idx = int(dataset.next_indices[int(dataset_idx)])
        lines = [
            f"step {local_step + 1}/{len(episode_indices)} raw {raw_idx}->{next_raw_idx}",
            f"{saliency['backend']} recon {saliency['recon_mean']:.4f} kl {saliency['kl_raw']:.4f}",
            f"erase {args.erase_rect}" if args.erase else "",
        ]
        lines = [line for line in lines if line]
        frame = compose_saliency_triptych(
            erase_image_tensor(
                sample["image"],
                args.erase_rect,
                fill_mode=args.erase_fill,
                enabled=args.erase,
            ),
            saliency["heatmap"],
            saliency["keypoints"],
            lines=lines,
            alpha=float(args.overlay_alpha),
        )
        frames.append(frame)
        rows.append(
            [
                local_step,
                int(dataset_idx),
                raw_idx,
                next_raw_idx,
                saliency["recon_mean"],
                saliency["recon_sum"],
                saliency["kl_raw"],
                len(saliency["keypoints"]),
            ]
        )
        if (local_step + 1) % 20 == 0 or local_step + 1 == len(episode_indices):
            print(f"computed saliency frames: {local_step + 1}/{len(episode_indices)}")
        print_transition_action_short(
            dataset,
            int(dataset_idx),
            step_label=f"step {local_step + 1}/{len(episode_indices)} raw {raw_idx}->{next_raw_idx}",
        )
        print_recon_action_diff_short(
            dataset,
            saliency["target_action_norm"],
            saliency["recon_action_norm"],
            step_label=f"step {local_step + 1}/{len(episode_indices)} raw {raw_idx}->{next_raw_idx}",
        )

    save_video(frames, video_path, fps=args.fps)
    csv_path = os.path.splitext(video_path)[0] + ".csv"
    np.savetxt(
        csv_path,
        np.asarray(rows, dtype=np.float32),
        delimiter=",",
        header="step,dataset_idx,raw_frame,next_raw_frame,recon_mean,recon_sum,kl_raw,num_keypoints",
        comments="",
    )
    if not args.no_window:
        show_episode_frames(frames, fps=args.fps, display_scale=args.display_scale)

    return {"video_path": video_path, "csv_path": csv_path, "num_frames": len(frames)}


def main():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--model_dir", required=True, type=str)
    parser.add_argument("--checkpoint_name", default="model", type=str)
    parser.add_argument("--load_best", nargs="?", const=True, default=False, type=str2bool)
    parser.add_argument("--dataset_path", required=True, nargs="+", type=str)
    parser.add_argument("--output_dir", default="", type=str)
    parser.add_argument("--sample_idx", default=None, type=int)
    parser.add_argument("--episode_id", default=None, type=int)
    parser.add_argument("--dataset_id", default=None, type=int)
    parser.add_argument("--dataset_episode", default=None, type=int)
    parser.add_argument("--step_id", default=None, type=parse_step_id)
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--device", default="cuda", type=str)
    parser.add_argument(
        "--saliency_loss",
        default="recon_mean",
        choices=["recon_mean", "recon_sum", "kl", "total"],
        type=str,
    )
    parser.add_argument("--overlay_alpha", default=0.45, type=float)
    parser.add_argument("--save_saliency", nargs="?", const=True, default=True, type=str2bool)
    parser.add_argument(
        "--saliency_backend",
        default="auto",
        choices=["auto", "spatial", "gradcam"],
        type=str,
        help="auto uses SpatialSoftmax when present, otherwise Grad-CAM on the average-pooling encoder.",
    )
    parser.add_argument("--fps", default=10.0, type=float)
    parser.add_argument(
        "--erase",
        action="store_true",
        help="Enable occlusion with --erase_rect before saliency.",
    )
    parser.add_argument(
        "--erase_rect",
        nargs=4,
        type=int,
        default=(70, 105, 165, 160),
        metavar=("X1", "Y1", "X2", "Y2"),
        help="Policy-input pixel rectangle to occlude when --erase is set.",
    )
    parser.add_argument(
        "--erase_fill",
        default="mean",
        choices=["mean", "zero", "random"],
        type=str,
        help="Fill value for --erase_rect.",
    )
    parser.add_argument("--display_scale", default=2.0, type=float)
    parser.add_argument("--video_path", default="", type=str)
    parser.add_argument("--no_window", action="store_true")
    parser.add_argument(
        "--threshold",
        action="store_true",
        help="Use chained threshold transitions inside the selected episode before applying --step_id.",
    )
    parser.add_argument(
        "--spatial_coord_range",
        default="minus_one_one",
        choices=["minus_one_one", "zero_one", "auto"],
        type=str,
    )
    parser.add_argument(
        "--spatial_weight_agg",
        default="max",
        choices=["max", "mean", "sum"],
        type=str,
        help="How to aggregate SpatialSoftmax per-keypoint probability maps into one region-weight map.",
    )
    args = parser.parse_args()

    model_dir = os.path.expanduser(args.model_dir)
    variant = load_json_if_exists(os.path.join(model_dir, "variant.json"))
    stats = load_json_if_exists(os.path.join(model_dir, "normalization_stats.json"))
    image_size = tuple(stats.get("image_size") or variant.get("image_size", [224, 224]))

    dataset = FrankaImageDataset(
        args.dataset_path,
        image_size=image_size,
        reward_scale=float(variant.get("reward_scale", 100.0)),
        success_reward=float(variant.get("success_reward", 1.0)),
        normalize_proprio=bool(stats.get("normalize_proprio", variant.get("normalize_proprio", True))),
        normalize_action=bool(stats.get("normalize_action", variant.get("normalize_action", True))),
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
        max_v=float(max(dataset.rewards.max(), float(variant.get("reward_scale", 100.0)), 1.0)),
        device=args.device,
        discount=float(variant.get("discount", 0.99)),
        tau=float(variant.get("tau", 0.005)),
        vae_lr=float(variant.get("vae_lr", 2e-4)),
        actor_lr=float(variant.get("actor_lr", 2e-4)),
        critic_lr=float(variant.get("critic_lr", 2e-4)),
        obs_encoder_lr=variant.get("obs_encoder_lr", None),
        max_latent_action=float(variant.get("max_latent_action", 0.675)),
        expectile=float(variant.get("expectile", 0.9)),
        kl_beta=float(variant.get("kl_beta", 1.0)),
        doubleq_min=float(variant.get("doubleq_min", 1.0)),
        image_shape=image_shape,
        robomimic_feature_dim=int(variant.get("robomimic_feature_dim", 256)),
        robomimic_crop_shape=tuple(variant["robomimic_crop_shape"])
        if variant.get("robomimic_crop_shape") is not None
        else None,
        robomimic_backbone_class=variant.get("robomimic_backbone_class", "ResNet18Conv"),
        robomimic_pool_class=variant.get("robomimic_pool_class", "SpatialSoftmax"),
        encoder_mode=variant.get("encoder_mode", "robomimic"),
        concat_mode=variant.get("concat_mode", "default"),
        zipper_backbone=variant.get("zipper_backbone", "resnet18"),
        zipper_normalize_image=bool(variant.get("zipper_normalize_image", True)),
        vae=bool(variant.get("vae", False)),
    )
    policy.load(args.checkpoint_name, model_dir, load_best=args.load_best)
    policy.eval()

    output_dir = args.output_dir or os.path.join(model_dir, "vision_encoder_inspect")
    os.makedirs(output_dir, exist_ok=True)

    if args.step_id == "all":
        if args.sample_idx is not None:
            raise ValueError("--sample_idx cannot be used with --step_id all; select an episode instead.")
        for param in policy.parameters():
            param.requires_grad_(False)
        episode_indices, picked = pick_episode_indices(
            dataset,
            episode_id=args.episode_id,
            seed=args.seed,
            dataset_id=args.dataset_id,
            dataset_episode=args.dataset_episode,
            threshold=args.threshold,
        )
        sample_info = describe_selected_sample(dataset, int(episode_indices[0]))
        sample_info["step_id"] = 0
        sample_info["episode_len"] = len(episode_indices)
        sample_info["raw_start"] = int(dataset.indices[int(episode_indices[0])])
        sample_info["raw_end"] = int(dataset.next_indices[int(episode_indices[-1])])
        result = inspect_episode_all_steps(
            policy,
            dataset,
            episode_indices,
            sample_info,
            output_dir,
            args,
        )
        print("picked:", picked)
        print(
            "source:",
            f"dataset_id={sample_info['segment_id']}",
            f"dataset={sample_info['dataset_name']}",
            f"dataset_episode={sample_info['episode_id_in_dataset']}",
            f"global_episode={sample_info['global_episode_id']}",
            f"transitions={len(episode_indices)}",
            f"before_threshold={picked.get('before_threshold_len', len(episode_indices))}",
            f"threshold={args.threshold}",
        )
        print("raw frames:", sample_info["raw_start"], "to", sample_info["raw_end"])
        print("saved saliency video:", result["video_path"])
        print("saved losses csv:", result["csv_path"])
        return

    dataset_idx, picked = pick_dataset_index(
        dataset,
        episode_id=args.episode_id,
        step_id=args.step_id,
        sample_idx=args.sample_idx,
        seed=args.seed,
        dataset_id=args.dataset_id,
        dataset_episode=args.dataset_episode,
        threshold=args.threshold,
    )
    sample = dataset.get_item(dataset_idx, augment=False)
    batch = default_collate([sample])
    sample_info = describe_selected_sample(dataset, dataset_idx)
    if picked is not None:
        sample_info["step_id"] = picked.get("step_id", sample_info["step_id"])
        sample_info["episode_len"] = picked.get("episode_len", sample_info["episode_len"])

    if args.save_saliency:
        for param in policy.parameters():
            param.requires_grad_(False)
        saliency = compute_recon_saliency(
            policy,
            batch,
            loss_mode=args.saliency_loss,
            spatial_coord_range=args.spatial_coord_range,
            spatial_weight_agg=args.spatial_weight_agg,
            saliency_backend=args.saliency_backend,
        )
        feature = saliency["feature"]
        erased_saliency = None
        if args.erase:
            erased_batch = apply_erase_to_batch(
                batch,
                args.erase_rect,
                fill_mode=args.erase_fill,
                enabled=True,
            )
            erased_saliency = compute_recon_saliency(
                policy,
                erased_batch,
                loss_mode=args.saliency_loss,
                spatial_coord_range=args.spatial_coord_range,
                spatial_weight_agg=args.spatial_weight_agg,
                saliency_backend=args.saliency_backend,
            )
    else:
        with torch.no_grad():
            feature = policy._batch_feature(batch).detach().cpu().numpy()[0]
        saliency = None
        erased_saliency = None

    prefix = (
        f"dataset_{sample_info['segment_id']}_episode_{sample_info['episode_id_in_dataset']}"
        f"_step_{sample_info['step_id']}_sample_{dataset_idx}"
    )
    feature_path = os.path.join(output_dir, f"{prefix}_feature.npy")
    plot_path = os.path.join(output_dir, f"{prefix}_feature.png")
    visual_path = os.path.join(output_dir, f"{prefix}_{args.saliency_loss}_visual.png")
    erased_visual_path = os.path.join(output_dir, f"{prefix}_{args.saliency_loss}_erase_visual.png")
    feature_path = unique_path(feature_path)
    plot_path = unique_path(plot_path)
    visual_path = unique_path(visual_path)
    erased_visual_path = unique_path(erased_visual_path)

    np.save(feature_path, feature)
    title_context = (
        f"dataset {sample_info['segment_id']}:{sample_info['dataset_name']} "
        f"file_ep={sample_info['episode_id_in_dataset']} "
        f"step={sample_info['step_id']}/{sample_info['episode_len']} "
        f"raw={int(dataset.indices[dataset_idx])}->{int(dataset.next_indices[dataset_idx])}"
    )
    save_feature_plot(
        feature,
        plot_path,
        title=f"Vision obs feature | {title_context} | dim={feature.shape[0]}",
    )
    if saliency is not None:
        visual_lines = [
            f"step {sample_info['step_id'] + 1}/{sample_info['episode_len']}"
            if sample_info["step_id"] is not None
            else f"sample {dataset_idx}",
            f"raw {int(dataset.indices[dataset_idx])}->{int(dataset.next_indices[dataset_idx])}",
            f"{saliency['backend']} recon {saliency['recon_mean']:.4f} kl {saliency['kl_raw']:.4f}",
        ]
        save_triptych_png(
            sample["image"],
            saliency["heatmap"],
            saliency["keypoints"],
            visual_path,
            lines=visual_lines,
            alpha=float(args.overlay_alpha),
        )
        if erased_saliency is not None:
            erased_lines = visual_lines + [
                f"erase {tuple(args.erase_rect)} fill={args.erase_fill}",
                f"erased recon {erased_saliency['recon_mean']:.4f} kl {erased_saliency['kl_raw']:.4f}",
            ]
            save_triptych_png(
                erase_image_tensor(
                    sample["image"],
                    args.erase_rect,
                    fill_mode=args.erase_fill,
                    enabled=True,
                ),
                erased_saliency["heatmap"],
                erased_saliency["keypoints"],
                erased_visual_path,
                lines=erased_lines,
                alpha=float(args.overlay_alpha),
            )

    print("picked:", picked if picked is not None else {"sample_idx": dataset_idx})
    print("dataset_idx:", dataset_idx)
    print(
        "source:",
        f"dataset_id={sample_info['segment_id']}",
        f"dataset={sample_info['dataset_name']}",
        f"dataset_episode={sample_info['episode_id_in_dataset']}",
        f"global_episode={sample_info['global_episode_id']}",
        f"step={sample_info['step_id']}/{sample_info['episode_len']}",
        f"before_threshold={picked.get('before_threshold_len', sample_info['episode_len']) if picked else sample_info['episode_len']}",
        f"threshold={args.threshold}",
    )
    print("raw frame:", int(dataset.indices[dataset_idx]), "next:", int(dataset.next_indices[dataset_idx]))
    print_transition_action(dataset, dataset_idx)
    print("feature shape:", feature.shape)
    print(
        "feature stats:",
        f"mean={feature.mean():.6f}",
        f"std={feature.std():.6f}",
        f"min={feature.min():.6f}",
        f"max={feature.max():.6f}",
        f"l2={np.linalg.norm(feature):.6f}",
    )
    if saliency is not None:
        print_recon_action_diff(dataset, saliency["target_action_norm"], saliency["recon_action_norm"])
        print(
            "sample losses:",
            f"recon_mean={saliency['recon_mean']:.6f}",
            f"recon_sum={saliency['recon_sum']:.6f}",
            f"kl_raw={saliency['kl_raw']:.6f}",
            f"saliency_loss={saliency['saliency_loss']:.6f}",
            f"keypoints={len(saliency['keypoints'])}",
        )
        if erased_saliency is not None:
            print_erase_comparison(dataset, saliency, erased_saliency)
    print("saved feature npy:", feature_path)
    print("saved feature plot:", plot_path)
    if saliency is not None:
        print("saved visual summary:", visual_path)
        if erased_saliency is not None:
            print("saved erased visual summary:", erased_visual_path)


if __name__ == "__main__":
    main()
