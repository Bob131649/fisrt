from pathlib import Path
import time

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F


def policy_image_tensor(image, device):
    image_t = torch.as_tensor(image, dtype=torch.float32, device=device)
    if image_t.ndim == 3:
        image_t = image_t.unsqueeze(0)
    if image_t.shape[-1] == 3:
        image_t = image_t.permute(0, 3, 1, 2)
    if image_t.max() > 1:
        image_t = image_t / 255.0
    return image_t.float()


def action_reconstruction_mse(policy, policy_state, image, action_n):
    if policy.actor_vae is None:
        return float("nan")
    with torch.no_grad():
        state_t = torch.as_tensor(policy_state.reshape(1, -1), dtype=torch.float32, device=policy.device)
        image_t = policy_image_tensor(image, policy.device)
        action_t = torch.as_tensor(action_n.reshape(1, -1), dtype=torch.float32, device=policy.device)
        obs = {"proprio": state_t, "image": image_t}
        recon_action, _, _ = policy.actor_vae(obs, action_t)
        mse = F.mse_loss(recon_action, action_t, reduction="none").mean(dim=1)
    return float(mse.item())


def _normalized_heatmap(heatmap):
    heatmap = np.maximum(np.asarray(heatmap, dtype=np.float32), 0.0)
    heatmap -= heatmap.min()
    if heatmap.max() > 1e-12:
        heatmap /= heatmap.max()
    return heatmap


def _overlay_heatmap(image_rgb, heatmap, alpha=0.45):
    heatmap_bgr = cv2.applyColorMap(np.uint8(255 * _normalized_heatmap(heatmap)), cv2.COLORMAP_JET)
    heatmap_rgb = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    image_rgb = image_rgb.astype(np.float32) / 255.0
    return np.uint8(np.clip(((1.0 - alpha) * image_rgb + alpha * heatmap_rgb) * 255.0, 0, 255))


def _resnet_attention_overlay(policy, policy_state, image):
    image_encoder = getattr(policy.actor.encoder, "image_encoder", None)
    target_layer = getattr(image_encoder, "feature_extractor", None)
    if target_layer is None:
        return None

    captured = {}

    def hook_fn(module, inputs, output):
        captured["activation"] = output
        output.retain_grad()

    handle = target_layer.register_forward_hook(hook_fn)
    policy.zero_grad(set_to_none=True)
    try:
        state_t = torch.as_tensor(policy_state.reshape(1, -1), dtype=torch.float32, device=policy.device)
        image_t = policy_image_tensor(image, policy.device).requires_grad_(True)
        obs = {"proprio": state_t, "image": image_t}
        latent_a = None if policy.vae else policy.actor(obs)
        action = policy.actor_vae.decode(obs, z=latent_a)
        q1, q2 = policy.critic(obs, action)
        (0.5 * (q1.mean() + q2.mean())).backward()
    finally:
        handle.remove()

    activation = captured.get("activation")
    if activation is None or activation.grad is None:
        return None
    weights = activation.grad.mean(dim=(2, 3), keepdim=True)
    heatmap = torch.relu((weights * activation).sum(dim=1))[0].detach().cpu().numpy()
    image_h, image_w = int(image_t.shape[-2]), int(image_t.shape[-1])
    if heatmap.shape != (image_h, image_w):
        heatmap = cv2.resize(heatmap, (image_w, image_h), interpolation=cv2.INTER_CUBIC)
    policy.zero_grad(set_to_none=True)
    return _overlay_heatmap(np.asarray(image, dtype=np.uint8), heatmap)


def _dino_patch_layout(dino_model, image_t):
    patch_size = getattr(dino_model, "patch_size", None)
    if patch_size is None:
        patch_size = getattr(getattr(dino_model, "patch_embed", None), "patch_size", 14)
    if isinstance(patch_size, (tuple, list)):
        patch_h, patch_w = int(patch_size[0]), int(patch_size[1])
    else:
        patch_h = patch_w = int(patch_size)
    return image_t.shape[-2] // patch_h, image_t.shape[-1] // patch_w


def _dino_attention_overlay(policy, image):
    image_encoder = getattr(policy.actor.encoder, "image_encoder", None)
    dino = getattr(image_encoder, "dino", None)
    if dino is None:
        return None
    image_t = policy_image_tensor(image, policy.device)
    if getattr(image_encoder, "normalize_image", True):
        image_t = (image_t - image_encoder.image_mean.to(image_t.device)) / image_encoder.image_std.to(image_t.device)

    blocks = list(getattr(dino, "blocks", []))
    if not blocks:
        return None
    with torch.no_grad():
        if hasattr(dino, "prepare_tokens_with_masks"):
            tokens = dino.prepare_tokens_with_masks(image_t, None)
        elif hasattr(dino, "prepare_tokens"):
            tokens = dino.prepare_tokens(image_t)
        else:
            return None
        for block in blocks[:-1]:
            tokens = block(tokens)
        norm_tokens = blocks[-1].norm1(tokens)
        qkv = blocks[-1].attn.qkv(norm_tokens)
        batch_size, num_tokens, three_times_dim = qkv.shape
        num_heads = int(blocks[-1].attn.num_heads)
        head_dim = three_times_dim // 3 // num_heads
        qkv = qkv.reshape(batch_size, num_tokens, 3, num_heads, head_dim).permute(2, 0, 3, 1, 4)
        q, k = qkv[0], qkv[1]
        scale = getattr(blocks[-1].attn, "scale", head_dim ** -0.5)
        attention = ((q @ k.transpose(-2, -1)) * scale).softmax(dim=-1)

    patch_h, patch_w = _dino_patch_layout(dino, image_t)
    patch_count = patch_h * patch_w
    patch_start = 1 + int(getattr(dino, "num_register_tokens", 0))
    heatmap = attention[0, :, 0, patch_start:patch_start + patch_count].mean(dim=0)
    heatmap = heatmap.reshape(patch_h, patch_w).detach().cpu().numpy()
    heatmap = cv2.resize(heatmap, (int(image_t.shape[-1]), int(image_t.shape[-2])), interpolation=cv2.INTER_CUBIC)
    return _overlay_heatmap(np.asarray(image, dtype=np.uint8), heatmap)


def encoder_attention_overlay(policy, policy_state, image):
    overlay = _dino_attention_overlay(policy, image)
    if overlay is not None:
        return overlay
    return _resnet_attention_overlay(policy, policy_state, image)


def render_preview(rgb, policy_input, crop, recon_values, step, attention_overlay=None):
    top, left, height, width = crop
    raw_rgb = rgb.copy()
    cv2.rectangle(raw_rgb, (left, top), (left + width, top + height), (0, 255, 0), 2)
    raw_bgr = cv2.cvtColor(raw_rgb, cv2.COLOR_RGB2BGR)
    input_bgr = cv2.resize(
        cv2.cvtColor(policy_input, cv2.COLOR_RGB2BGR),
        (raw_bgr.shape[1], raw_bgr.shape[0]),
        interpolation=cv2.INTER_NEAREST,
    )
    panels = [raw_bgr, input_bgr]
    if attention_overlay is not None:
        panels.append(cv2.resize(cv2.cvtColor(attention_overlay, cv2.COLOR_RGB2BGR), (raw_bgr.shape[1], raw_bgr.shape[0])))
    return np.hstack(panels + [_recon_panel(recon_values, raw_bgr.shape[0], step)])


def _recon_panel(recon_values, height, step, width=380):
    panel = np.full((height, width, 3), 245, dtype=np.uint8)
    cv2.putText(panel, "recon mse", (16, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (40, 40, 40), 2, cv2.LINE_AA)
    if not recon_values:
        return panel
    values = np.asarray(recon_values, dtype=np.float32)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return panel
    cv2.putText(panel, f"latest: {values[-1]:.6f}", (16, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (30, 30, 30), 1, cv2.LINE_AA)
    cv2.putText(panel, f"step: {step}", (16, 88), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (30, 30, 30), 1, cv2.LINE_AA)
    y_min, y_max = float(values.min()), float(values.max())
    if abs(y_max - y_min) < 1e-12:
        y_max = y_min + 1.0
    left, right, top, bottom = 18, width - 18, 118, height - 34
    xs = np.linspace(left, right, len(values))
    ys = bottom - (values - y_min) / (y_max - y_min) * (bottom - top)
    points = np.stack([xs, ys], axis=1).astype(np.int32).reshape(-1, 1, 2)
    cv2.rectangle(panel, (left, top), (right, bottom), (220, 220, 220), 1)
    cv2.polylines(panel, [points], False, (220, 90, 40), 2, cv2.LINE_AA)
    return panel


def show_debug_window(frame):
    cv2.imshow("deploy_debug", frame)
    return (cv2.waitKey(1) & 0xFF) == ord("q")


def close_debug_windows():
    cv2.destroyAllWindows()


def _unique_path(path):
    path = Path(path)
    if not path.exists():
        return path
    for idx in range(1, 10000):
        candidate = path.with_name(f"{path.stem}_{idx:03d}{path.suffix}")
        if not candidate.exists():
            return candidate
    raise RuntimeError(f"could not find an unused output path for {path}")


class DeployRecorder:
    def __init__(self, output, fps):
        self.video_path, self.plot_path = self._paths(output)
        self.fps = fps
        self.frames = []
        self.recon_values = []

    @property
    def enabled(self):
        return self.video_path is not None

    def _paths(self, output):
        if not output:
            return None, None
        output_path = Path(output).expanduser()
        if output_path.suffix:
            video_path = _unique_path(output_path)
            plot_path = _unique_path(video_path.with_name(f"{video_path.stem}_recon_mse.png"))
        else:
            output_path.mkdir(parents=True, exist_ok=True)
            run_id = time.strftime("%Y%m%d_%H%M%S")
            video_path = _unique_path(output_path / f"deploy_{run_id}.mp4")
            plot_path = _unique_path(output_path / f"deploy_{run_id}_recon_mse.png")
        video_path.parent.mkdir(parents=True, exist_ok=True)
        plot_path.parent.mkdir(parents=True, exist_ok=True)
        return video_path, plot_path

    def print_targets(self):
        if self.enabled:
            print(f"[output] deploy video: {self.video_path}")
            print(f"[output] recon MSE plot: {self.plot_path}")

    def record(self, preview_frame, recon_mse):
        self.recon_values.append(recon_mse)
        if not self.enabled:
            return
        self.frames.append(preview_frame)

    def save(self):
        if not self.enabled:
            return
        if self.frames:
            height, width = self.frames[0].shape[:2]
            writer = cv2.VideoWriter(str(self.video_path), cv2.VideoWriter_fourcc(*"mp4v"), float(self.fps), (width, height))
            if not writer.isOpened():
                raise RuntimeError(f"failed to open video writer: {self.video_path}")
            for frame in self.frames:
                writer.write(frame)
            writer.release()
        if self.recon_values:
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(np.arange(len(self.recon_values)), self.recon_values, linewidth=2)
            ax.set_xlabel("Deploy Step")
            ax.set_ylabel("Recon MSE per action dim")
            ax.grid(True, alpha=0.25)
            fig.tight_layout()
            fig.savefig(self.plot_path, dpi=180)
            plt.close(fig)
        print(f"[saved] deploy video: {self.video_path}")
        print(f"[saved] recon MSE plot: {self.plot_path}")
