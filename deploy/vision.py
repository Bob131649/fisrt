import cv2
import numpy as np


def crop_and_resize_rgb(rgb, image_size, crop):
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


def stats_crop_to_deploy_crop(crop):
    top, bottom, left, right = crop
    return (int(top), int(left), int(bottom) - int(top), int(right) - int(left))


def normalize(x, mean, std):
    return ((x - mean) / (std + 1e-6)).astype(np.float32)


def unnormalize(x, mean, std):
    return (x * (std + 1e-6) + mean).astype(np.float32)


def smooth_action(action, previous_action, smoothing):
    if previous_action is None or smoothing <= 0:
        return np.asarray(action, dtype=np.float32)
    smoothing = float(np.clip(smoothing, 0.0, 0.999))
    return smoothing * previous_action + (1.0 - smoothing) * np.asarray(action, dtype=np.float32)
