#!/usr/bin/env python3

import argparse
from concurrent.futures import ThreadPoolExecutor
import os

import numpy as np


TRAIN_CROP_TOP = 0
TRAIN_CROP_LEFT = 90
TRAIN_CROP_HEIGHT = 360
TRAIN_CROP_WIDTH = 480
DELTA_TRANSLATION_THRESHOLD = 0.005
DELTA_ROTATION_THRESHOLD = 0.005
DELTA_GRIPPER_THRESHOLD = 0.001

cv2 = None
h5py = None


def import_runtime_deps():
    global cv2, h5py
    if cv2 is None:
        import cv2 as cv2_module

        cv2 = cv2_module
    if h5py is None:
        import h5py as h5py_module

        h5py = h5py_module


def train_crop_rgb(rgb_frame, image_size):
    """Match FrankaImageDataset._image_to_tensor crop + bilinear resize."""
    h, w = rgb_frame.shape[:2]
    bottom = TRAIN_CROP_TOP + TRAIN_CROP_HEIGHT
    right = TRAIN_CROP_LEFT + TRAIN_CROP_WIDTH
    if bottom > h or right > w:
        raise ValueError(
            "RGB frame is smaller than the train crop window: "
            f"frame={w}x{h}, crop left/top/width/height="
            f"{TRAIN_CROP_LEFT}/{TRAIN_CROP_TOP}/{TRAIN_CROP_WIDTH}/{TRAIN_CROP_HEIGHT}"
        )

    cropped = rgb_frame[
        TRAIN_CROP_TOP:bottom,
        TRAIN_CROP_LEFT:right,
    ]
    out_h, out_w = image_size
    return cv2.resize(cropped, (out_w, out_h), interpolation=cv2.INTER_LINEAR)


def get_camera_fps(h5_file):
    if "camera" in h5_file and "fps" in h5_file["camera"].attrs:
        return float(h5_file["camera"].attrs["fps"])
    return 10.0


def build_episode_ranges(timeout, total_frames):
    if timeout is None:
        return [(0, total_frames - 1)] if total_frames > 0 else []

    timeout_values = np.asarray(timeout[:], dtype=bool)
    episode_ranges = []
    start = 0
    for idx, is_timeout in enumerate(timeout_values):
        if is_timeout:
            episode_ranges.append((start, idx))
            start = idx + 1

    if start < len(timeout_values):
        episode_ranges.append((start, len(timeout_values) - 1))
    return episode_ranges


def select_episode_range(h5_file, episode_num):
    total_frames = len(h5_file["rgb"])
    timeout = h5_file["timeout"] if "timeout" in h5_file else None
    episode_ranges = build_episode_ranges(timeout, total_frames)
    if not episode_ranges:
        raise ValueError("no episodes found in dataset")
    if episode_num < 0 or episode_num >= len(episode_ranges):
        raise IndexError(
            f"episode_num must be in [0, {len(episode_ranges) - 1}], got {episode_num}"
        )
    return episode_ranges[episode_num], len(episode_ranges)


def load_terminals(h5_file):
    total_frames = len(h5_file["rgb"])
    terminal = np.zeros(total_frames, dtype=bool)
    if "terminal" in h5_file:
        terminal = np.asarray(h5_file["terminal"][:], dtype=bool)
    elif "done" in h5_file:
        terminal = np.asarray(h5_file["done"][:], dtype=bool)

    timeout = np.zeros(total_frames, dtype=bool)
    if "timeout" in h5_file:
        timeout = np.asarray(h5_file["timeout"][:], dtype=bool)
    return np.logical_or(terminal, timeout)


def build_train_transitions(h5_file):
    """Match FrankaImageDataset threshold action selection exactly."""
    translation = np.asarray(h5_file["translation"][:], dtype=np.float32)
    rotation = np.asarray(h5_file["rotation"][:], dtype=np.float32)
    gripper = np.asarray(h5_file["gripper_w"][:], dtype=np.float32).reshape(-1, 1)
    terminals = load_terminals(h5_file)
    proprio = np.concatenate([translation, rotation, gripper], axis=1)

    transitions = []
    for start_idx in range(len(translation) - 1):
        if terminals[start_idx]:
            continue
        for end_idx in range(start_idx + 1, len(translation)):
            trans_delta = float(np.linalg.norm(translation[end_idx] - translation[start_idx]))
            quat_dot = abs(float(np.dot(rotation[start_idx], rotation[end_idx])))
            rot_delta = float(2.0 * np.arccos(np.clip(quat_dot, -1.0, 1.0)))
            gripper_delta = float(abs(gripper[end_idx, 0] - gripper[start_idx, 0]))
            if (
                trans_delta > DELTA_TRANSLATION_THRESHOLD
                or rot_delta > DELTA_ROTATION_THRESHOLD
                or gripper_delta > DELTA_GRIPPER_THRESHOLD
            ):
                transitions.append(
                    {
                        "start": start_idx,
                        "end": end_idx,
                        "skip": end_idx - start_idx,
                        "trans_delta": trans_delta,
                        "rot_delta": rot_delta,
                        "gripper_delta": gripper_delta,
                        "action": proprio[end_idx] - proprio[start_idx],
                        "reward": float(h5_file["reward"][end_idx]) if "reward" in h5_file else 0.0,
                        "not_done": float(1.0 - terminals[end_idx]),
                    }
                )
                break
            if terminals[end_idx]:
                break
    return transitions


def build_chained_train_transitions(h5_file):
    """Threshold-select transitions while dropping frames skipped by the selected action."""
    translation = np.asarray(h5_file["translation"][:], dtype=np.float32)
    rotation = np.asarray(h5_file["rotation"][:], dtype=np.float32)
    gripper = np.asarray(h5_file["gripper_w"][:], dtype=np.float32).reshape(-1, 1)
    terminals = load_terminals(h5_file)
    proprio = np.concatenate([translation, rotation, gripper], axis=1)

    transitions = []
    start_idx = 0
    while start_idx < len(translation) - 1:
        if terminals[start_idx]:
            start_idx += 1
            continue

        selected = None
        for end_idx in range(start_idx + 1, len(translation)):
            trans_delta = float(np.linalg.norm(translation[end_idx] - translation[start_idx]))
            quat_dot = abs(float(np.dot(rotation[start_idx], rotation[end_idx])))
            rot_delta = float(2.0 * np.arccos(np.clip(quat_dot, -1.0, 1.0)))
            gripper_delta = float(abs(gripper[end_idx, 0] - gripper[start_idx, 0]))
            if (
                trans_delta > DELTA_TRANSLATION_THRESHOLD
                or rot_delta > DELTA_ROTATION_THRESHOLD
                or gripper_delta > DELTA_GRIPPER_THRESHOLD
            ):
                selected = {
                    "start": start_idx,
                    "end": end_idx,
                    "skip": end_idx - start_idx,
                    "trans_delta": trans_delta,
                    "rot_delta": rot_delta,
                    "gripper_delta": gripper_delta,
                    "action": proprio[end_idx] - proprio[start_idx],
                    "reward": float(h5_file["reward"][end_idx]) if "reward" in h5_file else 0.0,
                    "not_done": float(1.0 - terminals[end_idx]),
                }
                break
            if terminals[end_idx]:
                break

        if selected is None:
            start_idx += 1
            continue

        transitions.append(selected)
        start_idx = selected["end"]

    return transitions


def preprocess_rgb_frame(rgb_frame, image_size):
    cropped_rgb = train_crop_rgb(rgb_frame, image_size)
    return cv2.cvtColor(cropped_rgb, cv2.COLOR_RGB2BGR)


def preload_episode(h5_file, start, end, image_size):
    frame_slice = slice(start, end + 1)
    rgb_frames = np.asarray(h5_file["rgb"][frame_slice])
    episode = {
        "start": start,
        "end": end,
        "frames": [preprocess_rgb_frame(rgb_frame, image_size) for rgb_frame in rgb_frames],
    }
    for key in ("translation", "rotation", "gripper_w", "done", "timeout", "reward"):
        if key in h5_file:
            episode[key] = np.asarray(h5_file[key][frame_slice])
    return episode


def value_at(source, key, frame_idx, local_idx, default=None):
    if source is None or key not in source:
        return default
    data = source[key]
    if isinstance(data, h5py.Dataset):
        return data[frame_idx]
    return data[local_idx]


def draw_reward_badge(canvas, reward):
    reward_value = 0.0 if reward is None else float(reward)
    reward_is_positive = reward_value > 0.0
    badge_text = f"reward {reward_value:.3f}"
    badge_color = (40, 180, 40) if reward_is_positive else (40, 40, 220)
    font_scale = 0.42
    thickness = 1
    padding_x = 6
    padding_y = 4
    margin = 6
    text_size, baseline = cv2.getTextSize(
        badge_text,
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        thickness,
    )
    badge_w = text_size[0] + padding_x * 2
    badge_h = text_size[1] + baseline + padding_y * 2
    badge_x = max(margin, canvas.shape[1] - badge_w - margin)
    badge_y = margin

    overlay = canvas.copy()
    cv2.rectangle(overlay, (badge_x, badge_y), (badge_x + badge_w, badge_y + badge_h), badge_color, -1)
    cv2.addWeighted(overlay, 0.58, canvas, 0.42, 0, canvas)
    cv2.rectangle(canvas, (badge_x, badge_y), (badge_x + badge_w, badge_y + badge_h), (255, 255, 255), 1)
    cv2.putText(
        canvas,
        badge_text,
        (badge_x + padding_x, badge_y + padding_y + text_size[1]),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        (255, 255, 255),
        thickness,
        cv2.LINE_AA,
    )


def draw_label(canvas, lines, origin=(8, 18), color=(255, 255, 255), font_scale=0.42):
    x, y = origin
    thickness = 1
    line_h = 16
    pad = 5
    text_w = 0
    for line in lines:
        size, _ = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
        text_w = max(text_w, size[0])
    box_h = line_h * len(lines) + pad * 2
    overlay = canvas.copy()
    cv2.rectangle(overlay, (x - pad, y - line_h + 2), (x + text_w + pad, y - line_h + 2 + box_h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, canvas, 0.45, 0, canvas)
    for idx, line in enumerate(lines):
        cv2.putText(
            canvas,
            line,
            (x, y + idx * line_h),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            color,
            thickness,
            cv2.LINE_AA,
        )


def draw_frame_counter(canvas, current, total):
    draw_label(
        canvas,
        [f"frame {current}/{total}"],
        origin=(8, 18),
        color=(255, 255, 255),
        font_scale=0.36,
    )


def compose_frame(rgb_frame, image_size, frame_idx, total_frames, source=None, local_idx=None, episode_text=None):
    local_idx = frame_idx if local_idx is None else local_idx
    canvas = preprocess_rgb_frame(rgb_frame, image_size)
    draw_frame_counter(canvas, local_idx + 1, total_frames)
    reward = value_at(source, "reward", frame_idx, local_idx)
    draw_reward_badge(canvas, reward)
    return canvas


def compose_transition_frame(h5_file, transition, image_size):
    start_idx = transition["start"]
    end_idx = transition["end"]
    start_frame = preprocess_rgb_frame(h5_file["rgb"][start_idx], image_size)
    end_frame = preprocess_rgb_frame(h5_file["rgb"][end_idx], image_size)
    draw_label(start_frame, [f"start {start_idx}", "train image"])
    draw_label(end_frame, [f"end {end_idx}", "train next_image"])
    draw_reward_badge(end_frame, transition["reward"])

    canvas = np.concatenate([start_frame, end_frame], axis=1)
    action = transition["action"]
    lines = [
        f"transition {start_idx}->{end_idx}  skip={transition['skip']}",
        f"dpos_norm={transition['trans_delta']:.4f}  drot={transition['rot_delta']:.4f}  dgrip={transition['gripper_delta']:.4f}",
        "action dpos="
        f"[{action[0]:+.4f}, {action[1]:+.4f}, {action[2]:+.4f}]  dgrip={action[7]:+.4f}",
        f"not_done={transition['not_done']:.0f}",
    ]
    draw_label(canvas, lines, origin=(8, canvas.shape[0] - 56), color=(230, 255, 230))
    return canvas


def compose_train_sample_frame(h5_file, transition, image_size):
    start_idx = transition["start"]
    frame = preprocess_rgb_frame(h5_file["rgb"][start_idx], image_size)
    action = transition["action"]
    lines = [
        f"train sample start={start_idx} end={transition['end']} skip={transition['skip']}",
        f"dpos=[{action[0]:+.4f}, {action[1]:+.4f}, {action[2]:+.4f}] dgrip={action[7]:+.4f}",
        f"dpos_norm={transition['trans_delta']:.4f} drot={transition['rot_delta']:.4f} dgrip_abs={transition['gripper_delta']:.4f}",
    ]
    draw_label(frame, lines, origin=(8, 38), color=(230, 255, 230))
    draw_reward_badge(frame, transition["reward"])
    return frame


def compose_preprocessed_frame(frame_bgr, frame_idx, source=None, local_idx=None):
    canvas = frame_bgr.copy()
    if source is not None and "frames" in source:
        draw_frame_counter(canvas, local_idx + 1, len(source["frames"]))
    reward = value_at(source, "reward", frame_idx, local_idx)
    draw_reward_badge(canvas, reward)
    return canvas


def scale_display_frame(frame, display_scale):
    if display_scale == 1.0:
        return frame
    h, w = frame.shape[:2]
    return cv2.resize(
        frame,
        (int(round(w * display_scale)), int(round(h * display_scale))),
        interpolation=cv2.INTER_NEAREST,
    )


def replay_frames(h5_file, image_size, fps, start, end, display_scale):
    rgb = h5_file["rgb"]
    total_frames = len(rgb)
    frame_idx = max(0, start)
    end = total_frames - 1 if end is None else min(end, total_frames - 1)
    replay_total_frames = end - frame_idx + 1
    replay_start = frame_idx
    paused = False
    delay_ms = max(1, int(1000.0 / max(fps, 1e-6)))

    while frame_idx <= end:
        frame = compose_frame(
            rgb_frame=rgb[frame_idx],
            image_size=image_size,
            frame_idx=frame_idx,
            total_frames=replay_total_frames,
            source=h5_file,
            local_idx=frame_idx - replay_start,
        )
        cv2.imshow("Train Crop Replay", scale_display_frame(frame, display_scale))
        key = cv2.waitKey(0 if paused else delay_ms) & 0xFF

        if key in (27, ord("q")):
            break
        if key in (ord("p"), 32):
            paused = not paused
            continue
        if key == ord("m"):
            frame_idx = max(start, frame_idx - 1)
            paused = True
            continue
        if key == ord("n"):
            frame_idx += 1
            paused = True
            continue
        if not paused:
            frame_idx += 1

    cv2.destroyAllWindows()


def replay_train_transitions(h5_file, image_size, fps, start, end, display_scale):
    transitions = build_train_transitions(h5_file)
    if start > 0 or end is not None:
        last = len(h5_file["rgb"]) - 1 if end is None else min(end, len(h5_file["rgb"]) - 1)
        transitions = [
            transition
            for transition in transitions
            if transition["start"] >= start and transition["start"] <= last
        ]
    if not transitions:
        print("no threshold-selected train transitions in the requested range")
        return

    skips = np.asarray([transition["skip"] for transition in transitions], dtype=np.int32)
    print(
        "train transitions:"
        f" count={len(transitions)}, skip min/mean/max={skips.min()}/{skips.mean():.2f}/{skips.max()},"
        f" thresholds trans>{DELTA_TRANSLATION_THRESHOLD:g}, rot>{DELTA_ROTATION_THRESHOLD:g},"
        f" gripper>{DELTA_GRIPPER_THRESHOLD:g}"
    )

    idx = 0
    paused = False
    delay_ms = max(1, int(1000.0 / max(fps, 1e-6)))
    while idx < len(transitions):
        frame = compose_train_sample_frame(h5_file, transitions[idx], image_size)
        draw_label(frame, [f"sample {idx + 1}/{len(transitions)}"], origin=(frame.shape[1] - 122, 18))
        draw_frame_counter(frame, idx + 1, len(transitions))
        cv2.imshow("Train Threshold Samples", scale_display_frame(frame, display_scale))
        key = cv2.waitKey(0 if paused else delay_ms) & 0xFF

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

    cv2.destroyAllWindows()


def replay_preloaded_episodes(h5_file, image_size, fps, start, end, display_scale):
    rgb = h5_file["rgb"]
    total_frames = len(rgb)
    timeout = h5_file["timeout"] if "timeout" in h5_file else None
    episode_ranges = build_episode_ranges(timeout, total_frames)
    episode_ranges = [
        (max(ep_start, start), min(ep_end, total_frames - 1 if end is None else end))
        for ep_start, ep_end in episode_ranges
        if ep_end >= start and ep_start <= (total_frames - 1 if end is None else end)
    ]
    episode_ranges = [(ep_start, ep_end) for ep_start, ep_end in episode_ranges if ep_start <= ep_end]
    if not episode_ranges:
        return

    paused = False
    frame_idx = episode_ranges[0][0]
    episode_idx = 0
    delay_ms = max(1, int(1000.0 / max(fps, 1e-6)))

    def submit_episode(executor, idx):
        if idx >= len(episode_ranges):
            return None
        ep_start, ep_end = episode_ranges[idx]
        print(f"preloading episode {idx + 1}/{len(episode_ranges)}: frames {ep_start}..{ep_end}")
        return executor.submit(preload_episode, h5_file, ep_start, ep_end, image_size)

    with ThreadPoolExecutor(max_workers=1) as executor:
        current_episode = preload_episode(h5_file, *episode_ranges[episode_idx], image_size)
        print(
            f"playing episode {episode_idx + 1}/{len(episode_ranges)}: "
            f"frames {current_episode['start']}..{current_episode['end']}"
        )
        next_future = submit_episode(executor, episode_idx + 1)

        while episode_idx < len(episode_ranges):
            if frame_idx > current_episode["end"]:
                episode_idx += 1
                if episode_idx >= len(episode_ranges):
                    break
                current_episode = next_future.result() if next_future is not None else preload_episode(
                    h5_file, *episode_ranges[episode_idx], image_size
                )
                print(
                    f"playing episode {episode_idx + 1}/{len(episode_ranges)}: "
                    f"frames {current_episode['start']}..{current_episode['end']}"
                )
                next_future = submit_episode(executor, episode_idx + 1)
                frame_idx = current_episode["start"]
                continue

            local_idx = frame_idx - current_episode["start"]
            frame = compose_preprocessed_frame(
                frame_bgr=current_episode["frames"][local_idx],
                frame_idx=frame_idx,
                source=current_episode,
                local_idx=local_idx,
            )
            cv2.imshow("Train Crop Replay", scale_display_frame(frame, display_scale))
            key = cv2.waitKey(0 if paused else delay_ms) & 0xFF

            if key in (27, ord("q")):
                break
            if key in (ord("p"), 32):
                paused = not paused
                continue
            if key == ord("m"):
                frame_idx = max(current_episode["start"], frame_idx - 1)
                paused = True
                continue
            if key == ord("n"):
                frame_idx += 1
                paused = True
                continue
            if not paused:
                frame_idx += 1

    cv2.destroyAllWindows()


def ensure_display_available():
    if os.environ.get("DISPLAY"):
        return
    raise RuntimeError("train crop replay requires a graphical display, but DISPLAY is not set")


def print_summary(h5_file, image_size, fps, display_scale):
    print(f"HDF5 file: {h5_file.filename}")
    for key in ("rgb", "translation", "rotation", "gripper_w", "done", "timeout", "reward"):
        if key in h5_file:
            print(f"{key}: shape={h5_file[key].shape}, dtype={h5_file[key].dtype}")
    print(
        "train crop:"
        f" top={TRAIN_CROP_TOP}, left={TRAIN_CROP_LEFT},"
        f" height={TRAIN_CROP_HEIGHT}, width={TRAIN_CROP_WIDTH},"
        f" output={image_size[1]}x{image_size[0]}, fps={fps:g},"
        f" display_scale={display_scale:g}"
    )
    print(
        "train transition thresholds:"
        f" trans>{DELTA_TRANSLATION_THRESHOLD:g},"
        f" rot>{DELTA_ROTATION_THRESHOLD:g},"
        f" gripper>{DELTA_GRIPPER_THRESHOLD:g}"
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Replay HDF5 RGB frames after applying the same crop/resize used by train_vision."
    )
    parser.add_argument("--dataset_path", required=True, help="Path to the .hdf5 dataset.")
    parser.add_argument("--fps", type=float, default=None, help="Playback fps. Defaults to dataset camera fps or 10.")
    parser.add_argument("--image_size", nargs=2, type=int, default=(224, 224), metavar=("HEIGHT", "WIDTH"))
    parser.add_argument("--display_scale", type=float, default=3.0, help="Scale factor used only for the replay window.")
    parser.add_argument("--start", type=int, default=0, help="First frame index to replay.")
    parser.add_argument("--end", type=int, default=None, help="Last frame index to replay, inclusive.")
    parser.add_argument(
        "--episode_num",
        type=int,
        default=None,
        help="Replay only one timeout-delimited episode by 0-based episode index. Overrides --start/--end.",
    )
    parser.add_argument(
        "--train_transitions",
        action="store_true",
        help="Replay only threshold-selected start->end transitions that enter FrankaImageDataset training.",
    )
    parser.add_argument(
        "--threshold",
        action="store_true",
        help="Alias for --train_transitions. Replay samples after threshold filtering.",
    )
    parser.add_argument(
        "--preload_episode",
        dest="preload_episode",
        action="store_true",
        help="Preload and preprocess one timeout-delimited episode at a time before playback.",
    )
    parser.add_argument(
        "--no_preload_episode",
        dest="preload_episode",
        action="store_false",
        help="Disable episode preloading and read/process frames during playback.",
    )
    parser.set_defaults(preload_episode=True)
    return parser.parse_args()


def main():
    args = parse_args()
    image_size = tuple(args.image_size)
    import_runtime_deps()
    ensure_display_available()

    with h5py.File(args.dataset_path, "r") as h5_file:
        if "rgb" not in h5_file:
            raise ValueError("dataset must contain rgb for train crop replay")
        fps = args.fps if args.fps is not None else get_camera_fps(h5_file)
        print_summary(h5_file, image_size, fps, args.display_scale)
        start = args.start
        end = args.end
        if args.episode_num is not None:
            (start, end), num_episodes = select_episode_range(h5_file, args.episode_num)
            print(
                f"selected episode {args.episode_num}/{num_episodes - 1}: "
                f"frames {start}..{end}"
            )
        if args.train_transitions or args.threshold:
            replay_train_transitions(h5_file, image_size, fps, start, end, args.display_scale)
        elif args.preload_episode:
            replay_preloaded_episodes(h5_file, image_size, fps, start, end, args.display_scale)
        else:
            replay_frames(h5_file, image_size, fps, start, end, args.display_scale)


if __name__ == "__main__":
    main()
