import h5py
import numpy as np
import os
import shutil
import torch
from PIL import Image
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF
from torch.utils.data import Dataset

'''

现在props和rgb和gripper是简单的concat到一起 xijie推荐了一种film的方法 先作为备选 可以让小数据的拟合的效果会更好 如果效果不好的话 要试试resnet之后 再加上一个film

'''
class FrankaImageDataset(Dataset):
    """HDF5 dataset for robot image observations plus EEF proprioception."""

    def __init__(
        self,
        dataset_path,
        image_size=(224, 224),
        reward_scale=100.0,
        success_reward=1.0,
        normalize_proprio=True,
        normalize_action=True,
    ):
        self.dataset_paths = self._normalize_dataset_paths(dataset_path)
        self.dataset_path = self.dataset_paths[0] if len(self.dataset_paths) == 1 else list(self.dataset_paths)
        self.image_size = tuple(image_size)
        self.reward_scale = reward_scale
        self.success_reward = success_reward
        self.normalize_proprio = normalize_proprio
        self.normalize_action_flag = normalize_action

        datasets = [self._load_hdf5(path) for path in self.dataset_paths]
        self._build_transitions(datasets)
        self.size = len(self.indices)

        print(
            "franka dataset size:",
            self.size,
            "proprio_dim:",
            self.state_dim,
            "action_dim:",
            self.action_dim,
            "image_size:",
            self.image_size,
            "reward_positive:",
            int((self.rewards > 0).sum()),
            "dataset_files:",
            len(self.dataset_paths),
            "rgb_cache:",
            self.rgb_shapes,
        )

    def _normalize_dataset_paths(self, dataset_path):
        if isinstance(dataset_path, (list, tuple)):
            paths = list(dataset_path)
        else:
            paths = [dataset_path]
        return paths

    def _load_hdf5(self, dataset_path):
        root = self._load_replay_cache(dataset_path)
        data = root["data"]
        return {
            "rgb": data["rgb"],
            "rgb_shape": tuple(root.attrs["raw_rgb_shape"]),
            "translation": data["translation"][:],
            "rotation": data["rotation"][:],
            "gripper_w": data["gripper_w"][:],
            "reward": data["reward"][:],
            "terminal": data["terminal"][:],
            "timeout": data["timeout"][:],
        }

    def _load_replay_cache(self, dataset_path):
        try:
            import zarr
            import numcodecs
        except ImportError as exc:
            raise ImportError(
                "FrankaImageDataset zarr cache requires zarr and numcodecs. "
                "Install them in the d4rl env before training."
            ) from exc

        cache_path = f"{dataset_path}.train_replay_{self.image_size[0]}x{self.image_size[1]}.zarr"
        if os.path.exists(cache_path):
            print(f"loading replay cache: {cache_path}", flush=True)
            return zarr.open_group(cache_path, mode="r")

        print(f"creating replay cache: {cache_path}", flush=True)
        tmp_cache_path = f"{cache_path}.tmp"
        if os.path.exists(tmp_cache_path):
            shutil.rmtree(tmp_cache_path)

        with h5py.File(dataset_path, "r") as h5_file:
            root = zarr.open_group(tmp_cache_path, mode="w")
            data_group = root.require_group("data")
            root.attrs["raw_rgb_shape"] = tuple(h5_file["rgb"].shape)

            for key in ["translation", "rotation", "gripper_w", "reward", "terminal", "timeout"]:
                value = h5_file[key][:]
                data_group.array(
                    name=key,
                    data=value,
                    chunks=value.shape,
                    compressor=None,
                    overwrite=True,
                )

            rgb = h5_file["rgb"]
            chunk_len = 178
            rgb_cache = data_group.require_dataset(
                name="rgb",
                shape=(len(rgb), self.image_size[0], self.image_size[1], 3),
                chunks=(chunk_len, self.image_size[0], self.image_size[1], 3),
                compressor=None,
                dtype=np.uint8,
                overwrite=True,
            )
            for start in range(0, len(rgb), chunk_len):
                end = min(start + chunk_len, len(rgb))
                rgb_block = rgb[start:end, 0:360, 90:570, :]
                resized_block = np.empty(
                    (end - start, self.image_size[0], self.image_size[1], 3),
                    dtype=np.uint8,
                )
                for local_idx, image in enumerate(rgb_block):
                    image = Image.fromarray(image).resize(
                        (self.image_size[1], self.image_size[0]),
                        resample=Image.BILINEAR,
                    )
                    resized_block[local_idx] = np.asarray(image, dtype=np.uint8)
                rgb_cache[start:end] = resized_block
                print(f"  cached rgb {end}/{len(rgb)}", flush=True)

        os.replace(tmp_cache_path, cache_path)
        return zarr.open_group(cache_path, mode="r")

    def _build_transitions(self, datasets):
        segments = [self._build_transition_segment(data) for data in datasets]
        segments = [segment for segment in segments if len(segment["indices"]) > 0]
        
        self.rgb_sources = [segment["rgb"] for segment in segments]
        self.rgb_shapes = [segment["rgb_shape"] for segment in segments]

        self.indices = np.concatenate([segment["indices"] for segment in segments], axis=0)
        self.next_indices = np.concatenate([segment["next_indices"] for segment in segments], axis=0)
        self.segment_ids = np.concatenate(
            [
                np.full(len(segment["indices"]), segment_idx, dtype=np.int64)
                for segment_idx, segment in enumerate(segments)
            ],
            axis=0,
        )
        self.raw_states = np.concatenate([segment["raw_states"] for segment in segments], axis=0)
        self.raw_next_states = np.concatenate([segment["raw_next_states"] for segment in segments], axis=0)
        self.actions = np.concatenate([segment["actions"] for segment in segments], axis=0)
        self.raw_actions = self.actions.copy()
        self.rewards = np.concatenate([segment["rewards"] for segment in segments], axis=0)
        self.not_dones = np.concatenate([segment["not_dones"] for segment in segments], axis=0)

        self.state_mean = self.raw_states.mean(axis=0)
        self.state_std = self.raw_states.std(axis=0)
        self.action_mean = self.actions.mean(axis=0)
        self.action_std = self.actions.std(axis=0)

        if self.normalize_proprio:
            self.states = self.normalize_state(self.raw_states)
            self.next_states = self.normalize_state(self.raw_next_states)
        else:
            self.states = self.raw_states.astype(np.float32)
            self.next_states = self.raw_next_states.astype(np.float32)

        if self.normalize_action_flag:
            self.actions = self.normalize_action(self.actions)

        self.state_dim = self.states.shape[1]
        self.action_dim = self.actions.shape[1]

    def _build_transition_segment(self, data):
        translation = data["translation"].astype(np.float32)
        rotation = data["rotation"].astype(np.float32)
        gripper = data["gripper_w"].astype(np.float32).reshape(-1, 1)
        reward = data["reward"].astype(np.float32)
        terminal = data["terminal"].astype(bool)
        timeout = data["timeout"].astype(bool)
        terminals = np.logical_or(terminal, timeout)

        proprio = np.concatenate([translation, rotation, gripper], axis=1)

        indices, next_indices, actions = self._make_delta_transitions(
            translation, rotation, gripper, proprio, terminals
        )

        return {
            "rgb": data["rgb"],
            "rgb_shape": data["rgb_shape"],
            "indices": indices,
            "next_indices": next_indices,
            "raw_states": proprio[indices],
            "raw_next_states": proprio[next_indices],
            "actions": actions,
            "rewards": reward[next_indices].reshape(-1, 1) * self.reward_scale,
            "not_dones": (1.0 - terminals[next_indices].astype(np.float32)).reshape(-1, 1),
        }

    def _make_delta_transitions(self, translation, rotation, gripper, proprio, terminals):
        delta_translation_threshold = 0.005   # 5 mm
        delta_rotation_threshold = 0.005      # rad
        delta_gripper_threshold = 0.001       # 1 mm

        indices = []
        next_indices = []
        actions = []
        for start in range(len(translation) - 1):
            if terminals[start]:
                continue
            for end in range(start + 1, len(translation)):
                trans_delta = np.linalg.norm(translation[end] - translation[start])
                quat_dot = abs(np.dot(rotation[start], rotation[end]))
                rot_delta = 2.0 * np.arccos(np.clip(quat_dot, -1.0, 1.0))
                gripper_delta = abs(gripper[end, 0] - gripper[start, 0])
                if (
                    trans_delta > delta_translation_threshold
                    or rot_delta > delta_rotation_threshold
                    or gripper_delta > delta_gripper_threshold
                ):
                    indices.append(start)
                    next_indices.append(end)
                    actions.append(proprio[end] - proprio[start])
                    break
                if terminals[end]:
                    break
        return (
            np.asarray(indices, dtype=np.int64),
            np.asarray(next_indices, dtype=np.int64),
            np.asarray(actions, dtype=np.float32),
        )

    def normalize_state(self, state):
        return ((state - self.state_mean) / (self.state_std + 0.000001)).astype(np.float32)

    def unnormalize_state(self, state):
        return state * (self.state_std + 0.000001) + self.state_mean

    def normalize_action(self, action):
        return ((action - self.action_mean) / (self.action_std + 0.000001)).astype(np.float32)

    def unnormalize_action(self, action):
        return action * (self.action_std + 0.000001) + self.action_mean

    def _image_to_tensor(self, image):
        image = torch.from_numpy(image).permute(2, 0, 1).contiguous()
        return image

    def _augment_images(self, image, next_image):
        random_shift_pad = 8
        random_erasing_p = 0.2
        erase_scale = (0.02, 0.08)
        erase_ratio = (0.3, 3.3)

        if random_shift_pad > 0:
            _, h, w = image.shape
            image = TF.pad(image, random_shift_pad, padding_mode="edge")
            next_image = TF.pad(next_image, random_shift_pad, padding_mode="edge")
            top = torch.randint(0, 2 * random_shift_pad + 1, (1,)).item()
            left = torch.randint(0, 2 * random_shift_pad + 1, (1,)).item()
            image = TF.crop(image, top, left, h, w)
            next_image = TF.crop(next_image, top, left, h, w)

        if torch.rand(1).item() < random_erasing_p:
            _, h, w = image.shape
            area = h * w
            for _ in range(10):
                target_area = area * torch.empty(1).uniform_(*erase_scale).item()
                aspect = torch.empty(1).uniform_(np.log(erase_ratio[0]), np.log(erase_ratio[1])).exp().item()
                erase_h = int(round((target_area * aspect) ** 0.5))
                erase_w = int(round((target_area / aspect) ** 0.5))
                if 0 < erase_h < h and 0 < erase_w < w:
                    top = torch.randint(0, h - erase_h + 1, (1,)).item()
                    left = torch.randint(0, w - erase_w + 1, (1,)).item()
                    image[:, top : top + erase_h, left : left + erase_w] = 0.0
                    next_image[:, top : top + erase_h, left : left + erase_w] = 0.0
                    break

        return image, next_image

    def __len__(self):
        return self.size

    def get_item(self, idx, augment=True):
        segment_id = self.segment_ids[idx]
        sample_idx = self.indices[idx]
        next_sample_idx = self.next_indices[idx]
        rgb = self.rgb_sources[segment_id]
        image = self._image_to_tensor(rgb[sample_idx])
        next_image = self._image_to_tensor(rgb[next_sample_idx])
        if augment:
            image, next_image = self._augment_images(image, next_image)
        # else:
        #     print("warning: no augmentation applied to image")
        state = self.states[idx]
        next_state = self.next_states[idx]
        return {
            "image": image,
            "next_image": next_image,
            "state": state,
            "next_state": next_state,
            "raw_state": self.raw_states[idx],
            "raw_next_state": self.raw_next_states[idx],
            "action": self.actions[idx],
            "reward": self.rewards[idx],
            "not_done": self.not_dones[idx],
        }

    def __getitem__(self, idx):
        return self.get_item(idx, augment=False)
