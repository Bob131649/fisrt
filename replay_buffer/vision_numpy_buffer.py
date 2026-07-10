import json

import numpy as np
import numpy.linalg as LA
import torch
from torch.utils.data import DataLoader, Dataset, RandomSampler
from torchvision import transforms

from replay_buffer.vision_zarr_cache import cached_bad_indices, ensure_rgb_cache, open_cached_rgb


class WaterPipeDataset(Dataset):
    image_size = (224, 224)
    rgb_crop = (0, 360, 90, 570)  # top, bottom, left, right
    delta_translation_threshold = 0.003
    delta_rotation_threshold = 0.003
    delta_gripper_threshold = 0.0007

    def __init__(self, env_name, proprio_dim, action_dim, device):
        super().__init__()
        self.env_name = env_name
        self.proprio_dim = proprio_dim
        self.action_dim = action_dim
        self.device = torch.device(device)
        self.storage = {}
        self.stats = {}
        self.dataset_paths = []
        self.rgb_cache_paths = []
        self.rgb_caches = []
        self.bad_image_indices = []
        self.size = 0
        self.image_aug = transforms.Compose([
            transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.1),
            transforms.RandomAffine(degrees=90, translate=(0.45, 0.45), scale=(0.6, 1.4), fill=0),
            transforms.RandomErasing(p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=0),
        ])

    def __len__(self):
        return self.size

    def __getstate__(self):
        state = self.__dict__.copy()
        state["rgb_caches"] = [None] * len(self.rgb_cache_paths)
        return state

    def _load_rgb_cache(self, source_idx):
        if self.rgb_caches[source_idx] is None:
            self.rgb_caches[source_idx] = open_cached_rgb(self.rgb_cache_paths[source_idx])
        return self.rgb_caches[source_idx]

    def _image_tensor(self, rgb_cache, index):
        rgb = np.ascontiguousarray(rgb_cache[int(index)])
        return torch.from_numpy(rgb).permute(2, 0, 1).contiguous()

    def __getitem__(self, idx):
        idx = int(idx)
        source_idx = int(self.storage["image_source_index"][idx])
        rgb_cache = self._load_rgb_cache(source_idx)
        return (
            {
                "proprio": torch.from_numpy(self.storage["state"][idx]).float(),
                "image": self._image_tensor(rgb_cache, self.storage["image_index"][idx]),
            },
            torch.from_numpy(self.storage["action"][idx]).float(),
            {
                "proprio": torch.from_numpy(self.storage["next_state"][idx]).float(),
                "image": self._image_tensor(rgb_cache, self.storage["next_image_index"][idx]),
            },
            torch.from_numpy(self.storage["reward"][idx]).float().view(1),
            torch.from_numpy(self.storage["not_done"][idx]).float().view(1),
            torch.from_numpy(self.storage["next_action"][idx]).float(),
        )

    def _exceeds_motion_threshold(self, start_state, end_state):
        trans_delta = LA.norm(end_state[:3] - start_state[:3])
        quat_dot = abs(np.dot(start_state[3:7], end_state[3:7]))
        rot_delta = 2.0 * np.arccos(np.clip(quat_dot, -1.0, 1.0))
        gripper_delta = abs(end_state[7] - start_state[7])
        return (
            trans_delta > self.delta_translation_threshold
            or rot_delta > self.delta_rotation_threshold
            or gripper_delta > self.delta_gripper_threshold
        )

    def _find_transition_end(self, start, observations, terminals):
        for end in range(start + 1, len(observations)):
            if self._exceeds_motion_threshold(observations[start], observations[end]):
                return end
            if terminals[end]:
                break
        return None

    def _episode_ranges(self, terminals):
        episode_start = 0
        for terminal_idx in np.flatnonzero(terminals):
            if episode_start < terminal_idx:
                yield episode_start, terminal_idx
            episode_start = terminal_idx + 1
        if episode_start < len(terminals) - 1:
            yield episode_start, len(terminals) - 1

    def load(self, data, terminal_reward_count=None):
        if terminal_reward_count is not None and terminal_reward_count < 0:
            raise ValueError("terminal_reward_count must be non-negative or None")

        datasets = data if isinstance(data, (list, tuple)) else [data]
        self.dataset_paths = [dataset["_dataset_path"] for dataset in datasets]
        self.rgb_cache_paths = [
            ensure_rgb_cache(path, self.image_size, self.rgb_crop)
            for path in self.dataset_paths
        ]
        self.bad_image_indices = [
            cached_bad_indices(path)
            for path in self.rgb_cache_paths
        ]
        self.rgb_caches = [None] * len(self.rgb_cache_paths)

        states, actions, next_states, rewards, not_done, next_actions = [], [], [], [], [], []
        image_indices, next_image_indices, image_source_indices = [], [], []

        for source_idx, dataset in enumerate(datasets):
            observations = dataset["observations"].astype(np.float32)
            dataset_rewards = np.asarray(dataset["rewards"], dtype=np.float32).reshape(-1)
            terminals = dataset["terminals"].astype(bool)
            bad_indices = self.bad_image_indices[source_idx]

            if len(dataset_rewards) != len(observations):
                raise ValueError(
                    "rewards and observations must have the same length: "
                    f"{len(dataset_rewards)} != {len(observations)}"
                )

            for episode_start, episode_end in self._episode_ranges(terminals):
                episode_reward_indices = []
                i = episode_start

                while i < episode_end:
                    if terminals[i]:
                        i += 1
                        continue
                    if i in bad_indices:
                        i += 1
                        continue
                    end = self._find_transition_end(i, observations, terminals)
                    if end is None:
                        break
                    if end > episode_end:
                        break
                    if end in bad_indices:
                        i = end
                        continue

                    next_end = self._find_transition_end(end, observations, terminals)
                    if next_end is not None and (next_end > episode_end or next_end in bad_indices):
                        next_end = None
                    next_action = (
                        observations[next_end] - observations[end]
                        if next_end is not None
                        else observations[end] - observations[i]
                    )

                    states.append(observations[i])
                    actions.append(observations[end] - observations[i])
                    next_states.append(observations[end])
                    reward = dataset_rewards[end] if terminal_reward_count is None else 0.0
                    rewards.append([reward])
                    not_done.append([1.0 - terminals[end]])
                    next_actions.append(next_action)
                    image_indices.append(i)
                    next_image_indices.append(end)
                    image_source_indices.append(source_idx)
                    episode_reward_indices.append(len(rewards) - 1)
                    i = end

                if terminal_reward_count:
                    for reward_idx in episode_reward_indices[-terminal_reward_count:]:
                        rewards[reward_idx][0] = 1.0

        self.storage["state"] = np.asarray(states, dtype=np.float32)
        self.storage["action"] = np.asarray(actions, dtype=np.float32)
        self.storage["next_state"] = np.asarray(next_states, dtype=np.float32)
        self.storage["reward"] = np.asarray(rewards, dtype=np.float32)
        self.storage["not_done"] = np.asarray(not_done, dtype=np.float32)
        self.storage["next_action"] = np.asarray(next_actions, dtype=np.float32)
        self.storage["image_index"] = np.asarray(image_indices, dtype=np.int64)
        self.storage["next_image_index"] = np.asarray(next_image_indices, dtype=np.int64)
        self.storage["image_source_index"] = np.asarray(image_source_indices, dtype=np.int64)
        self.size = self.storage["state"].shape[0]
        bad_count = sum(len(indices) for indices in self.bad_image_indices)
        if bad_count:
            print(f"filtered bad image indices from replay cache: {bad_count}")
        return self

    def subset(self, indices):
        indices = np.asarray(indices, dtype=np.int64)
        subset = WaterPipeDataset(self.env_name, self.proprio_dim, self.action_dim, self.device)
        subset.dataset_paths = list(self.dataset_paths)
        subset.rgb_cache_paths = list(self.rgb_cache_paths)
        subset.rgb_caches = [None] * len(subset.rgb_cache_paths)
        subset.bad_image_indices = [set(bad_indices) for bad_indices in self.bad_image_indices]
        subset.storage = {key: value[indices].copy() for key, value in self.storage.items()}
        subset.size = len(indices)
        if self.stats:
            subset.apply_stats(self.stats, normalize=False)
        return subset

    def split_val(self, val_fraction=0.05, seed=0):
        indices = np.arange(self.size)
        rng = np.random.default_rng(seed)
        rng.shuffle(indices)
        val_size = max(1, int(self.size * val_fraction))
        val_indices = indices[:val_size]
        train_indices = indices[val_size:]
        return self.subset(train_indices), self.subset(val_indices)

    @classmethod
    def concat(cls, pipes):
        merged = cls(pipes[0].env_name, pipes[0].proprio_dim, pipes[0].action_dim, pipes[0].device)
        storage = {key: [] for key in pipes[0].storage}
        source_offset = 0
        for pipe in pipes:
            merged.dataset_paths.extend(pipe.dataset_paths)
            merged.rgb_cache_paths.extend(pipe.rgb_cache_paths)
            merged.bad_image_indices.extend(set(bad_indices) for bad_indices in pipe.bad_image_indices)
            for key, value in pipe.storage.items():
                if key == "image_source_index":
                    value = value + source_offset
                storage[key].append(value)
            source_offset += len(pipe.rgb_cache_paths)
        merged.rgb_caches = [None] * len(merged.rgb_cache_paths)
        merged.storage = {key: np.concatenate(values, axis=0) for key, values in storage.items()}
        merged.size = merged.storage["state"].shape[0]
        if pipes[0].stats:
            merged.apply_stats(pipes[0].stats, normalize=False)
        return merged

    def _set_torch_stats(self):
        self.state_mean = self.stats["s_mean"]
        self.state_std = self.stats["s_std"]
        self.action_mean = self.stats["a_mean"]
        self.action_std = self.stats["a_std"]
        self.state_mean_torch = torch.as_tensor(self.state_mean, device=self.device)
        self.state_std_torch = torch.as_tensor(self.state_std, device=self.device)
        self.action_mean_torch = torch.as_tensor(self.action_mean, device=self.device)
        self.action_std_torch = torch.as_tensor(self.action_std, device=self.device)

    def normalize_state(self, state):
        if isinstance(state, dict):
            state = dict(state)
            state["proprio"] = self.normalize_state(state["proprio"])
            return state
        if isinstance(state, torch.Tensor):
            return (state - self.state_mean_torch) / (self.state_std_torch + 1e-6)
        return (state - self.state_mean) / (self.state_std + 1e-6)

    def unnormalize_state(self, state):
        if isinstance(state, dict):
            state = dict(state)
            state["proprio"] = self.unnormalize_state(state["proprio"])
            return state
        if isinstance(state, torch.Tensor):
            return state * (self.state_std_torch + 1e-6) + self.state_mean_torch
        return state * (self.state_std + 1e-6) + self.state_mean

    def normalize_action(self, action):
        if isinstance(action, torch.Tensor):
            return (action - self.action_mean_torch) / (self.action_std_torch + 1e-6)
        return (action - self.action_mean) / (self.action_std + 1e-6)

    def normalize_state_with_stats(self, state, stats):
        if isinstance(state, dict):
            state = dict(state)
            state["proprio"] = self.normalize_state_with_stats(state["proprio"], stats)
            return state
        if isinstance(state, torch.Tensor):
            mean = torch.as_tensor(stats["s_mean"], device=state.device)
            std = torch.as_tensor(stats["s_std"], device=state.device)
            return (state - mean) / (std + 1e-6)
        return (state - stats["s_mean"]) / (stats["s_std"] + 1e-6)

    def normalize_action_with_stats(self, action, stats):
        if isinstance(action, torch.Tensor):
            mean = torch.as_tensor(stats["a_mean"], device=action.device)
            std = torch.as_tensor(stats["a_std"], device=action.device)
            return (action - mean) / (std + 1e-6)
        return (action - stats["a_mean"]) / (stats["a_std"] + 1e-6)

    def renormalize_state_to_stats(self, state, target_stats):
        raw_state = self.unnormalize_state(state)
        return self.normalize_state_with_stats(raw_state, target_stats)

    def renormalize_action_to_stats(self, action, target_stats):
        raw_action = self.unnormalize_action(action)
        return self.normalize_action_with_stats(raw_action, target_stats)

    def unnormalize_action(self, action):
        if isinstance(action, torch.Tensor):
            return action * (self.action_std_torch + 1e-6) + self.action_mean_torch
        return action * (self.action_std + 1e-6) + self.action_mean

    def _normalize_storage(self):
        self.storage["state"] = self.normalize_state(self.storage["state"])
        self.storage["next_state"] = self.normalize_state(self.storage["next_state"])
        self.storage["action"] = self.normalize_action(self.storage["action"])
        self.storage["next_action"] = self.normalize_action(self.storage["next_action"])

    def apply_stats(self, stats, normalize=True):
        self.stats = {key: np.asarray(value, dtype=np.float32) for key, value in stats.items()}
        self._set_torch_stats()
        if normalize:
            self._normalize_storage()
        return self

    @staticmethod
    def compute_joint_stats(pipes):
        states = np.concatenate([pipe.storage["state"] for pipe in pipes], axis=0)
        actions = np.concatenate([pipe.storage["action"] for pipe in pipes], axis=0)
        return {
            "s_mean": np.mean(states, axis=0).astype(np.float32),
            "s_std": np.std(states, axis=0).astype(np.float32),
            "a_mean": np.mean(actions, axis=0).astype(np.float32),
            "a_std": np.std(actions, axis=0).astype(np.float32),
        }

    def save_stats(self, path):
        stats = {
            "state_mean": self.state_mean.tolist(),
            "state_std": self.state_std.tolist(),
            "action_mean": self.action_mean.tolist(),
            "action_std": self.action_std.tolist(),
            "image_size": list(self.image_size),
            "rgb_crop": list(self.rgb_crop),
        }
        with open(path, "w") as f:
            json.dump(stats, f, indent=2)

    def _augment_pair(self, image, next_image):
        if torch.rand(1, device=self.device) >= 0.5:
            return image, next_image
        seed = torch.randint(0, 2**32, (1,), device=self.device).item()
        devices = [self.device.index if self.device.index is not None else torch.cuda.current_device()] if self.device.type == "cuda" else []
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(seed)
            image = self.image_aug(image)
            torch.manual_seed(seed)
            next_image = self.image_aug(next_image)
        return image, next_image

    def _augment_batch(self, image, next_image):
        images, next_images = [], []
        for img, nxt in zip(image, next_image):
            img, nxt = self._augment_pair(img, nxt)
            images.append(img)
            next_images.append(nxt)
        return torch.stack(images, dim=0), torch.stack(next_images, dim=0)

    def _prepare_image_batch(self, image):
        if image.dtype == torch.uint8:
            return image.float() / 255.0
        return image.float()

    def batch_to_device(self, batch):
        state, action, next_state, reward, not_done, next_action = batch
        state = {
            "proprio": state["proprio"].to(self.device, non_blocking=True),
            "image": self._prepare_image_batch(state["image"].to(self.device, non_blocking=True)),
        }
        next_state = {
            "proprio": next_state["proprio"].to(self.device, non_blocking=True),
            "image": self._prepare_image_batch(next_state["image"].to(self.device, non_blocking=True)),
        }
        image, next_image = self._augment_batch(state["image"], next_state["image"])
        return (
            {"proprio": state["proprio"], "image": image},
            action.to(self.device, non_blocking=True),
            {"proprio": next_state["proprio"], "image": next_image},
            reward.to(self.device, non_blocking=True).view(-1, 1),
            not_done.to(self.device, non_blocking=True).view(-1, 1),
            next_action.to(self.device, non_blocking=True),
        )

    def make_dataloader(self, batch_size, epoch_size=500, num_workers=4, prefetch_factor=4):
        sampler = RandomSampler(self, replacement=True, num_samples=int(batch_size) * int(epoch_size))
        loader_kwargs = {}
        if num_workers > 0:
            loader_kwargs["prefetch_factor"] = prefetch_factor
            loader_kwargs["persistent_workers"] = True
        return DataLoader(
            self,
            sampler=sampler,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True,
            **loader_kwargs,
        )
