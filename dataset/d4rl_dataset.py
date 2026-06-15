import pickle, torch, os
import numpy as np
from torch.utils.data import Dataset, DataLoader
import numpy.linalg as LA
from networks.net_v2 import OPEValue

class D4rlDataset(Dataset):
    """A simple image dataset class."""
    def __init__(self, data, env_name):
        self.n_episodes = 0

        self.states = []
        self.next_states = []
        self.actions = []
        self.rewards = []
        self.not_dones = []

        self.load(data, env_name)
        self.size = len(self.states)

        print('dataset size:', len(self.states))

    def load(self, data, env_name):
        assert('next_observations' in data.keys())
        dataset_size = data['observations'].shape[0]

        for i in range(0, dataset_size):
            self.states.append(data['observations'][i])
            self.next_states.append(data['next_observations'][i])
            self.actions.append(data['actions'][i])
            self.rewards.append([data['rewards'][i]])
            self.not_dones.append([1 - data['terminals'][i]])


        self.states = np.array(self.states)
        self.next_states = np.array(self.next_states)
        self.actions = np.array(self.actions)
        self.rewards = np.array(self.rewards)
        self.not_dones = np.array(self.not_dones)

        self.action_mean = np.mean(self.actions, axis=0)
        self.action_std = np.std(self.actions, axis=0)
        self.state_mean = np.mean(self.states, axis=0)
        self.state_std = np.std(self.states, axis=0)

        self.raw_states = self.states.copy()
        self.raw_next_states = self.next_states.copy()

        self.states = self.normalize_state(self.states)
        self.next_states = self.normalize_state(self.next_states)
        self.actions = self.normalize_action(self.actions)

    def normalize_state(self, state):
        return (state - self.state_mean)/(self.state_std+0.000001)

    def unnormalize_state(self, state):
        return state * (self.state_std+0.000001) + self.state_mean

    def normalize_action(self, action):
        return (action - self.action_mean)/(self.action_std+0.000001)

    def unnormalize_action(self, action):
        return action * (self.action_std+0.000001) + self.action_mean

    def apply_reference_reward_shaping(
        self,
        state_dim,
        ope_ref_dir,
        ope_ref_name,
        discount,
        shape_lambda,
        shape_clip,
        device,
        batch_size=512,
    ):
        value_path = os.path.join(ope_ref_dir, f"{ope_ref_name}_value.pth")
        if not os.path.isfile(value_path):
            raise FileNotFoundError(
                f"OPE value checkpoint not found: {value_path}. "
                "This code uses the OPE value network as Vref(s); the critic checkpoint is Q(s,a)."
            )

        ref_value = OPEValue(state_dim).to(device)
        value_state_dict = torch.load(value_path, map_location=device)
        # if "l5.weight" not in value_state_dict:
        #     for src_idx, dst_idx in zip(range(1, 5), range(5, 9)):
        #         value_state_dict[f"l{dst_idx}.weight"] = value_state_dict[
        #             f"l{src_idx}.weight"
        #         ].clone()
        #         value_state_dict[f"l{dst_idx}.bias"] = value_state_dict[
        #             f"l{src_idx}.bias"
        #         ].clone()
        ref_value.load_state_dict(value_state_dict)
        ref_value.eval()

        def predict_phi(states):
            values = []
            with torch.no_grad():
                for start in range(0, len(states), batch_size):
                    state = torch.as_tensor(
                        states[start : start + batch_size],
                        dtype=torch.float32,
                        device=device,
                    )
                    v1, v2 = ref_value(state)
                    value = torch.min(v1, v2).clamp(-shape_clip, shape_clip)
                    values.append(value.cpu().numpy())
            return np.concatenate(values, axis=0)

        phi = predict_phi(self.states)
        next_phi = predict_phi(self.next_states)
        original_rewards = self.rewards.astype(np.float32)
        shaping = discount * self.not_dones * next_phi - phi
        self.rewards = (original_rewards + shape_lambda * shaping).astype(np.float32)

        print(
            "applied reference reward shaping: "
            f"reward mean {original_rewards.mean():.4f}->{self.rewards.mean():.4f}, "
            f"shape mean {shaping.mean():.4f}, "
            f"shape range [{shaping.min():.4f}, {shaping.max():.4f}]"
        )
        return self.rewards.min() / (1 - discount), self.rewards.max() / (1 - discount)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        sample_idx = idx
        sample = {
            'state': self.states[sample_idx],
            'action': self.actions[sample_idx],
            'next_state': self.next_states[sample_idx],
            'raw_state': self.raw_states[sample_idx],
            'raw_next_state': self.raw_next_states[sample_idx],
            'reward': self.rewards[sample_idx],
            'not_done': self.not_dones[sample_idx],
        }

        return sample
