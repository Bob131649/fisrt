"""
Based on https://github.com/sfujim/BCQ
"""
import numpy as np
import torch
import numpy.linalg as LA

GOAL = np.array([33.0, 25.0])
GOAL_Dist = 0.5

class ReplayBuffer(object):
    def __init__(self, env_name, state_dim, action_dim, device, max_size=int(2e7)):
        self.online_size = int(max_size * 0.2)
        self.pre_size = max_size
        self.max_size = max_size #+ self.online_size
        self.ptr = 0
        self.size = 0
        self.device = torch.device(device)
        self.env_name = env_name

        self.storage = dict()
        self.storage['state'] = np.zeros((self.max_size, state_dim))
        self.storage['action'] = np.zeros((self.max_size, action_dim))
        self.storage['next_action'] = np.zeros((self.max_size, action_dim))
        self.storage['next_state'] = np.zeros((self.max_size, state_dim))
        self.storage['reward'] = np.zeros((self.max_size, 1))
        self.storage['not_done'] = np.zeros((self.max_size, 1))

        self.stats = {}

        self.min_r, self.max_r = 0, 0

    def add(self, state, action, next_state, reward, done, next_action):
        self.storage['state'][self.ptr] = state.copy()
        self.storage['action'][self.ptr] = action.copy()
        self.storage['next_action'][self.ptr] = next_action.copy()
        self.storage['next_state'][self.ptr] = next_state.copy()

        self.storage['reward'][self.ptr] = reward
        self.storage['not_done'][self.ptr] = 1. - done

        self.size = min(self.size + 1, self.max_size)

        # if self.size == self.max_size:
        #     self.ptr = self.ptr = (self.ptr + 1) % self.online_size + self.pre_size
        # else:
        self.ptr = (self.ptr + 1) % self.max_size
            
    def sample(self, batch_size):
        ind = np.random.randint(0, self.size, size=batch_size)

        return (
            torch.FloatTensor(self.storage['state'][ind]).to(self.device),
            torch.FloatTensor(self.storage['action'][ind]).to(self.device),
            torch.FloatTensor(self.storage['next_state'][ind]).to(self.device),
            torch.FloatTensor(self.storage['reward'][ind]).to(self.device).view(-1,1),
            torch.FloatTensor(self.storage['not_done'][ind]).to(self.device).view(-1,1),
            torch.FloatTensor(self.storage['next_action'][ind]).to(self.device),
        )

    def save(self, filename):
        np.save("./buffers/" + filename + ".npy", self.storage)

    def normalize_state(self, state):
        if isinstance(state, torch.Tensor):
            return (state - self.state_mean_torch)/(self.state_std_torch+0.000001)
        else:
            return (state - self.state_mean)/(self.state_std+0.000001)

    def unnormalize_state(self, state):
        if isinstance(state, torch.Tensor):
            return state * (self.state_std_torch+0.000001) + self.state_mean_torch
        else:
            return state * (self.state_std+0.000001) + self.state_mean

    def normalize_action(self, action):
        if isinstance(action, torch.Tensor):
            return (action - self.action_mean_torch)/(self.action_std_torch+0.000001)
        else:
            return (action - self.action_mean)/(self.action_std+0.000001)

    def unnormalize_action(self, action):
        if isinstance(action, torch.Tensor):
            return action * (self.action_std_torch+0.000001) + self.action_mean_torch
        else:
            return action * (self.action_std+0.000001) + self.action_mean

    def renormalize(self):
        self.storage['state'] = self.unnormalize_state(self.storage['state'])
        self.storage['next_state'] = self.unnormalize_state(self.storage['next_state'])
        self.storage['action'] = self.unnormalize_action(self.storage['action'])
        self.storage['next_action'] = self.unnormalize_action(self.storage['next_action'])

        self.action_mean = np.mean(self.storage['action'][:self.size], axis=0)
        self.action_std = np.std(self.storage['action'][:self.size], axis=0)
        self.state_mean = np.mean(self.storage['state'][:self.size], axis=0)
        self.state_std = np.std(self.storage['state'][:self.size], axis=0)        

        self.storage['state'] = self.normalize_state(self.storage['state'])
        self.storage['next_state'] = self.normalize_state(self.storage['next_state'])
        self.storage['action'] = self.normalize_action(self.storage['action'])
        self.storage['next_action'] = self.normalize_action(self.storage['next_action'])

        self.min_r = self.storage['reward'].min()
        self.max_r = self.storage['reward'].max()

    def load(self, data, normalize=True, stats=None):
        assert('next_observations' in data.keys())

        for i in range(data['observations'].shape[0]-1):
            dist_to_nextobs = LA.norm(data['next_observations'][i][0:2] - data['observations'][i+1][0:2])
            if dist_to_nextobs > 0:
                # print(i, dist_to_nextobs, data['terminals'][i])
                continue
            
            next_action = data['actions'][i + 1]

            reward = data['rewards'][i]
            if 'antmaze' in self.env_name:
            #     if data['rewards'][i] != 0:
            #         data['terminals'][i] = True
                next_state_pos = data['next_observations'][i][:2]
                distance_to_goal = LA.norm(next_state_pos - GOAL)
                if distance_to_goal < GOAL_Dist:
                    reward = 1
                else:
                    reward = 0

            self.add(data['observations'][i], data['actions'][i], data['next_observations'][i],
                    reward, data['terminals'][i], next_action)

        if stats is None:
            print('compute stats')
            self.action_mean = np.mean(self.storage['action'][:self.size], axis=0)
            self.action_std = np.std(self.storage['action'][:self.size], axis=0)
            self.state_mean = np.mean(self.storage['state'][:self.size], axis=0)
            self.state_std = np.std(self.storage['state'][:self.size], axis=0)
            self.stats['s_mean'] = self.state_mean
            self.stats['a_mean'] = self.action_mean
            self.stats['s_std'] = self.state_std
            self.stats['a_std'] = self.action_std
        else:
            print('use input stats')
            self.state_mean = self.stats['s_mean']
            self.state_std = self.stats['s_std']
            self.action_mean = self.stats['a_mean']
            self.action_std = self.stats['a_std']

        if normalize:
            print('normalize dataset')
            self.action_mean_torch = torch.FloatTensor(self.action_mean).to(self.device)
            self.action_std_torch = torch.FloatTensor(self.action_std).to(self.device)
            self.state_mean_torch = torch.FloatTensor(self.state_mean).to(self.device)
            self.state_std_torch = torch.FloatTensor(self.state_std).to(self.device)

            self.storage['state'] = self.normalize_state(self.storage['state'])
            self.storage['next_state'] = self.normalize_state(self.storage['next_state'])
            self.storage['action'] = self.normalize_action(self.storage['action'])
            self.storage['next_action'] = self.normalize_action(self.storage['next_action'])