#!/usr/bin/env python3

import argparse, os, sys, torch, h5py
import numpy as np
import gym, d4rl
import algos.algos_v2 as algos
from tqdm.auto import tqdm
from logger import logger, setup_logger
from dataset.d4rl_dataset import D4rlDataset
from fixed_reset_wrapper import FixedResetWrapper
from torch.utils.data import DataLoader


class TeeStream:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for stream in self.streams:
            stream.write(data)
            stream.flush()

    def flush(self):
        for stream in self.streams:
            stream.flush()


def load_hdf5_dataset(dataset_path):
    dataset = {}
    with h5py.File(dataset_path, "r") as f:
        for key in f.keys():
            obj = f[key]
            if isinstance(obj, h5py.Dataset):
                dataset[key] = obj[:]
            elif isinstance(obj, h5py.Group):
                for sub_key in obj.keys():
                    dataset[f"{key}/{sub_key}"] = obj[sub_key][:]

    required_keys = ["observations", "actions", "rewards", "terminals"]
    missing_keys = [key for key in required_keys if key not in dataset]
    if missing_keys:
        raise ValueError(
            f"Dataset file is missing required keys: {', '.join(missing_keys)}"
        )
    return dataset


def build_wrapped_env(env_name, start_mode="cycle", start_noise_scale=0.1, goal_noise_scale=0):
    base_env = gym.make(env_name)
    return FixedResetWrapper(
        base_env,
        env_name=env_name,
        start_mode=start_mode,
        start_noise_scale=start_noise_scale,
        goal_noise_scale=goal_noise_scale,
    )


def estimate_ope(policy, env, replay_buffer, eval_episodes=50):
    states = []
    for _ in range(eval_episodes):
        state = env.reset()
        states.append(replay_buffer.normalize_state(np.array(state)))
    states = torch.FloatTensor(np.array(states)).to(policy.device)
    return policy.estimate_value(states).mean().item()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ExpID", default=427, type=int)
    parser.add_argument('--log_dir', default='./results/', type=str)
    parser.add_argument("--load_model", default=0, type=int)
    parser.add_argument("--save_model", default=True, type=bool)
    parser.add_argument("--save_freq", default=50, type=int)
    parser.add_argument("--env_name", default="maze2d-large-v1")
    parser.add_argument("--dataset_path", default="", type=str)
    parser.add_argument("--seed", default=789, type=int)
    parser.add_argument("--eval_freq", default=10000, type=int)
    parser.add_argument("--max_timesteps", default=1e6, type=int)
    parser.add_argument('--batch_size', default=512, type=int)
    parser.add_argument('--critic_lr', default=2e-4, type=float)
    parser.add_argument('--tau', default=0.005, type=float)
    parser.add_argument('--discount', default=0.99, type=float)
    parser.add_argument('--max_latent_action', default=0.675, type=float)
    parser.add_argument('--doubleq_min', default=1.0, type=float)
    parser.add_argument('--device', default='cuda', type=str)
    parser.add_argument("--target_policy_dir", required=True, type=str)
    parser.add_argument("--target_policy_name", default="model", type=str)
    parser.add_argument(
        "--target_policy_mode",
        default="lapo",
        choices=["lapo", "vae", "vae_bc"],
    )
    parser.add_argument(
        "--terminal_log_path",
        default="",
        type=str,
        help="Optional file path to save terminal print output.",
    )
    args = parser.parse_args()


    file_name = f"Exp{args.ExpID:04d}/{args.env_name}_ope"
    folder_name = os.path.join(args.log_dir, file_name)
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

    terminal_log_file = None
    if args.terminal_log_path:
        terminal_log_path = os.path.abspath(args.terminal_log_path)
        os.makedirs(os.path.dirname(terminal_log_path), exist_ok=True)
        terminal_log_file = open(terminal_log_path, "a", buffering=1)
        sys.stdout = TeeStream(sys.stdout, terminal_log_file)
        sys.stderr = TeeStream(sys.stderr, terminal_log_file)
        print(f"terminal output is being saved to: {terminal_log_path}")

    variant = vars(args)
    variant.update(node=os.uname()[1])
    setup_logger(os.path.basename(folder_name), variant=variant, log_dir=folder_name)

    env = build_wrapped_env(args.env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    env.seed(args.seed)
    env.action_space.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if args.dataset_path:
        if not os.path.isfile(args.dataset_path):
            raise FileNotFoundError(f"Dataset file not found: {args.dataset_path}")
        print(f"loading custom dataset from: {args.dataset_path}")
        raw_dataset = load_hdf5_dataset(args.dataset_path)
        dataset = d4rl.qlearning_dataset(env, dataset=raw_dataset)
    else:
        dataset = d4rl.qlearning_dataset(env)

    if 'antmaze' in args.env_name:
        dataset['rewards'] = dataset['rewards'] * 100
        min_v = 0
        max_v = 100
    else:
        dataset['rewards'] = dataset['rewards'] - dataset['rewards'].min()
        dataset['rewards'] = dataset['rewards'] / dataset['rewards'].max()
        min_v = dataset['rewards'].min() / (1 - args.discount)
        max_v = dataset['rewards'].max() / (1 - args.discount)

    d4rl_dataset = D4rlDataset(dataset, args.env_name)
    dataloader = DataLoader(
        d4rl_dataset,
        sampler=torch.utils.data.RandomSampler(
            d4rl_dataset,
            num_samples=args.batch_size * args.eval_freq,
            replacement=True,
        ),
        batch_size=args.batch_size,
        num_workers=2,
        pin_memory=True,
        drop_last=True,
    )

    latent_dim = int(action_dim * 2)
    policy = algos.Latent(
        state_dim,
        action_dim,
        latent_dim,
        min_v,
        max_v,
        device=args.device,
        discount=args.discount,
        tau=args.tau,
        critic_lr=args.critic_lr,
        max_latent_action=args.max_latent_action,
        doubleq_min=args.doubleq_min,
        target_policy_dir=args.target_policy_dir,
        target_policy_name=args.target_policy_name,
        target_policy_mode=args.target_policy_mode,
    )

    if args.load_model:
        policy.load('model', folder_name)

    num_itr = int(args.max_timesteps / args.eval_freq)
    with tqdm(range(num_itr), desc='Epoch', leave=False) as tglobal:
        for epoch_idx in tglobal:
            crit_list, w_list = [], []
            iter_id = 0
            np.random.seed()
            with tqdm(dataloader, desc='Batch', leave=False) as tepoch:
                for batch in tepoch:
                    sample_w, _, crit_loss, _, _ = policy.train_step(batch, iter_id)
                    iter_id += 1
                    crit_list.append(crit_loss)
                    w_list.append(np.mean(sample_w))
                    tepoch.set_postfix(w=np.mean(w_list), crit=np.mean(crit_list))

            if epoch_idx % args.save_freq == 0 and args.save_model and epoch_idx != 0:
                policy.save('model', folder_name)

            logger.record_tabular('Training Epochs', int(epoch_idx))
            policy.eval()
            logger.record_tabular('OPE_Estimate', estimate_ope(policy, env, d4rl_dataset))
            policy.train()
            logger.record_tabular('L_Crit', np.mean(crit_list))
            logger.record_tabular('Weight', np.mean(w_list))
            logger.dump_tabular()

            tglobal.set_postfix(w=np.mean(w_list), crit=np.mean(crit_list))

    policy.save('model', folder_name)

    if terminal_log_file is not None:
        terminal_log_file.close()
