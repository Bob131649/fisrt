#!/usr/bin/env python3

import argparse, os, sys, torch, h5py
import numpy as np
import matplotlib
matplotlib.use("TkAgg")  # needs Tk installed
import matplotlib.pyplot as plt
from matplotlib import cm
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


def get_env_goal(env):
    if "maze2d" in env.spec.id:
        if hasattr(env, "get_target"):
            return np.array(env.get_target()[:2], dtype=np.float32)
        return np.array(getattr(env, "_target", np.zeros(2))[:2], dtype=np.float32)

    base_env = env.unwrapped
    if hasattr(base_env, "target_goal") and base_env.target_goal is not None:
        return np.array(base_env.target_goal[:2], dtype=np.float32)
    if hasattr(base_env, "_goal") and base_env._goal is not None:
        return np.array(base_env._goal[:2], dtype=np.float32)
    if hasattr(base_env, "get_target"):
        return np.array(base_env.get_target()[:2], dtype=np.float32)
    return np.array(getattr(base_env, "_target", np.zeros(2))[:2], dtype=np.float32)


def get_env_start_xy(env):
    base_env = env.unwrapped
    if hasattr(base_env, "get_xy"):
        return np.array(base_env.get_xy()[:2], dtype=np.float32)
    if hasattr(base_env, "sim") and hasattr(base_env.sim, "data"):
        return np.array(base_env.sim.data.qpos[:2], dtype=np.float32)
    if hasattr(base_env, "physics") and hasattr(base_env.physics, "data"):
        return np.array(base_env.physics.data.qpos[:2], dtype=np.float32)
    return np.array([np.nan, np.nan], dtype=np.float32)


def estimate_ope(policy, env, replay_buffer, eval_episodes=50):
    states = []
    for _ in range(eval_episodes):
        state = env.reset()
        states.append(replay_buffer.normalize_state(np.array(state)))
    states = torch.FloatTensor(np.array(states)).to(policy.device)
    return policy.estimate_value(states).mean().item()


def eval_target_policy(policy, replay_buffer, env_name, eval_episodes=10, plot=False, figure_path=None):
    avg_reward = 0.0
    start_states = []
    color_list = cm.rainbow(np.linspace(0, 1, eval_episodes + 2))
    env = build_wrapped_env(env_name)

    if plot:
        plt.clf()

    for i in range(eval_episodes):
        state, done = env.reset(), False
        start_xy = get_env_start_xy(env)
        goal_xy = get_env_goal(env)
        print(
            f"ope eval episode {i}: start={start_xy.tolist()}, "
            f"goal={goal_xy.tolist()}, start_id={getattr(env, 'last_reset_start_idx', None)}"
        )
        states_list = []
        q1_list, q2_list = [], []
        ep_reward = []
        start_states.append(state)

        while not done:
            norm_state = replay_buffer.normalize_state(np.array(state))
            action, q1, q2 = policy.select_action(norm_state)
            q1_list.append(q1)
            q2_list.append(q2)
            env_action = replay_buffer.unnormalize_action(action)
            state, reward, done, _ = env.step(env_action)
            avg_reward += reward
            states_list.append(state)
            ep_reward.append(reward)

        print('   ---', np.mean(ep_reward), q1_list[0], q1_list[-1], np.mean(q1_list))
        print('---', state[0], state[1], len(states_list))

        if plot and states_list:
            states_list = np.array(states_list)
            plt.scatter(states_list[:, 0], states_list[:, 1], color=color_list[i], alpha=0.1)

    if plot and start_states:
        start_states = np.array(start_states)
        plt.scatter(start_states[:, 0], start_states[:, 1], color='red')
        if figure_path is None:
            figure_path = './ope_eval_fig'
        plt.savefig(figure_path)

    avg_reward /= eval_episodes
    normalized_score = env.get_normalized_score(avg_reward)
    env.close()

    info = {'AverageReturn': avg_reward, 'NormReturn': normalized_score}
    print("---------------------------------------")
    print(f"OPE rollout evaluation over {eval_episodes} episodes: {avg_reward:.3f}, {normalized_score:.3f}")
    print("---------------------------------------")
    return info


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
    parser.add_argument('--plot', action='store_true')
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
            crit_list, value_list, vpred_list, w_list = [], [], [], []
            iter_id = 0
            np.random.seed()
            with tqdm(dataloader, desc='Batch', leave=False) as tepoch:
                for batch in tepoch:
                    sample_w, value_pred, crit_loss, value_loss, _ = policy.train_step(batch, iter_id)
                    iter_id += 1
                    crit_list.append(crit_loss)
                    if value_loss is not None:
                        value_list.append(value_loss)
                    if value_pred is not None:
                        vpred_list.append(value_pred)
                    w_list.append(np.mean(sample_w))
                    tepoch.set_postfix(
                        w=np.mean(w_list),
                        crit=np.mean(crit_list),
                        value=np.mean(value_list) if value_list else 0.0,
                    )

            if epoch_idx % args.save_freq == 0 and args.save_model and epoch_idx != 0:
                policy.save('model', folder_name)

            logger.record_tabular('Training Epochs', int(epoch_idx))
            policy.eval()
            logger.record_tabular('OPE_Estimate', estimate_ope(policy, env, d4rl_dataset))
            rollout_info = eval_target_policy(
                policy,
                d4rl_dataset,
                args.env_name,
                plot=args.plot,
                figure_path=os.path.join(folder_name, f"ope_eval_fig_epoch{epoch_idx:04d}.png"),
            )
            policy.train()
            for k, v in rollout_info.items():
                logger.record_tabular(k, v)
            logger.record_tabular('L_Crit', np.mean(crit_list))
            logger.record_tabular('L_Value', np.mean(value_list) if value_list else 0.0)
            logger.record_tabular('V_Pred', np.mean(vpred_list) if vpred_list else 0.0)
            logger.record_tabular('Weight', np.mean(w_list))
            logger.dump_tabular()

            tglobal.set_postfix(
                w=np.mean(w_list),
                crit=np.mean(crit_list),
                value=np.mean(value_list) if value_list else 0.0,
            )

    policy.save('model', folder_name)

    if terminal_log_file is not None:
        terminal_log_file.close()
