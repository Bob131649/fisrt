#!/usr/bin/env python3

import argparse, os, sys, torch, h5py
import numpy as np
import matplotlib
matplotlib.use("TkAgg")  # needs Tk installed
import matplotlib.pyplot as plt
from matplotlib import cm
import algos.algos_v2 as algos
import gym, d4rl
from tqdm.auto import tqdm
from logger import logger, setup_logger
from dataset.d4rl_dataset import D4rlDataset
from fixed_reset_wrapper import FixedResetWrapper
from torch.utils.data import DataLoader
from networks.net_v2 import OPEValue


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


def build_wrapped_env(env_name, start_mode="cycle", start_noise_scale=0.5, goal_noise_scale=0.1):
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


def eval_policy(policy, env, replay_buffer, eval_episodes=10, plot=False):
    avg_reward = 0.
    plt.clf()
    start_states = []
    color_list = cm.rainbow(np.linspace(0, 1, eval_episodes+2))
    env = build_wrapped_env(args.env_name)

    for i in range(eval_episodes):
        state, done = env.reset(), False
        start_xy = get_env_start_xy(env)
        goal_xy = get_env_goal(env)
        print(
            f"eval episode {i}: start={start_xy.tolist()}, "
            f"goal={goal_xy.tolist()}, start_id={getattr(env, 'last_reset_start_idx', None)}"
        )
        states_list = []
        q1_list, q2_list = [], []
        ep_reward = []
        start_states.append(state)
        while not done:
            state = replay_buffer.normalize_state(np.array(state))
            action, q1, q2 = policy.select_action(state)
            q1_list.append(q1)
            q2_list.append(q2)
            action = replay_buffer.unnormalize_action(action)

            state, reward, done, _ = env.step(action)
            avg_reward += reward
            states_list.append(state)
            ep_reward.append(reward)
        print('   ---', np.mean(ep_reward), q1_list[0], q1_list[-1], np.mean(q1_list))
        print('---', state[0], state[1], len(states_list))

        states_list = np.array(states_list)

        if plot:
            plt.scatter(states_list[:,0], states_list[:,1], color = color_list[i], alpha=0.1)
            plt.scatter(8, 10, color = 'white', alpha=0.1)
            plt.scatter(2, 0, color = 'white', alpha=0.1)
    if plot:
        start_states = np.array(start_states)
        plt.scatter(start_states[:,0], start_states[:,1], color='red')
        plt.savefig('./vae_eval_fig') 

    avg_reward /= eval_episodes
    normalized_score = env.get_normalized_score(avg_reward)
    env.close()

    info = {'AverageReturn': avg_reward, 'NormReturn': normalized_score}
    print ("---------------------------------------")
    print(f"Evaluation over {eval_episodes} episodes: {avg_reward:.3f}, {normalized_score:.3f}")
    print ("---------------------------------------")
    return info


def apply_reference_reward_shaping(
    replay_buffer,
    state_dim,
    ope_ref_dir,
    ope_ref_name,
    discount,
    shape_lambda,
    shape_clip,
    device,
    batch_size=8192,
):
    value_path = os.path.join(ope_ref_dir, f"{ope_ref_name}_value.pth")
    if not os.path.isfile(value_path):
        raise FileNotFoundError(
            f"OPE value checkpoint not found: {value_path}. "
            "This code uses the OPE value network as Vref(s); the critic checkpoint is Q(s,a)."
        )

    ref_value = OPEValue(state_dim).to(device)
    ref_value.load_state_dict(torch.load(value_path, map_location=device))
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
                value = ref_value(state).clamp(-shape_clip, shape_clip)
                values.append(value.cpu().numpy())
        return np.concatenate(values, axis=0)

    phi = predict_phi(replay_buffer.states)
    next_phi = predict_phi(replay_buffer.next_states)
    original_rewards = replay_buffer.rewards.astype(np.float32)
    shaping = discount * replay_buffer.not_dones * next_phi - phi
    replay_buffer.rewards = (original_rewards + shape_lambda * shaping).astype(np.float32)

    print(
        "applied reference reward shaping: "
        f"reward mean {original_rewards.mean():.4f}->{replay_buffer.rewards.mean():.4f}, "
        f"shape mean {shaping.mean():.4f}, "
        f"shape range [{shaping.min():.4f}, {shaping.max():.4f}]"
    )
    return replay_buffer.rewards.min() / (1 - discount), replay_buffer.rewards.max() / (1 - discount)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Additional parameters
    parser.add_argument("--ExpID", default=427, type=int)              # Experiment ID
    parser.add_argument('--log_dir', default='./results/', type=str)    # Logging directory
    parser.add_argument("--load_model", default=0, type=int)          # Load model and optimizer parameters
    parser.add_argument("--save_model", default=True, type=bool)        # Save model and optimizer parameters
    parser.add_argument("--save_freq", default=50, type=int)           # How often it saves the model (epoch)
    parser.add_argument("--env_name", default="maze2d-large-v1")     # OpenAI gym environment name
    parser.add_argument("--dataset_path", default="", type=str)      # Optional custom dataset path (.hdf5)
    parser.add_argument("--seed", default=789, type=int)                  # Sets Gym, PyTorch and Numpy seeds
    parser.add_argument("--eval_freq", default=10000, type=int)           # How often (time steps) we evaluate
    parser.add_argument("--max_timesteps", default=1e6, type=int)      # Max time steps to run environment for
    parser.add_argument('--batch_size', default=512, type=int)
    parser.add_argument('--vae_lr', default=2e-4, type=float)	        # action policy (VAE) learning rate
    parser.add_argument('--actor_lr', default=2e-4, type=float)	        # latent policy learning rate
    parser.add_argument('--critic_lr', default=2e-4, type=float)	    # critic learning rate
    parser.add_argument('--tau', default=0.005, type=float)	            # delayed learning rate
    parser.add_argument('--discount', default=0.99, type=float)	        # discount factor

    parser.add_argument('--expectile', default=0.5, type=float)	        # expectile to compute weight for samples
    parser.add_argument('--kl_beta', default=1.0, type=float)	            # weight for kl loss to train CVAE
    parser.add_argument('--max_latent_action', default=0.675, type=float)	# maximum value for the latent policy
    parser.add_argument('--doubleq_min', default=1.0, type=float)         # weight for the minimum Q value
    parser.add_argument('--no_noise', action='store_true')              # adding noise to the latent policy or not

    parser.add_argument('--plot', action='store_true')
    parser.add_argument('--device', default='cuda', type=str)
    parser.add_argument("--ope_ref_dir", default="", type=str)
    parser.add_argument("--ope_ref_name", default="model", type=str)
    parser.add_argument("--shape_lambda", default=1.0, type=float)
    parser.add_argument("--shape_clip", default=10.0, type=float)
    parser.add_argument("--shape_batch_size", default=8192, type=int)
    parser.add_argument(
        "--terminal_log_path",
        default="",
        type=str,
        help="Optional file path to save terminal print output.",
    )

    args = parser.parse_args()

    # Setup Logging
    file_name = f"Exp{args.ExpID:04d}/{args.env_name}"
    folder_name = os.path.join(args.log_dir, file_name)
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

    if os.path.exists(os.path.join(folder_name, 'progress.csv')):
        print('exp file already exist')
        # raise AssertionError

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

    # Setup Environment
    env = build_wrapped_env(args.env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    # Set seeds
    env.seed(args.seed)
    env.action_space.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Load Dataset
    if args.dataset_path:
        if not os.path.isfile(args.dataset_path):
            raise FileNotFoundError(f"Dataset file not found: {args.dataset_path}")
        print(f"loading custom dataset from: {args.dataset_path}")
        raw_dataset = load_hdf5_dataset(args.dataset_path)
        dataset = d4rl.qlearning_dataset(env, dataset=raw_dataset)
    else:
        dataset = d4rl.qlearning_dataset(env)  # Load d4rl dataset
    if 'antmaze' in args.env_name:
        dataset['rewards'] = (dataset['rewards']*100) #(dataset['rewards']*300)
        min_v = 0
        max_v = 100
    else:
        dataset['rewards'] = dataset['rewards'] - dataset['rewards'].min()
        dataset['rewards'] = dataset['rewards']/dataset['rewards'].max()
        min_v = dataset['rewards'].min()/(1-args.discount)
        max_v = dataset['rewards'].max()/(1-args.discount)

    d4rl_dataset = D4rlDataset(dataset, args.env_name)
    if args.ope_ref_dir:
        min_v, max_v = apply_reference_reward_shaping(
            d4rl_dataset,
            state_dim=state_dim,
            ope_ref_dir=args.ope_ref_dir,
            ope_ref_name=args.ope_ref_name,
            discount=args.discount,
            shape_lambda=args.shape_lambda,
            shape_clip=args.shape_clip,
            device=args.device,
            batch_size=args.shape_batch_size,
        )

    dataloader = DataLoader(
        d4rl_dataset,
        sampler=torch.utils.data.RandomSampler(d4rl_dataset, num_samples=args.batch_size*args.eval_freq, replacement=True),
        batch_size=args.batch_size,
        num_workers=2,
        pin_memory=True,
        drop_last=True,
    )

    latent_dim = int(action_dim*2)
    policy = algos.Latent(state_dim, action_dim, latent_dim, min_v, max_v,
                        device=args.device, discount=args.discount, tau=args.tau, 
                        vae_lr=args.vae_lr, actor_lr=args.actor_lr, critic_lr=args.critic_lr, 
                        max_latent_action=args.max_latent_action, expectile=args.expectile, kl_beta=args.kl_beta, 
                        doubleq_min=args.doubleq_min)

    num_itr = int(args.max_timesteps/args.eval_freq)
    with tqdm(range(num_itr), desc='Epoch', leave=False) as tglobal:
        for epoch_idx in tglobal:
            act_list, crit_list, rec_list, kl_list, w_list = [], [], [], [], []
            iter = 0
            np.random.seed()
            with tqdm(dataloader, desc='Batch', leave=False) as tepoch:
                for batch in tepoch:
                    # Train
                    sample_w, act_loss, crit_loss, recons_loss, kl_loss = policy.train_step(batch, iter)
                    iter += 1
                    crit_list.append(crit_loss)
                    if act_loss is not None:
                        act_list.append(act_loss)
                    if recons_loss is not None:
                        rec_list.append(recons_loss)
                    if kl_loss is not None:
                        kl_list.append(kl_loss)

                    w_list.append(np.mean(sample_w))
                    tepoch.set_postfix(w=np.mean(w_list), act=np.mean(act_list), crit=np.mean(crit_list), 
                                        rec=np.mean(rec_list), kl=np.mean(kl_list))

            # Save Model
            if epoch_idx % args.save_freq == 0 and args.save_model and epoch_idx != 0:
                policy.save('model', folder_name)

            # Eval
            logger.record_tabular('Training Epochs', int(epoch_idx))
            print('done')
            policy.eval()
            policy.copy_bn_param()
            info = eval_policy(policy, env, d4rl_dataset, plot=args.plot)
            policy.train()

            for k, v in info.items():
                logger.record_tabular(k, v)
            logger.record_tabular('L_Act', np.mean(act_list))
            logger.record_tabular('L_Crit', np.mean(crit_list))
            logger.record_tabular('Weight', np.mean(w_list))
            logger.record_tabular('kl', policy.kl_beta)

            logger.dump_tabular()
            tglobal.set_postfix(w=np.mean(w_list), act=np.mean(act_list), crit=np.mean(crit_list), 
                                           rec=np.mean(rec_list), kl=np.mean(kl_list))

    policy.save('model', folder_name)

    if terminal_log_file is not None:
        terminal_log_file.close()
