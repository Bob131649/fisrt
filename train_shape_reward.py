#!/usr/bin/env python3

import argparse, os, sys, torch, h5py
import numpy as np
import matplotlib
matplotlib.use("TkAgg")  # needs Tk installed
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib import cm
import algos.algos_v2 as algos
import gym, d4rl
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


def load_qlearning_dataset(env, dataset_path):
    if not os.path.isfile(dataset_path):
        raise FileNotFoundError(f"Dataset file not found: {dataset_path}")
    raw_dataset = load_hdf5_dataset(dataset_path)
    return d4rl.qlearning_dataset(env, dataset=raw_dataset)


def compute_state_normalizer(env, dataset_paths):
    observations = []
    for dataset_path in dataset_paths:
        print(f"loading OPE reference normalizer dataset from: {dataset_path}")
        ref_dataset = load_qlearning_dataset(env, dataset_path)
        observations.append(ref_dataset["observations"])
    observations = np.concatenate(observations, axis=0)
    return np.mean(observations, axis=0), np.std(observations, axis=0)


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


def build_heatmap(xy, values, grid_size, agg="p90"):
    x = xy[:, 0]
    y = xy[:, 1]
    x_edges = np.linspace(np.min(x), np.max(x), grid_size + 1)
    y_edges = np.linspace(np.min(y), np.max(y), grid_size + 1)
    x_idx = np.clip(np.digitize(x, x_edges) - 1, 0, grid_size - 1)
    y_idx = np.clip(np.digitize(y, y_edges) - 1, 0, grid_size - 1)
    flat_idx = x_idx * grid_size + y_idx

    order = np.argsort(flat_idx)
    flat_sorted = flat_idx[order]
    values_sorted = values[order]
    unique_bins, start_idx = np.unique(flat_sorted, return_index=True)
    end_idx = np.append(start_idx[1:], values_sorted.shape[0])

    heatmap = np.full((grid_size, grid_size), np.nan, dtype=np.float64)
    counts = np.zeros((grid_size, grid_size), dtype=np.float64)
    for flat_bin, begin, end in zip(unique_bins, start_idx, end_idx):
        bucket = values_sorted[begin:end]
        bucket = bucket[~np.isnan(bucket)]
        if bucket.size == 0:
            continue
        if agg == "mean":
            agg_value = float(np.mean(bucket))
        elif agg == "max":
            agg_value = float(np.max(bucket))
        elif agg == "p90":
            agg_value = float(np.percentile(bucket, 90))
        elif agg == "p75":
            agg_value = float(np.percentile(bucket, 75))
        else:
            raise ValueError(f"Unsupported agg mode: {agg}")

        xi = flat_bin // grid_size
        yi = flat_bin % grid_size
        heatmap[xi, yi] = agg_value
        counts[xi, yi] = end - begin

    return heatmap.T, counts.T, x_edges, y_edges


def predict_dataset_policy_q(policy, states, batch_size):
    values = []
    for start in range(0, states.shape[0], batch_size):
        batch_states = torch.as_tensor(
            states[start : start + batch_size],
            dtype=torch.float32,
            device=policy.device,
        )
        with torch.no_grad():
            if policy.ope_mode:
                actions = policy.target_policy.select_action_tensor(batch_states)
            else:
                latent_actions = policy.actor(batch_states)
                actions = policy.actor_vae.decode(batch_states, z=latent_actions)
            q1, q2 = policy.critic(batch_states, actions)
            q = policy.get_min_q(q1, q2).detach().cpu().numpy().reshape(-1)
        values.append(q)
    return np.concatenate(values, axis=0)


def plot_critic_heatmap(
    policy,
    replay_buffer,
    output_path,
    title,
    raw_states=None,
    starts=None,
    goal=None,
    grid_size=1200,
    batch_size=4096,
    agg="p90",
    vmin=None,
    vmax=None,
    vmin_quantile=0.05,
    vmax_quantile=0.95,
    band_width=None,
):
    if raw_states is None:
        raw_states = replay_buffer.raw_states
    normalized_states = replay_buffer.normalize_state(raw_states)
    xy = raw_states[:, :2]
    values = predict_dataset_policy_q(policy, normalized_states, batch_size)
    heatmap, counts, x_edges, y_edges = build_heatmap(xy, values, grid_size, agg=agg)
    valid_values = heatmap[~np.isnan(heatmap)]

    if valid_values.size > 0 and vmin is None and vmin_quantile is not None:
        vmin = float(np.quantile(valid_values, vmin_quantile))
    if valid_values.size > 0 and vmax is None and vmax_quantile is not None:
        vmax = float(np.quantile(valid_values, vmax_quantile))

    fig, ax = plt.subplots(figsize=(9, 7))
    image_kwargs = {
        "origin": "lower",
        "extent": [x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]],
        "aspect": "auto",
        "cmap": "viridis",
        "vmin": vmin,
        "vmax": vmax,
    }
    if band_width is not None and band_width > 0 and valid_values.size > 0:
        band_min = vmin if vmin is not None else float(np.min(valid_values))
        band_max = vmax if vmax is not None else float(np.max(valid_values))
        band_start = band_width * np.floor(band_min / band_width)
        band_stop = band_width * np.ceil(band_max / band_width)
        boundaries = np.arange(
            band_start, band_stop + band_width, band_width, dtype=np.float64
        )
        if boundaries.size >= 2:
            cmap = plt.get_cmap("viridis", boundaries.size - 1)
            norm = colors.BoundaryNorm(boundaries, cmap.N, clip=True)
            image_kwargs["cmap"] = cmap
            image_kwargs["norm"] = norm
            image_kwargs.pop("vmin", None)
            image_kwargs.pop("vmax", None)

    image = ax.imshow(heatmap, **image_kwargs)
    plt.colorbar(image, ax=ax, label="Critic Q(s, pi(s))")
    if starts is not None and len(starts) > 0:
        starts = np.array(starts, dtype=np.float32)
        ax.scatter(starts[:, 0], starts[:, 1], c="red", s=55, label="Eval starts")
    if goal is not None:
        ax.scatter(goal[0], goal[1], c="white", edgecolors="black", s=85, marker="*", label="Goal")
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)

    print(
        f"saved critic heatmap to: {output_path} "
        f"(q min={np.nanmin(values):.4f}, max={np.nanmax(values):.4f}, mean={np.nanmean(values):.4f})"
    )
    return {
        "CriticHeatmapMeanQ": float(np.nanmean(values)),
        "CriticHeatmapMaxQ": float(np.nanmax(values)),
    }


def eval_policy(
    policy,
    env,
    replay_buffer,
    eval_episodes=10,
    plot=False,
    epoch_idx=None,
    heatmap_dir=None,
    heatmap_raw_states=None,
):
    avg_reward = 0.
    plt.clf()
    start_states = []
    goal_xy = None
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
    if args.plot_critic_heatmap:
        epoch_suffix = "final" if epoch_idx is None else f"epoch_{epoch_idx:04d}"
        heatmap_output_dir = heatmap_dir or "."
        os.makedirs(heatmap_output_dir, exist_ok=True)
        heatmap_info = plot_critic_heatmap(
            policy=policy,
            replay_buffer=replay_buffer,
            output_path=os.path.join(
                heatmap_output_dir, f"critic_heatmap_{epoch_suffix}.png"
            ),
            title=f"Critic Heatmap: {args.env_name} ({epoch_suffix})",
            raw_states=heatmap_raw_states,
            starts=start_states,
            goal=goal_xy,
            grid_size=args.heatmap_grid_size,
            batch_size=args.heatmap_batch_size,
            agg=args.heatmap_agg,
            vmin=args.heatmap_vmin,
            vmax=args.heatmap_vmax,
            vmin_quantile=args.heatmap_vmin_quantile,
            vmax_quantile=args.heatmap_vmax_quantile,
            band_width=args.heatmap_band_width,
        )
        info.update(heatmap_info)
    print ("---------------------------------------")
    print(f"Evaluation over {eval_episodes} episodes: {avg_reward:.3f}, {normalized_score:.3f}")
    print ("---------------------------------------")
    return info

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Additional parameters
    parser.add_argument("--ExpID", default=427, type=int)              # Experiment ID
    parser.add_argument('--log_dir', default='./results/', type=str)    # Logging directory
    parser.add_argument("--load_model", default=0, type=int)          # Load model and optimizer parameters
    parser.add_argument("--save_model", default=True, type=bool)        # Save model and optimizer parameters
    parser.add_argument("--save_freq", default=1, type=int)           # How often it saves the model (epoch)
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

    parser.add_argument('--expectile', default=0.9, type=float)	        # expectile to compute weight for samples
    parser.add_argument('--kl_beta', default=1.0, type=float)	            # weight for kl loss to train CVAE
    parser.add_argument('--max_latent_action', default=0.675, type=float)	# maximum value for the latent policy
    parser.add_argument('--doubleq_min', default=1.0, type=float)         # weight for the minimum Q value
    parser.add_argument('--no_noise', action='store_true')              # adding noise to the latent policy or not

    parser.add_argument('--plot', action='store_true')
    parser.add_argument('--plot_critic_heatmap', action='store_true')
    parser.add_argument('--plot_heatmap', dest='plot_critic_heatmap', action='store_true')
    parser.add_argument(
        '--heatmap_dataset_path',
        default='',
        type=str,
        help='Optional hdf5 dataset used only for critic heatmap states.',
    )
    parser.add_argument('--heatmap_grid_size', default=1200, type=int)
    parser.add_argument('--heatmap_batch_size', default=4096, type=int)
    parser.add_argument('--heatmap_agg', default='mean', choices=['mean', 'max', 'p90', 'p75'])
    parser.add_argument('--heatmap_vmin', default=0, type=float)
    parser.add_argument('--heatmap_vmax', default=100, type=float)
    parser.add_argument('--heatmap_vmin_quantile', default=0.05, type=float)
    parser.add_argument('--heatmap_vmax_quantile', default=0.95, type=float)
    parser.add_argument('--heatmap_band_width', default=None, type=float)
    parser.add_argument('--device', default='cuda', type=str)
    parser.add_argument("--ope_ref_dir", default="", type=str)
    parser.add_argument("--ope_ref_name", default="model", type=str)
    parser.add_argument(
        "--ope_ref_dataset_path",
        nargs="+",
        default=None,
        type=str,
        help="Dataset path(s) used to train the OPE value, for its state normalization.",
    )
    # parser.add_argument("--shape_lambda", default=0.1, type=float)
    parser.add_argument("--shape_clip", default=60.0, type=float)
    parser.add_argument("--shape_batch_size", default=512, type=int)
    parser.add_argument(
        "--terminal_log_path",
        default="",
        type=str,
        help="Optional file path to save terminal print output.",
    )

    args = parser.parse_args()

    # Setup Logging
    file_name = f"Exp{args.ExpID:04d}/{args.env_name}_shaping_reward"
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
    heatmap_raw_states = None
    if args.heatmap_dataset_path:
        if not os.path.isfile(args.heatmap_dataset_path):
            raise FileNotFoundError(
                f"Heatmap dataset file not found: {args.heatmap_dataset_path}"
            )
        print(f"loading heatmap dataset from: {args.heatmap_dataset_path}")
        heatmap_dataset = load_qlearning_dataset(env, args.heatmap_dataset_path)
        heatmap_raw_states = np.array(
            heatmap_dataset["observations"], dtype=np.float32
        )
        print(f"heatmap dataset size: {heatmap_raw_states.shape[0]}")
    elif args.plot_critic_heatmap:
        print("using training dataset for critic heatmap")

    ope_ref_state_mean, ope_ref_state_std = None, None
    if args.ope_ref_dataset_path is not None:
        ope_ref_state_mean, ope_ref_state_std = compute_state_normalizer(
            env, args.ope_ref_dataset_path
        )
    elif args.ope_ref_dir:
        print("no --ope_ref_dataset_path provided; using current dataset normalization for OPE reference value")

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
                        doubleq_min=args.doubleq_min,
                        ope_ref_dir=args.ope_ref_dir, ope_ref_name=args.ope_ref_name,
                        ope_ref_clip=args.shape_clip,
                        ope_ref_state_mean=ope_ref_state_mean,
                        ope_ref_state_std=ope_ref_state_std)

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
                print("save model success")

            # Eval
            logger.record_tabular('Training Epochs', int(epoch_idx))
            print('done')
            policy.eval()
            policy.copy_bn_param()
            info = eval_policy(
                policy,
                env,
                d4rl_dataset,
                plot=args.plot,
                epoch_idx=epoch_idx,
                heatmap_dir=folder_name,
                heatmap_raw_states=heatmap_raw_states,
            )
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
