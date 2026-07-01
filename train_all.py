#!/usr/bin/env python3

import argparse, os, torch
import numpy as np
import matplotlib
matplotlib.use("TkAgg")  # needs Tk installed
import matplotlib.pyplot as plt
from matplotlib import cm
import algos.algos_guide as algos
import gym, d4rl
from tqdm.auto import tqdm
from logger import logger, setup_logger
from replay_buffer.numpy_buffer import ReplayBuffer
from envs.fixed_reset_wrapper import build_wrapped_env, get_env_start_xy, get_env_goal
from datasets.utils import get_dataset

color_list = cm.rainbow(np.linspace(0, 1, 12))

def eval_value(policy, buffer_expert, buffer_random, sample_size=1500, mode='q'):
    state_random, _, _, _, _, _ = buffer_random.sample(sample_size)
    state_expert, _, _, _, _, _ = buffer_expert.sample(sample_size)

    if mode == 'q':
        state_random = renorm_random_expert(state_random, buffer_expert, buffer_random)
        value_random = policy.get_pi_q(state_random, policy.actor, policy.critic, 
                                       policy.actor_vae, use_noise=False)
        value_expert = policy.get_pi_q(state_expert, policy.actor, policy.critic, 
                                       policy.actor_vae, use_noise=False)
    if mode == 'v':
        state_expert = renorm_expert_random(state_expert, buffer_expert, buffer_random)
        value_random = policy.vnet(state_random)
        value_expert = policy.vnet(state_expert)

    value_random_np = value_random.detach().cpu().numpy().squeeze()
    value_expert_np = value_expert.detach().cpu().numpy().squeeze()

    if isinstance(state_random, torch.Tensor):
        state_random = state_random.detach().cpu().numpy()
        state_expert = state_expert.detach().cpu().numpy()
    else:
        state_random = np.asarray(state_random)
        state_expert = np.asarray(state_expert)

    plt.clf()
    sc = plt.scatter(state_random[:, 0], state_random[:, 1], c=value_random_np, cmap="viridis", alpha=0.3,
                     vmin=0, vmax=50
    )
    plt.colorbar(sc, label=mode + "_value")
    plt.savefig("./eval_random_" + mode + ".png", dpi=300, bbox_inches="tight")


    plt.clf()
    sc = plt.scatter(state_expert[:, 0], state_expert[:, 1], c=value_expert_np, cmap="viridis", alpha=0.3,
                     vmin=0, vmax=50
    )
    plt.colorbar(sc, label=mode + "_value")
    plt.savefig("./eval_expert_" + mode + ".png", dpi=300, bbox_inches="tight")


def eval_policy(policy, env, replay_buffer, eval_episodes=10, plot=False):
    avg_reward = 0.
    plt.clf()
    start_states = []
    color_list = cm.rainbow(np.linspace(0, 1, eval_episodes+2))
    # env = build_wrapped_env(args.env_name)
    # env = gym.make(args.env_name)

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
            action = np.clip(action, env.action_space.low, env.action_space.high)

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

def renorm_random_expert(state, replay_buffer_expert, replay_buffer_random):
    state_unorm_random = replay_buffer_random.unnormalize_state(state)
    state_norm_expert = replay_buffer_expert.normalize_state(state_unorm_random)
    return state_norm_expert

def renorm_expert_random(state, replay_buffer_expert, replay_buffer_random):
    state_unorm_policy = replay_buffer_expert.unnormalize_state(state)
    state_norm_random = replay_buffer_random.normalize_state(state_unorm_policy)
    return state_norm_random

def renorma_random_expert(action, replay_buffer_expert, replay_buffer_random):
    action_unorm_random = replay_buffer_random.unnormalize_action(action)
    action_norm_expert = replay_buffer_expert.normalize_action(action_unorm_random)
    return action_norm_expert

def renorma_expert_random(action, replay_buffer_expert, replay_buffer_random):
    action_unorm_policy = replay_buffer_expert.unnormalize_action(action)
    action_norm_random = replay_buffer_random.normalize_action(action_unorm_policy)
    return action_norm_random

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Additional parameters
    parser.add_argument("--mode", default='pi', type=str)              # train policy or value
    parser.add_argument("--ExpID", default=7, type=int)              # Experiment ID
    parser.add_argument('--log_dir', default='./results/', type=str)    # Logging directory
    parser.add_argument("--load_model", default=100, type=int)          # Load model and optimizer parameters
    parser.add_argument("--save_model", default=True, type=bool)        # Save model and optimizer parameters
    parser.add_argument("--save_freq", default=1, type=int)           # How often it saves the model (epoch)
    parser.add_argument("--env_name", default="maze2d-large-v1")     # OpenAI gym environment name
    parser.add_argument("--seed", default=456, type=int)                  # Sets Gym, PyTorch and Numpy seeds
    parser.add_argument("--eval_freq", default=10000, type=int)           # How often (time steps) we evaluate
    parser.add_argument("--max_timesteps", default=1e6, type=int)      # Max time steps to run environment for
    parser.add_argument('--batch_size', default=512, type=int)
    parser.add_argument('--vae_lr', default=2e-4, type=float)	        # action policy (VAE) learning rate
    parser.add_argument('--actor_lr', default=2e-4, type=float)	        # latent policy learning rate
    parser.add_argument('--critic_lr', default=2e-4, type=float)	    # critic learning rate
    parser.add_argument('--tau', default=0.005, type=float)	            # delayed learning rate
    parser.add_argument('--discount', default=0.99, type=float)	        # discount factor

    parser.add_argument('--expectile', default=0.85, type=float)	        # expectile to compute weight for samples
    parser.add_argument('--kl_beta', default=1.0, type=float)	            # weight for kl loss to train CVAE
    parser.add_argument('--max_latent_action', default=0.675, type=float)	# maximum value for the latent policy
    parser.add_argument('--doubleq_min', default=1.0, type=float)         # weight for the minimum Q value
    parser.add_argument('--no_noise', action='store_true')              # adding noise to the latent policy or not

    parser.add_argument('--plot', action='store_true')
    parser.add_argument('--device', default='cuda', type=str)

    args = parser.parse_args()

    # Setup Logging
    file_name = f"Exp{args.ExpID:04d}/{args.env_name}"
    folder_name = os.path.join(args.log_dir, file_name)
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

    if os.path.exists(os.path.join(folder_name, 'progress.csv')):
        print('exp file already exist')
        # raise AssertionError

    variant = vars(args)
    variant.update(node=os.uname()[1])
    setup_logger(os.path.basename(folder_name), variant=variant, log_dir=folder_name)

    # Setup Environment
    # env = gym.make(args.env_name)   
    # env_train = gym.make(args.env_name)   
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
    # dataset = d4rl.qlearning_dataset(env)  # Load d4rl dataset
    if 'antmaze' in args.env_name:
        dataset_path_expert = f"./datasets/antmaze/expert/antmaze-expert-success500.hdf5"
        dataset_path_random = f"./datasets/antmaze/random/Ant_maze_hardest-maze_noisy_multistart_True_multigoal_False_sparse_fixed.hdf5"
        dataset_path_mix = f"./datasets/antmaze/mix/antmaze_mixed.hdf5"
    if 'maze2d' in args.env_name:
        dataset_path_expert = f"./datasets/maze2d/expert/maze2d-large-500-keep10.hdf5"
        dataset_path_random = f"./datasets/maze2d/random/maze2d-large-sparse-v1.hdf5"
        dataset_path_mix = f"./datasets/maze2d/mix/maze2d-mix.hdf5"

    dataset_expert = get_dataset(env, dataset_path_expert)
    dataset_random = get_dataset(env, dataset_path_random)
    if 'antmaze' in args.env_name:
        # dataset_expert['rewards'] = (dataset_expert['rewards']*100) #(dataset['rewards']*300)
        # dataset_random['rewards'] = (dataset_random['rewards']*100) #(dataset['rewards']*300)
        dataset_random['terminals'] = dataset_random['terminals'] * 0

    min_v = 0
    max_v = 110

    replay_buffer_expert = ReplayBuffer(args.env_name, state_dim, action_dim, args.device)
    replay_buffer_random = ReplayBuffer(args.env_name, state_dim, action_dim, args.device)
    replay_buffer_expert.load(dataset_expert)
    # replay_buffer_random.load(dataset_random)
    replay_buffer_random.load(dataset_random, normalize=False)
    replay_buffer_random.load(dataset_expert, stats=replay_buffer_random.stats)

    latent_dim = max(4, int(action_dim*2.0))
    print("---- latent_dim:", latent_dim, 'dataset size', replay_buffer_expert.size, replay_buffer_random.size)
    policy = algos.Latent(state_dim, action_dim, latent_dim, min_v, max_v,
                        device=args.device, discount=args.discount, tau=args.tau, 
                        vae_lr=args.vae_lr, actor_lr=args.actor_lr, critic_lr=args.critic_lr, 
                        max_latent_action=args.max_latent_action, expectile=args.expectile, kl_beta=args.kl_beta, 
                        doubleq_min=args.doubleq_min)

    load_id = args.load_model
    model_file_name = f"Exp{load_id:04d}/{args.env_name}"
    model_folder_name = os.path.join(args.log_dir, model_file_name)
    policy.load_reference('model', model_folder_name)
    # policy.load('model', model_folder_name)
    print(f"Using value of pre-trained policy model {model_folder_name}")
    # eval_value(policy, replay_buffer_expert, replay_buffer_random, mode='v')
    # info = eval_policy(policy, env, replay_buffer_random, plot=args.plot)

    num_itr = int(args.max_timesteps/args.eval_freq)
    with tqdm(range(num_itr), desc='Epoch', leave=False) as tglobal:
        for epoch_idx in tglobal:
            v_list, crit_list, rec_list, kl_list, w_list, act_list = [], [], [], [], [], []
            qvalue_list, vrandom_list, vexpert_list = [], [], []

            with tqdm(range(args.eval_freq), desc='Batch', leave=False) as tepoch:
                for idx in tepoch:
                    batch = replay_buffer_random.sample(args.batch_size)
                    # state, action, _, _, _ = batch
                    # state = renorm_expert_random(state, replay_buffer_expert, replay_buffer_random)
                    # action = renorma_expert_random(action, replay_buffer_expert, replay_buffer_random)

                    sample_w, act_loss, crit_loss, recons_loss, kl_loss, adv = policy.train_step(batch, idx)

                    rec_list.append(recons_loss)
                    kl_list.append(kl_loss)
                    w_list.append(sample_w)
                    crit_list.append(crit_loss)
                    act_list.append(act_loss)

                    tepoch.set_postfix(rec=np.mean(rec_list), kl=np.mean(kl_list), weight=np.mean(w_list),
                                       crit=np.mean(crit_list), act=np.mean(act_list))

                tglobal.set_postfix(rec=np.mean(rec_list), kl=np.mean(kl_list), weight=np.mean(w_list),
                                       crit=np.mean(crit_list), act=np.mean(act_list))
                # Eval
                logger.record_tabular('Training Epochs', int(epoch_idx))
                print('done')
                policy.eval()
                info = eval_policy(policy, env, replay_buffer_random, plot=args.plot)
                eval_value(policy, replay_buffer_expert, replay_buffer_random, mode='q')
                # eval_value(policy, replay_buffer_expert, replay_buffer_random, mode='v')
                policy.train()

                for k, v in info.items():
                    logger.record_tabular(k, v)
                logger.record_tabular('L_Rec', np.mean(rec_list))
                logger.record_tabular('L_KL', np.mean(kl_list))
                logger.record_tabular('L_Act', np.mean(act_list))
                logger.record_tabular('L_Crit', np.mean(crit_list))
                logger.record_tabular('Weight', np.mean(w_list))
                logger.dump_tabular()

            # Save Model
            if epoch_idx % args.save_freq == 0 and args.save_model and epoch_idx != 0:
                policy.save('model', folder_name)

    policy.save('model', folder_name)

#  python train_all.py --ExpID 100 --env_name antmaze-large-diverse-v2 --plot --mode v --device cuda:1 --load_model 
#  python train_all.py --ExpID 100 --env_name maze2d-large-v1 --plot --mode v --device cuda:1 --load_model 