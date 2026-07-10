#!/usr/bin/env python3

import argparse, os, torch
import numpy as np
import algos.vision_algos_guide as algos
from tqdm.auto import tqdm
from logger import logger, setup_logger
from replay_buffer.vision_numpy_buffer import WaterPipeDataset
from datasets.utils import get_real_dataset
from plot_vnet_xyz import sample_xyz_and_value, save_plot
from utils.vision_all_eval import eval_ref_metrics, run_visual_eval

def device_batches(water_pipe, dataloader):
    for idx, cpu_batch in enumerate(dataloader):
        yield idx, water_pipe.batch_to_device(cpu_batch)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Additional parameters
    parser.add_argument("--mode", default='pi', type=str)              # train policy or value
    parser.add_argument("--ExpID", default=100, type=int)              # Experiment ID
    parser.add_argument('--log_dir', default='./results/', type=str)    # Logging directory
    parser.add_argument("--load_model", default=100, type=int)          # Load model and optimizer parameters
    parser.add_argument("--save_model", default=True, type=bool)        # Save model and optimizer parameters
    parser.add_argument("--save_freq", default=1, type=int)           # How often it saves the model (epoch)
    parser.add_argument("--env_name", default="franka")     # Logging name
    parser.add_argument("--seed", default=456, type=int)                  # Sets Gym, PyTorch and Numpy seeds
    parser.add_argument("--eval_freq", default=100, type=int)           # Training batches per epoch
    parser.add_argument("--max_timesteps", default=3e6, type=int)
    parser.add_argument('--batch_size', default=256, type=int)
    parser.add_argument('--loader_workers', default=4, type=int)
    parser.add_argument('--loader_prefetch', default=4, type=int)
    parser.add_argument('--plot', action='store_true')
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

    parser.add_argument('--device', default='cuda:0', type=str)

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

    state_dim = 8
    action_dim = 8

    # Set seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    dataset_path_expert_list = [
        "./datasets/real_dataset/expert/franka_pickplace_ep20_20260629_152135_add_rewarding20.hdf5",
        "./datasets/real_dataset/expert/franka_pickplace_ep30_20260626_162422_add_rewarding20.hdf5",
        "./datasets/real_dataset/expert/franka_pickplace_ep31_20260625_112544_add_rewarding20.hdf5",
        "./datasets/real_dataset/expert/franka_pickplace_ep39_20260626_143912_add_rewarding20.hdf5",
    ]
    dataset_path_random_list = [
        "./datasets/real_dataset/random/franka_random_20260701_170633_step1909.hdf5",
        "./datasets/real_dataset/random/franka_random_20260702_154715_step2056.hdf5",
        "./datasets/real_dataset/random/franka_random_20260702_155326_step1535.hdf5",
        "./datasets/real_dataset/random/franka_random_20260702_155755_step1499.hdf5",
        "./datasets/real_dataset/random/franka_random_20260703_142557_step2013.hdf5",
    ]
    dataset_expert = [get_real_dataset(path) for path in dataset_path_expert_list]
    dataset_random = [get_real_dataset(path) for path in dataset_path_random_list]

    min_v = 0
    max_v = 100

    expert_pipe = WaterPipeDataset(args.env_name, state_dim, action_dim, args.device)
    random_pipe = WaterPipeDataset(args.env_name, state_dim, action_dim, args.device)
    expert_pipe.load(dataset_expert, terminal_reward_count=2)
    random_pipe.load(dataset_random, terminal_reward_count=0)
    expert_train_pipe, expert_val_pipe = expert_pipe.split_val(0.05, seed=args.seed)
    random_train_pipe, random_val_pipe = random_pipe.split_val(0.05, seed=args.seed + 1)
    train_pipe = WaterPipeDataset.concat([random_train_pipe, expert_train_pipe])
    # train_pipe = random_train_pipe
    # Train all uses a shared normalization space for expert and random data.
    normalize_stats = WaterPipeDataset.compute_joint_stats([train_pipe])
    train_pipe.apply_stats(normalize_stats)
    expert_val_pipe.apply_stats(normalize_stats)
    random_val_pipe.apply_stats(normalize_stats)
    train_pipe.save_stats(os.path.join(folder_name, "normalization_stats.json"))
    train_loader = train_pipe.make_dataloader(
        args.batch_size,
        epoch_size=args.eval_freq,
        num_workers=args.loader_workers,
        prefetch_factor=args.loader_prefetch,
    )
    expert_val_loader = expert_val_pipe.make_dataloader(
        args.batch_size,
        epoch_size=4,
        num_workers=args.loader_workers,
        prefetch_factor=args.loader_prefetch,
    )
    random_val_loader = random_val_pipe.make_dataloader(
        args.batch_size,
        epoch_size=4,
        num_workers=args.loader_workers,
        prefetch_factor=args.loader_prefetch,
    )
                                                                                                                             
    latent_dim = max(4, int(action_dim*2.0))
    print(
        "---- latent_dim:", latent_dim,
        "train size", train_pipe.size,
        "expert val", expert_val_pipe.size,
        "random val", random_val_pipe.size,
    )
    policy = algos.Latent(state_dim, action_dim, latent_dim, min_v, max_v,
                        device=args.device, discount=args.discount, tau=args.tau, 
                        vae_lr=args.vae_lr, actor_lr=args.actor_lr, critic_lr=args.critic_lr, 
                        max_latent_action=args.max_latent_action, expectile=args.expectile, kl_beta=args.kl_beta, 
                        doubleq_min=args.doubleq_min)
    policy.state_mean_torch = train_pipe.state_mean_torch
    policy.state_std_torch = train_pipe.state_std_torch

    load_id = args.load_model
    model_file_name = f"Exp{load_id:04d}/{args.env_name}"
    model_folder_name = os.path.join(args.log_dir, model_file_name)
    policy.load_reference('model', model_folder_name)
    print(f"Using value of pre-trained policy model {model_folder_name}")

    num_itr = int(args.max_timesteps/args.eval_freq)
    with tqdm(range(num_itr), desc='Epoch', leave=False) as tglobal:
        for epoch_idx in tglobal:
            v_list, crit_list, rec_list, kl_list, w_list, act_list = [], [], [], [], [], []

            batches = device_batches(train_pipe, train_loader)
            with tqdm(batches, total=args.eval_freq, desc='Batch', leave=False) as tepoch:
                for idx, batch in tepoch:

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
                expert_val = eval_ref_metrics(policy, expert_val_pipe, expert_val_loader)
                random_val = eval_ref_metrics(policy, random_val_pipe, random_val_loader)
                tglobal.set_postfix(
                    rec=np.mean(rec_list),
                    kl=np.mean(kl_list),
                    weight=np.mean(w_list),
                    crit=np.mean(crit_list),
                    act=np.mean(act_list),
                    exp_val_rec=expert_val["rec"],
                    rnd_val_rec=random_val["rec"],
                )
                logger.record_tabular('Training Epochs', int(epoch_idx))
                print('done')
                logger.record_tabular('L_Rec', np.mean(rec_list))
                logger.record_tabular('L_KL', np.mean(kl_list))
                logger.record_tabular('L_Act', np.mean(act_list))
                logger.record_tabular('L_Crit', np.mean(crit_list))
                logger.record_tabular('Weight', np.mean(w_list))
                logger.record_tabular('Val_Expert_Rec', expert_val["rec"])
                logger.record_tabular('Val_Expert_KL', expert_val["kl"])
                logger.record_tabular('Val_Random_Rec', random_val["rec"])
                logger.record_tabular('Val_Random_KL', random_val["kl"])
                if args.plot:
                    run_visual_eval(
                        policy,
                        expert_val_pipe,
                        random_val_pipe,
                        folder_name,
                        epoch_idx,
                        512,
                    )
                logger.dump_tabular()

            # Save Model
            if epoch_idx % args.save_freq == 0 and args.save_model and epoch_idx != 0:
                policy.save('model', folder_name)

    policy.save('model', folder_name)
