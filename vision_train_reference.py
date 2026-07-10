#!/usr/bin/env python3

import argparse, os, torch
import numpy as np
import torch.nn.functional as F
import algos.vision_algos_reference as algos
from tqdm.auto import tqdm
from logger import logger, setup_logger
from replay_buffer.vision_numpy_buffer import WaterPipeDataset
from datasets.utils import get_real_dataset
from plot_3d_value import plot_xyz_value, plot_xyz_value_mixed
from utils.resume import load_training_state, save_training_state, set_rng_state

def safe_mean(values):
    return float(np.mean(values)) if values else float("nan")

def safe_column_mean(values):
    return np.mean(np.stack(values, axis=0), axis=0) if values else np.asarray([])

def eval_reconstruction_loss(policy, water_pipe, dataloader, num_batches=4):
    recon_values, recon_sum_values = [], []
    raw_recon_values, raw_recon_sum_values, raw_recon_dim_values = [], [], []
    kl_values = []
    was_training = policy.training
    policy.eval()

    with torch.no_grad():
        for batch_idx, cpu_batch in enumerate(dataloader):
            if batch_idx >= num_batches:
                break
            state, action, _, _, _, _ = water_pipe.batch_to_device(cpu_batch)
            state = policy.encode_state(state)
            recons_action, mu, log_var = policy.actor_vae(state, action)

            recon_per_dim = F.mse_loss(recons_action, action, reduction="none")
            recon_values.append(recon_per_dim.mean().item())
            recon_sum_values.append(torch.sum(recon_per_dim, dim=1).mean().item())

            raw_action = water_pipe.unnormalize_action(action)
            raw_recons_action = water_pipe.unnormalize_action(recons_action)
            raw_recon_per_dim = F.mse_loss(raw_recons_action, raw_action, reduction="none")
            raw_recon_values.append(raw_recon_per_dim.mean().item())
            raw_recon_sum_values.append(torch.sum(raw_recon_per_dim, dim=1).mean().item())
            raw_recon_dim_values.append(raw_recon_per_dim.mean(dim=0).cpu().numpy())

            free_bits = 0.5
            kl_per_dim = -0.5 * (1 + log_var - mu.pow(2) - log_var.exp())
            kl_freebits = torch.maximum(
                kl_per_dim,
                torch.tensor(free_bits, device=kl_per_dim.device),
            )
            kl_values.append(kl_freebits.sum(dim=1).mean().item())

    if was_training:
        policy.train()

    raw_recon_dim_mse = safe_column_mean(raw_recon_dim_values)
    raw_recon_dim_rmse = np.sqrt(raw_recon_dim_mse)
    return {
        "recon": safe_mean(recon_values),
        "recon_sum": safe_mean(recon_sum_values),
        "raw_recon": safe_mean(raw_recon_values),
        "raw_recon_sum": safe_mean(raw_recon_sum_values),
        "raw_recon_dim_mse": raw_recon_dim_mse.tolist(),
        "raw_recon_dim_rmse": raw_recon_dim_rmse.tolist(),
        "kl": safe_mean(kl_values),
    }

def device_batches(water_pipe, dataloader):
    for idx, cpu_batch in enumerate(dataloader):
        yield idx, water_pipe.batch_to_device(cpu_batch)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Additional parameters
    parser.add_argument("--mode", default='pi', type=str)              # train policy or value
    parser.add_argument("--ExpID", default=7, type=int)              # Experiment ID
    parser.add_argument('--log_dir', default='./results/', type=str)    # Logging directory
    parser.add_argument("--load_model", default=-1, type=int)          # Load model and optimizer parameters
    parser.add_argument("--resume", action="store_true")               # Resume this ExpID from its latest checkpoint
    parser.add_argument("--save_model", default=True, type=bool)        # Save model and optimizer parameters
    parser.add_argument("--save_freq", default=1, type=int)           # How often it saves the model (epoch)
    parser.add_argument("--env_name", default="franka")     # Logging name
    parser.add_argument("--seed", default=789, type=int)                  # Sets Gym, PyTorch and Numpy seeds
    parser.add_argument("--eval_freq", default=100, type=int)           # How often (time steps) we evaluate
    parser.add_argument("--max_timesteps", default=3e6, type=int)
    parser.add_argument('--batch_size', default=256, type=int)
    parser.add_argument('--loader_workers', default=4, type=int)
    parser.add_argument('--loader_prefetch', default=4, type=int)
    parser.add_argument('--vae_lr', default=3e-4, type=float)	        # action policy (VAE) learning rate
    parser.add_argument('--actor_lr', default=3e-4, type=float)	        # latent policy learning rate
    parser.add_argument('--critic_lr', default=3e-4, type=float)	    # critic learning rate
    parser.add_argument('--tau', default=0.005, type=float)	            # delayed learning rate
    parser.add_argument('--discount', default=0.99, type=float)	        # discount factor

    parser.add_argument('--expectile', default=0.85, type=float)	        # expectile to compute weight for samples
    parser.add_argument('--kl_beta', default=1.0, type=float)	            # weight for kl loss to train CVAE
    parser.add_argument('--max_latent_action', default=0.675, type=float)	# maximum value for the latent policy
    parser.add_argument('--doubleq_min', default=1.0, type=float)         # weight for the minimum Q value
    parser.add_argument('--no_noise', action='store_true')              # adding noise to the latent policy or not

    parser.add_argument('--plot', action='store_true')
    parser.add_argument('--device', default='cuda:0', type=str)

    args = parser.parse_args()
    if args.resume and args.load_model != -1:
        raise ValueError("--resume resumes the current ExpID; do not combine it with --load_model.")
    
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
    # max_v = 10000
    max_v = 100

    expert_pipe = WaterPipeDataset(args.env_name, state_dim, action_dim, args.device)
    random_pipe = WaterPipeDataset(args.env_name, state_dim, action_dim, args.device)
    expert_pipe.load(dataset_expert, terminal_reward_count=2)
    random_pipe.load(dataset_random, terminal_reward_count=0)
    expert_stats = WaterPipeDataset.compute_joint_stats([expert_pipe])
    mixed_stats = WaterPipeDataset.compute_joint_stats([expert_pipe, random_pipe])
    if args.mode == 'v':
        expert_pipe.apply_stats(mixed_stats)
        random_pipe.apply_stats(mixed_stats)
        expert_pipe.save_stats(os.path.join(folder_name, "normalization_stats.json"))
        print("normalize expert and random with mixed stats")
    else:
        expert_pipe.apply_stats(expert_stats)
        random_pipe.apply_stats(mixed_stats)
        print("normalize expert with expert stats")
    expert_loader = expert_pipe.make_dataloader(
        args.batch_size,
        epoch_size=args.eval_freq,
        num_workers=args.loader_workers,
        prefetch_factor=args.loader_prefetch,
    )
    random_loader = random_pipe.make_dataloader(
        args.batch_size,
        epoch_size=args.eval_freq,
        num_workers=args.loader_workers,
        prefetch_factor=args.loader_prefetch,
    )

    latent_dim = max(4, int(action_dim*2.0))
    print("---- latent_dim:", latent_dim, 'dataset size', expert_pipe.size, random_pipe.size)

    policy = algos.Latent(state_dim, action_dim, latent_dim, min_v, max_v,
                        device=args.device, discount=args.discount, tau=args.tau, 
                        vae_lr=args.vae_lr, actor_lr=args.actor_lr, critic_lr=args.critic_lr, 
                        max_latent_action=args.max_latent_action, expectile=args.expectile, kl_beta=args.kl_beta, 
                        doubleq_min=args.doubleq_min)

    start_epoch = 0
    if args.resume:
        checkpoint_state = load_training_state(folder_name)
        policy.load('model', folder_name)
        set_rng_state(checkpoint_state.get("rng_state"))
        start_epoch = int(checkpoint_state["epoch_idx"]) + 1
        print(f"Resuming training from {folder_name}, start_epoch={start_epoch}")
    elif args.mode == 'v' or args.load_model != -1:
        load_id = args.load_model
        model_file_name = f"Exp{load_id:04d}/{args.env_name}"
        model_folder_name = os.path.join(args.log_dir, model_file_name)
        policy.load('model', model_folder_name)
        print(f"Evaluating value of pre-trained policy model {model_folder_name}")

    num_itr = int(args.max_timesteps/args.eval_freq)

    with tqdm(range(start_epoch, num_itr), desc='Epoch', leave=False, initial=start_epoch, total=num_itr) as tglobal:
        for epoch_idx in tglobal:
            v_list, crit_list, rec_list, kl_list = [], [], [], []
            qvalue_list, vrandom_list, vexpert_list = [], [], []

            if args.mode == 'pi':
                batches = device_batches(expert_pipe, expert_loader)
                with tqdm(batches, total=args.eval_freq, desc='Batch', leave=False) as tepoch:
                    for idx, batch in tepoch:
                        state, action, _, _, _, _ = batch
                        recons_loss, kl_loss = policy.train_policy(state, action)

                        rec_list.append(recons_loss)
                        kl_list.append(kl_loss)

                        tepoch.set_postfix(rec=np.mean(rec_list), kl=np.mean(kl_list))

                tglobal.set_postfix(rec=np.mean(rec_list), kl=np.mean(kl_list))

                # Eval
                logger.record_tabular('Training Epochs', int(epoch_idx))
                print('done')
                recon_metrics = eval_reconstruction_loss(
                    policy, expert_pipe, expert_loader
                )
                tglobal.set_postfix(
                    rec=np.mean(rec_list),
                    kl=np.mean(kl_list),
                    val_rec=recon_metrics["recon"],
                    raw_rec=recon_metrics["raw_recon"],
                )
                logger.record_tabular('L_Rec', np.mean(rec_list))
                logger.record_tabular('L_KL', np.mean(kl_list))
                logger.record_tabular('Val_Recon', recon_metrics["recon"])
                logger.record_tabular('Val_Recon_Sum', recon_metrics["recon_sum"])
                logger.record_tabular('Val_Raw_Recon', recon_metrics["raw_recon"])
                logger.record_tabular('Val_Raw_Recon_Sum', recon_metrics["raw_recon_sum"])
                logger.record_tabular('Val_KL', recon_metrics["kl"])
                for dim, value in enumerate(recon_metrics["raw_recon_dim_mse"]):
                    logger.record_tabular(f"Val_Raw_Recon_Dim{dim}_MSE", value)
                for dim, value in enumerate(recon_metrics["raw_recon_dim_rmse"]):
                    logger.record_tabular(f"Val_Raw_Recon_Dim{dim}_RMSE", value)
                if args.plot:
                    plot_xyz_value(
                        policy,
                        expert_pipe,
                        args.mode,
                        os.path.join(folder_name, f"xyz_{args.mode}_epoch{epoch_idx:04d}.png"),
                        value_min=0,
                        value_max=1,
                    )
                logger.dump_tabular()

            if args.mode == 'q':
                batches = device_batches(expert_pipe, expert_loader)
                with tqdm(batches, total=args.eval_freq, desc='Batch', leave=False) as tepoch:
                    for idx, batch in tepoch:
                        crit_loss, q_value = policy.train_q(batch, idx)

                        crit_list.append(crit_loss)
                        qvalue_list.append(q_value)
                        tepoch.set_postfix(q_loss=np.mean(crit_list), q_value=np.mean(qvalue_list))

                q_loss_mean = np.mean(crit_list)
                q_value_mean = np.mean(qvalue_list)
                q_scale = policy.qnet.scale.item()
                print()
                print(' q scale', q_scale)
                tglobal.set_postfix(q_loss=q_loss_mean, q_value=q_value_mean)
                logger.record_tabular('Training Epochs', int(epoch_idx))
                logger.record_tabular('Mode', args.mode)
                logger.record_tabular('L_Crit', q_loss_mean)
                logger.record_tabular('Q_Value', q_value_mean)
                logger.record_tabular('Q_Scale', q_scale)
                logger.dump_tabular()
                if args.plot:
                    plot_xyz_value(
                        policy,
                        expert_pipe,
                        args.mode,
                        os.path.join(folder_name, f"xyz_{args.mode}_epoch{epoch_idx:04d}.png"),
                        value_min=0,
                        value_max=80,
                    )

            if args.mode == 'v':
                random_batches = device_batches(random_pipe, random_loader)
                expert_batches = device_batches(expert_pipe, expert_loader)
                with tqdm(zip(random_batches, expert_batches), total=args.eval_freq, desc='Batch', leave=False) as tepoch:
                    for (idx, batch_random), (_, batch_expert) in tepoch:
                        state_random_inm, _, _, _, _, _ = batch_random
                        state_expert_inm, a_expert_inm, _, _, _, _ = batch_expert

                        raw_expert_state = expert_pipe.unnormalize_state(state_expert_inm)
                        raw_expert_action = expert_pipe.unnormalize_action(a_expert_inm)
                        state_expert_ine = expert_pipe.normalize_state_with_stats(raw_expert_state, expert_stats)
                        a_expert_ine = expert_pipe.normalize_action_with_stats(raw_expert_action, expert_stats)

                        v_loss, v_random, v_expert = policy.train_value(
                            state_random_inm,
                            state_expert_inm,
                            state_expert_ine,
                            a_expert_ine,
                        )

                        v_list.append(v_loss)
                        vrandom_list.append(v_random)
                        vexpert_list.append(v_expert)
                        tepoch.set_postfix(v_loss=np.mean(v_list), 
                                           v_random=np.mean(vrandom_list),
                                           v_expert=np.mean(vexpert_list))

                v_loss_mean = np.mean(v_list)
                v_random_mean = np.mean(vrandom_list)
                v_expert_mean = np.mean(vexpert_list)
                v_scale = policy.vnet.scale.item()
                print(' value scale', v_scale)
                tglobal.set_postfix(v_loss=v_loss_mean,
                                     v_random=v_random_mean,
                                     v_expert=v_expert_mean)
                logger.record_tabular('Training Epochs', int(epoch_idx))
                logger.record_tabular('Mode', args.mode)
                logger.record_tabular('L_V', v_loss_mean)
                logger.record_tabular('V_Random', v_random_mean)
                logger.record_tabular('V_Expert', v_expert_mean)
                logger.record_tabular('V_Scale', v_scale)
                logger.dump_tabular()
                if args.plot:
                    plot_xyz_value_mixed(
                        policy,
                        expert_pipe,
                        random_pipe,
                        args.mode,
                        os.path.join(folder_name, f"xyz_{args.mode}_epoch{epoch_idx:04d}.png"),
                        value_min=0,
                        value_max=1,
                    )
                
            # Save Model
            if epoch_idx % args.save_freq == 0 and args.save_model and epoch_idx != 0:
                policy.save('model', folder_name)
                save_training_state(folder_name, epoch_idx, args)

    policy.save('model', folder_name)
    save_training_state(folder_name, num_itr - 1, args)

# python vision_train_reference.py --env_name franka --ExpID 2 --plot --mode q --device cuda:0
