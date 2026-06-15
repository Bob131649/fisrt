#!/usr/bin/env python3

import argparse
import csv
import json
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

import algos.algos_v2_vision as algos
from dataset.franka_dataset import FrankaImageDataset
from logger import logger, setup_logger


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


def safe_mean(values):
    return float(np.mean(values)) if values else float("nan")


def str2bool(value):
    if isinstance(value, bool):
        return value
    value = value.lower()
    if value in ("yes", "true", "t", "1", "y"):
        return True
    if value in ("no", "false", "f", "0", "n"):
        return False
    raise argparse.ArgumentTypeError("Boolean value expected.")


def get_last_logged_epoch(log_dir):
    progress_path = os.path.join(log_dir, "progress.csv")
    if not os.path.isfile(progress_path):
        return -1

    last_epoch = -1
    with open(progress_path, "r", newline="") as progress_file:
        reader = csv.DictReader(progress_file)
        for row in reader:
            epoch = row.get("Training Epochs")
            if epoch not in (None, ""):
                last_epoch = int(float(epoch))

    return last_epoch


def get_best_logged_metric(log_dir, metric_name="L_Crit"):
    meta_path = os.path.join(log_dir, "model_best_meta.json")
    if os.path.isfile(meta_path):
        with open(meta_path, "r") as meta_file:
            meta = json.load(meta_file)
        metric = meta.get("metric")
        if metric is not None:
            return float(metric)

    progress_path = os.path.join(log_dir, "progress.csv")
    if not os.path.isfile(progress_path):
        return float("inf")

    best_metric = float("inf")
    with open(progress_path, "r", newline="") as progress_file:
        reader = csv.DictReader(progress_file)
        for row in reader:
            metric = row.get(metric_name)
            if metric not in (None, ""):
                metric = float(metric)
                if np.isfinite(metric):
                    best_metric = min(best_metric, metric)
    return best_metric


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ExpID", default=1, type=int)
    parser.add_argument("--exp_name", default="franka_vision", type=str)
    parser.add_argument("--log_dir", default="./results/", type=str)
    parser.add_argument("--dataset_path", required=True, type=str)
    parser.add_argument("--load_model", default=0, type=int)
    parser.add_argument("--resume", nargs="?", const=True, default=False, type=str2bool)
    parser.add_argument("--model_dir", default="", type=str)
    parser.add_argument("--load_best", nargs="?", const=True, default=False, type=str2bool)
    parser.add_argument("--save_model", default=True, type=bool)
    parser.add_argument("--save_freq", default=1, type=int)
    parser.add_argument("--seed", default=789, type=int)
    parser.add_argument("--max_timesteps", default=1e5, type=float)
    parser.add_argument("--steps_per_epoch", default=1000, type=int)
    parser.add_argument("--batch_size", default=256, type=int)
    parser.add_argument("--num_workers", default=0, type=int)
    parser.add_argument("--pin_memory", action="store_true")
    parser.add_argument("--image_size", nargs=2, default=(224, 224), type=int)
    parser.add_argument("--action_mode", default="absolute_eef", type=str)
    parser.add_argument("--reward_scale", default=100.0, type=float)
    parser.add_argument("--success_reward", default=1.0, type=float)
    parser.add_argument("--min_v", default=None, type=float)
    parser.add_argument("--max_v", default=None, type=float)
    parser.add_argument("--normalize_proprio", default=True, type=bool)
    parser.add_argument("--normalize_action", default=True, type=bool)

    parser.add_argument("--vae_lr", default=2e-4, type=float)
    parser.add_argument("--actor_lr", default=2e-4, type=float)
    parser.add_argument("--critic_lr", default=2e-4, type=float)
    parser.add_argument("--obs_encoder_lr", default=None, type=float)
    parser.add_argument("--tau", default=0.005, type=float)
    parser.add_argument("--discount", default=0.99, type=float)
    parser.add_argument("--expectile", default=0.9, type=float)
    parser.add_argument("--kl_beta", default=1.0, type=float)
    parser.add_argument("--max_latent_action", default=0.675, type=float)
    parser.add_argument("--doubleq_min", default=1.0, type=float)

    parser.add_argument("--robomimic_feature_dim", default=256, type=int)
    parser.add_argument("--robomimic_crop_shape", nargs=2, default=None, type=int)
    parser.add_argument("--robomimic_backbone_class", default="ResNet18Conv", type=str)
    parser.add_argument("--robomimic_pool_class", default="SpatialSoftmax", type=str)

    parser.add_argument("--ope_ref_dir", default="", type=str)
    parser.add_argument("--ope_ref_name", default="model", type=str)
    parser.add_argument("--ope_ref_clip", default=100.0, type=float)
    parser.add_argument("--device", default="cuda", type=str)
    parser.add_argument("--terminal_log_path", default="", type=str)
    parser.add_argument("--vae",default=False, type=bool)
    args = parser.parse_args()

    if not os.path.isfile(args.dataset_path):
        raise FileNotFoundError(f"Dataset file not found: {args.dataset_path}")

    file_name = f"Exp{args.ExpID:04d}/{args.exp_name}"
    folder_name = os.path.join(args.log_dir, file_name)
    start_epoch = 0
    best_critic_loss = float("inf")
    if args.resume:
        if not args.model_dir:
            raise ValueError("--model_dir must be set when --resume is true")
        folder_name = os.path.expanduser(args.model_dir)
        if not os.path.isdir(folder_name):
            raise FileNotFoundError(f"Resume model_dir not found: {folder_name}")
        start_epoch = get_last_logged_epoch(folder_name) + 1
        best_critic_loss = get_best_logged_metric(folder_name)
        print(f"resume enabled, loading checkpoint from: {folder_name}")
        print(f"resume training from epoch: {start_epoch}")
        print(f"current best critic loss: {best_critic_loss}")
    os.makedirs(folder_name, exist_ok=True)

    if os.path.exists(os.path.join(folder_name, "progress.csv")):
        print("exp file already exists")

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

    if args.vae:
        args.expectile = 0.5
        print("Training with VAE,and expectile is ",args.expectile)
        
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    image_size = tuple(args.image_size)
    dataset = FrankaImageDataset(
        args.dataset_path,
        image_size=image_size,
        reward_scale=args.reward_scale,
        success_reward=args.success_reward,
        action_mode=args.action_mode,
        normalize_proprio=args.normalize_proprio,
        normalize_action=args.normalize_action,
    )
    normalization_stats = {
        "state_mean": dataset.state_mean.astype(float).tolist(),
        "state_std": dataset.state_std.astype(float).tolist(),
        "action_mean": dataset.action_mean.astype(float).tolist(),
        "action_std": dataset.action_std.astype(float).tolist(),
        "state_dim": int(dataset.state_dim),
        "action_dim": int(dataset.action_dim),
        "action_mode": args.action_mode,
        "normalize_proprio": bool(args.normalize_proprio),
        "normalize_action": bool(args.normalize_action),
        "image_size": list(dataset.image_size),
        "dataset_path": os.path.abspath(args.dataset_path),
    }
    normalization_stats_path = os.path.join(folder_name, "normalization_stats.json")
    with open(normalization_stats_path, "w") as stats_file:
        json.dump(normalization_stats, stats_file, indent=2)
    print(f"saved normalization stats to: {normalization_stats_path}")

    state_dim = dataset.state_dim
    action_dim = dataset.action_dim
    image_shape = (3, image_size[0], image_size[1])
    latent_dim = int(action_dim * 2)
    min_v = 0.0 if args.min_v is None else args.min_v
    if args.max_v is None:
        max_v = float(max(dataset.rewards.max(), args.reward_scale * args.success_reward, 1.0))
    else:
        max_v = args.max_v

    print(
        "vision train dims:",
        f"image_shape={image_shape}",
        f"state_dim={state_dim}",
        f"action_dim={action_dim}",
        f"latent_dim={latent_dim}",
        f"value_range=[{min_v:.3f}, {max_v:.3f}]",
    )

    dataloader = DataLoader(
        dataset,
        sampler=torch.utils.data.RandomSampler(
            dataset,
            num_samples=args.batch_size * args.steps_per_epoch,
            replacement=True,
        ),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        drop_last=True,
    )

    policy = algos.Latent(
        state_dim,
        action_dim,
        latent_dim,
        min_v,
        max_v,
        device=args.device,
        discount=args.discount,
        tau=args.tau,
        vae_lr=args.vae_lr,
        actor_lr=args.actor_lr,
        critic_lr=args.critic_lr,
        obs_encoder_lr=args.obs_encoder_lr,
        max_latent_action=args.max_latent_action,
        expectile=args.expectile,
        kl_beta=args.kl_beta,
        doubleq_min=args.doubleq_min,
        image_shape=image_shape,
        robomimic_feature_dim=args.robomimic_feature_dim,
        robomimic_crop_shape=tuple(args.robomimic_crop_shape)
        if args.robomimic_crop_shape is not None
        else None,
        robomimic_backbone_class=args.robomimic_backbone_class,
        robomimic_pool_class=args.robomimic_pool_class,
        ope_ref_dir=args.ope_ref_dir,
        ope_ref_name=args.ope_ref_name,
        ope_ref_clip=args.ope_ref_clip,
        vae=args.vae,
    )

    if args.resume:
        policy.load("model", folder_name, load_best=args.load_best)
    elif args.load_model:
        policy.load("model", folder_name, load_best=args.load_best)

    num_itr = int(args.max_timesteps / args.steps_per_epoch)
    with tqdm(range(start_epoch, num_itr), desc="Epoch", leave=False) as tglobal:
        for epoch_idx in tglobal:
            act_list, crit_list, rec_list, kl_list, w_list = [], [], [], [], []
            np.random.seed()

            with tqdm(dataloader, desc="Batch", leave=False) as tepoch:
                for iter_id, batch in enumerate(tepoch):
                    sample_w, act_loss, crit_loss, recons_loss, kl_loss = policy.train_step(
                        batch, iter_id
                    )

                    crit_list.append(crit_loss)
                    if act_loss is not None:
                        act_list.append(act_loss)
                    if recons_loss is not None:
                        rec_list.append(recons_loss)
                    if kl_loss is not None:
                        kl_list.append(kl_loss)
                    w_list.append(np.mean(sample_w))

                    tepoch.set_postfix(
                        w=safe_mean(w_list),
                        act=safe_mean(act_list),
                        crit=safe_mean(crit_list),
                        rec=safe_mean(rec_list),
                        kl=safe_mean(kl_list),
                    )

            mean_crit = safe_mean(crit_list)
            is_best = np.isfinite(mean_crit) and mean_crit < best_critic_loss
            if is_best:
                best_critic_loss = mean_crit
                if args.save_model:
                    policy.save(
                        "model",
                        folder_name,
                        is_best=True,
                        best_metric=best_critic_loss,
                        best_epoch=int(epoch_idx),
                    )
                    print(
                        f"save best model success: epoch={epoch_idx}, "
                        f"L_Crit={best_critic_loss:.6f}"
                    )

            if epoch_idx % args.save_freq == 0 and args.save_model and epoch_idx != 0:
                policy.save("model", folder_name)

            logger.record_tabular("Training Epochs", int(epoch_idx))
            logger.record_tabular("L_Act", safe_mean(act_list))
            logger.record_tabular("L_Crit", mean_crit)
            logger.record_tabular("L_Recon", safe_mean(rec_list))
            logger.record_tabular("L_KL", safe_mean(kl_list))
            logger.record_tabular("Weight", safe_mean(w_list))
            logger.record_tabular("kl", policy.kl_beta)
            logger.dump_tabular()

            tglobal.set_postfix(
                w=safe_mean(w_list),
                act=safe_mean(act_list),
                crit=safe_mean(crit_list),
                rec=safe_mean(rec_list),
                kl=safe_mean(kl_list),
            )

    policy.save("model", folder_name)

    if terminal_log_file is not None:
        terminal_log_file.close()
