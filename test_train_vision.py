#!/usr/bin/env python3

import argparse
import csv
import json
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from tqdm.auto import tqdm

import algos.algos_v2_vision as algos
from dataset.franka_dataset import FrankaImageDataset
from logger import logger, setup_logger
from vision_eval.loss_plot import (
    NoAugmentSubset,
    chain_threshold_episodes,
    evaluate_vae_validation,
    save_epoch_loss_plot,
    split_train_val_episode_indices,
)


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


def format_dim_values(values, precision=6):
    if values is None:
        return "[]"
    return "[" + ", ".join(f"{float(value):.{precision}g}" for value in values) + "]"


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


def get_best_logged_metric(log_dir, metric_name="L_Recon"):
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
    default_dataset_paths = [
        "/home/guannan/first/dataset/real_dataset/raw_dataset/franka_pickplace_ep24_20260617_171251.hdf5",
        "/home/guannan/first/dataset/real_dataset/raw_dataset/franka_pickplace_ep17_20260616_202143.hdf5",
        "/home/guannan/first/dataset/real_dataset/raw_dataset/franka_pickplace_ep19_20260617_114829.hdf5",
        "/home/guannan/first/dataset/real_dataset/raw_dataset/franka_pickplace_ep25_20260617_151011.hdf5",
        "/home/guannan/first/dataset/real_dataset/raw_dataset/franka_pickplace_ep31_early.hdf5",
    ]
    defaults = {
        "save_model": True,
        "reward_scale": 100.0,
        "success_reward": 1.0,
        "normalize_proprio": True,
        "normalize_action": True,
        "min_v": None,
        "max_v": None,
        "val_batch_size": None,
        "val_chain_threshold_transitions": True,
        "eval_loss_plot_freq": 1,
        "pin_memory": False,
    }

    parser = argparse.ArgumentParser()
    parser.add_argument("--ExpID", default=1, type=int)
    parser.add_argument("--exp_name", default="franka_vision", type=str)
    parser.add_argument("--log_dir", default="./results/", type=str)
    parser.add_argument("--dataset_path", default=default_dataset_paths, nargs="+", type=str)
    parser.add_argument("--load_model", default=0, type=int)
    parser.add_argument("--resume", nargs="?", const=True, default=False, type=str2bool)
    parser.add_argument("--model_dir", default="", type=str)
    parser.add_argument("--load_best", nargs="?", const=True, default=False, type=str2bool)
    parser.add_argument("--save_freq", default=1, type=int)
    parser.add_argument("--seed", default=789, type=int)
    parser.add_argument("--max_timesteps", default=1e4, type=float)
    parser.add_argument("--steps_per_epoch", default=500, type=int)
    parser.add_argument("--batch_size", default=256, type=int)
    parser.add_argument("--val_fraction", default=0.1, type=float)
    parser.add_argument("--num_workers", default=0, type=int)
    parser.add_argument("--image_size", nargs=2, default=(224, 224), type=int)
    parser.add_argument("--encoder_mode", default="zipper", choices=["robomimic", "zipper"])
    parser.add_argument("--zipper_backbone", default="resnet18", choices=["resnet18", "resnet50"], type=str)
    parser.add_argument("--zipper_normalize_image", nargs="?", const=True, default=True, type=str2bool)
    parser.add_argument("--device", default="cuda", type=str)
    parser.add_argument("--terminal_log_path", default="", type=str)
    parser.add_argument("--vae", action="store_true")
    parser.add_argument("--concat_mode", default="default", choices=["default", "film"],type=str)
    args = parser.parse_args()

    for dataset_path in args.dataset_path:
        if not os.path.isfile(dataset_path):
            raise FileNotFoundError(f"Dataset file not found: {dataset_path}")

    file_name = f"Exp{args.ExpID:04d}/{args.exp_name}"
    folder_name = os.path.join(args.log_dir, file_name)
    start_epoch = 0
    best_recon_loss = float("inf")
    if args.resume:
        if not args.model_dir:
            raise ValueError("--model_dir must be set when --resume is true")
        folder_name = os.path.expanduser(args.model_dir)
        if not os.path.isdir(folder_name):
            raise FileNotFoundError(f"Resume model_dir not found: {folder_name}")
        start_epoch = get_last_logged_epoch(folder_name) + 1
        best_recon_loss = get_best_logged_metric(folder_name)
        print(f"resume enabled, loading checkpoint from: {folder_name}")
        print(f"resume training from epoch: {start_epoch}")
        print(f"current best recon loss: {best_recon_loss}")
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
    variant.update(defaults)
    variant.update(node=os.uname()[1])
    setup_logger(os.path.basename(folder_name), variant=variant, log_dir=folder_name)

    policy_kwargs = {}
    if args.vae:
        policy_kwargs["expectile"] = 0.5
        print("Training with VAE,and expectile is ", policy_kwargs["expectile"])
        
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    image_size = tuple(args.image_size)
    dataset = FrankaImageDataset(
        args.dataset_path,
        image_size=image_size,
        reward_scale=defaults["reward_scale"],
        success_reward=defaults["success_reward"],
        normalize_proprio=defaults["normalize_proprio"],
        normalize_action=defaults["normalize_action"],
    )
    normalization_stats = {
        "state_mean": dataset.state_mean.astype(float).tolist(),
        "state_std": dataset.state_std.astype(float).tolist(),
        "action_mean": dataset.action_mean.astype(float).tolist(),
        "action_std": dataset.action_std.astype(float).tolist(),
        "state_dim": int(dataset.state_dim),
        "action_dim": int(dataset.action_dim),
        "action_type": "threshold_delta_eef",
        "normalize_proprio": bool(defaults["normalize_proprio"]),
        "normalize_action": bool(defaults["normalize_action"]),
        "image_size": list(dataset.image_size),
        "rgb_crop": [0, 90, 360, 480],
        "dataset_path": [os.path.abspath(path) for path in args.dataset_path],
    }
    normalization_stats_path = os.path.join(folder_name, "normalization_stats.json")
    with open(normalization_stats_path, "w") as stats_file:
        json.dump(normalization_stats, stats_file, indent=2)
    print(f"saved normalization stats to: {normalization_stats_path}")

    state_dim = dataset.state_dim
    action_dim = dataset.action_dim
    image_shape = (3, image_size[0], image_size[1])
    latent_dim = int(action_dim * 2)
    min_v = 0.0 if defaults["min_v"] is None else defaults["min_v"]
    if defaults["max_v"] is None:
        max_v = float(
            max(
                dataset.rewards.max(),
                defaults["reward_scale"] * defaults["success_reward"],
                1.0,
            )
        )
    else:
        max_v = defaults["max_v"]

    print(
        "vision train dims:",
        f"image_shape={image_shape}",
        f"state_dim={state_dim}",
        f"action_dim={action_dim}",
        f"latent_dim={latent_dim}",
        f"encoder_mode={args.encoder_mode}",
        f"concat_mode={args.concat_mode}",
        f"zipper_backbone={args.zipper_backbone}",
        f"value_range=[{min_v:.3f}, {max_v:.3f}]",
    )

    train_indices, val_indices, val_episodes = split_train_val_episode_indices(
        dataset, args.val_fraction, args.seed
    )
    raw_val_size = len(val_indices)
    if defaults["val_chain_threshold_transitions"] and val_episodes:
        val_episodes = chain_threshold_episodes(dataset, val_episodes)
        val_indices = [idx for episode in val_episodes for idx in episode]
    train_dataset = Subset(dataset, train_indices)
    val_dataset = NoAugmentSubset(dataset, val_indices)
    val_batch_size = defaults["val_batch_size"] or args.batch_size
    eval_loss_plot_dir = os.path.join(folder_name, "eval_loss_plots")
    print(
        "dataset split:",
        f"train={len(train_dataset)}",
        f"val={len(val_dataset)}",
        f"raw_val={raw_val_size}",
        f"val_episodes={len(val_episodes)}",
        f"val_fraction={args.val_fraction}",
        f"val_chain_threshold={defaults['val_chain_threshold_transitions']}",
    )

    dataloader = DataLoader(
        train_dataset,
        sampler=torch.utils.data.RandomSampler(
            train_dataset,
            num_samples=args.batch_size * args.steps_per_epoch,
            replacement=True,
        ),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=defaults["pin_memory"],
        drop_last=True,
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=val_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=defaults["pin_memory"],
        drop_last=False,
    )

    policy = algos.Latent(
        state_dim,
        action_dim,
        latent_dim,
        min_v,
        max_v,
        device=args.device,
        image_shape=image_shape,
        encoder_mode=args.encoder_mode,
        concat_mode=args.concat_mode,
        zipper_backbone=args.zipper_backbone,
        zipper_normalize_image=args.zipper_normalize_image,
        vae=args.vae,
        **policy_kwargs,
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
            mean_recon = safe_mean(rec_list)
            val_metrics = evaluate_vae_validation(policy, val_dataloader)
            plot_info = None
            if (
                defaults["eval_loss_plot_freq"] > 0
                and epoch_idx % defaults["eval_loss_plot_freq"] == 0
            ):
                plot_info = save_epoch_loss_plot(
                    policy,
                    dataset,
                    val_episodes,
                    epoch_idx,
                    eval_loss_plot_dir,
                    batch_size=val_batch_size,
                    seed=args.seed,
                )
                if plot_info is not None:
                    print(
                        "saved eval loss plot:",
                        plot_info["path"],
                        f"episode={plot_info['episode']}",
                        f"dataset={plot_info['dataset']}",
                        f"dataset_episode={plot_info['dataset_episode']}",
                        f"frames={plot_info['raw_start']}..{plot_info['raw_end']}",
                        f"transitions={plot_info['transitions']}",
                        f"recon={plot_info['recon']:.6f}",
                        f"raw_recon={plot_info['raw_recon']:.6f}",
                        f"raw_dim_mse={format_dim_values(plot_info['raw_recon_dim_mse'])}",
                        f"raw_dim_rmse={format_dim_values(plot_info['raw_recon_dim_rmse'])}",
                        f"kl={plot_info['kl']:.6f}",
                    )
                elif not val_episodes:
                    print("skip eval loss plot: no validation episodes were found")
            is_best = np.isfinite(mean_recon) and mean_recon < best_recon_loss
            if is_best:
                best_recon_loss = mean_recon
                if defaults["save_model"]:
                    policy.save(
                        "model",
                        folder_name,
                        is_best=True,
                        best_metric=best_recon_loss,
                        best_epoch=int(epoch_idx),
                    )
                    print(
                        f"save best model success: epoch={epoch_idx}, "
                        f"L_Recon={best_recon_loss:.6f}"
                    )

            if epoch_idx % args.save_freq == 0 and defaults["save_model"] and epoch_idx != 0:
                policy.save("model", folder_name)

            logger.record_tabular("Training Epochs", int(epoch_idx))
            logger.record_tabular("L_Act", safe_mean(act_list))
            logger.record_tabular("L_Crit", mean_crit)
            logger.record_tabular("L_Recon", mean_recon)
            logger.record_tabular("L_KL", safe_mean(kl_list))
            logger.record_tabular("Val_Recon", val_metrics["recon"])
            logger.record_tabular("Val_Recon_Sum", val_metrics["recon_sum"])
            logger.record_tabular("Val_Raw_Recon", val_metrics["raw_recon"])
            logger.record_tabular("Val_Raw_Recon_Sum", val_metrics["raw_recon_sum"])
            for dim, value in enumerate(val_metrics["raw_recon_dim_mse"]):
                logger.record_tabular(f"Val_Raw_Recon_Dim{dim}_MSE", value)
            for dim, value in enumerate(val_metrics["raw_recon_dim_rmse"]):
                logger.record_tabular(f"Val_Raw_Recon_Dim{dim}_RMSE", value)
            logger.record_tabular("Val_KL", val_metrics["kl"])
            logger.record_tabular(
                "Val_Plot_Recon",
                plot_info["recon"] if plot_info is not None else float("nan"),
            )
            logger.record_tabular(
                "Val_Plot_Raw_Recon",
                plot_info["raw_recon"] if plot_info is not None else float("nan"),
            )
            logger.record_tabular(
                "Val_Plot_KL",
                plot_info["kl"] if plot_info is not None else float("nan"),
            )
            logger.record_tabular("Weight", safe_mean(w_list))
            logger.record_tabular("kl", policy.kl_beta)
            logger.dump_tabular()

            tglobal.set_postfix(
                w=safe_mean(w_list),
                act=safe_mean(act_list),
                crit=safe_mean(crit_list),
                rec=safe_mean(rec_list),
                kl=safe_mean(kl_list),
                val_rec=val_metrics["recon"],
                val_raw_rec=val_metrics["raw_recon"],
                val_kl=val_metrics["kl"],
            )

    policy.save("model", folder_name)

    if terminal_log_file is not None:
        terminal_log_file.close()
