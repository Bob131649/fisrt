import argparse
import json
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def get_franka_dataset_paths():
    return [
        "./datasets/real_dataset/expert/franka_pickplace_ep20_20260629_152135_add_rewarding20.hdf5",
        "./datasets/real_dataset/expert/franka_pickplace_ep30_20260626_162422_add_rewarding20.hdf5",
        "./datasets/real_dataset/expert/franka_pickplace_ep31_20260625_112544_add_rewarding20.hdf5",
        "./datasets/real_dataset/expert/franka_pickplace_ep39_20260626_143912_add_rewarding20.hdf5",
    ], []


def choose_dataset_paths(paths, seed, dataset_index=None):
    if not paths:
        return []
    if dataset_index is not None:
        if dataset_index < 0 or dataset_index >= len(paths):
            raise ValueError(f"--dataset_index must be in [0, {len(paths) - 1}], got {dataset_index}")
        chosen_index = dataset_index
    else:
        rng = np.random.default_rng(seed)
        chosen_index = int(rng.integers(len(paths)))
    print(f"plot dataset [{chosen_index}/{len(paths)}]: {paths[chosen_index]}")
    return [paths[chosen_index]]


def plot_xyz_value(
    policy,
    water_pipe,
    mode,
    path,
    sample_size=1500,
    value_min=None,
    value_max=None,
):
    fig = build_xyz_value_figure(
        policy,
        water_pipe,
        mode,
        sample_size=sample_size,
        value_min=value_min,
        value_max=value_max,
    )
    fig.savefig(path, dpi=250)
    plt.close(fig)


def print_value_stats(label, value):
    value_p10, value_p90 = np.percentile(value, [10, 90])
    print(
        f"{label} stats: "
        f"min={value.min():.6f}, "
        f"max={value.max():.6f}, "
        f"mean={value.mean():.6f}, "
        f"p10={value_p10:.6f}, "
        f"p90={value_p90:.6f}"
    )


def plot_xyz_value_mixed(
    policy,
    expert_pipe,
    random_pipe,
    mode,
    path,
    sample_size=1500,
    value_min=None,
    value_max=None,
):
    fig = build_xyz_value_mixed_figure(
        policy,
        expert_pipe,
        random_pipe,
        mode,
        sample_size=sample_size,
        value_min=value_min,
        value_max=value_max,
    )
    fig.savefig(path, dpi=250)
    plt.close(fig)


def build_xyz_value_figure(
    policy,
    water_pipe,
    mode,
    sample_size=1500,
    value_min=None,
    value_max=None,
):
    import torch

    dataloader = water_pipe.make_dataloader(sample_size, epoch_size=1, num_workers=0)
    state, action, _, _, _, _ = water_pipe.batch_to_device(next(iter(dataloader)))
    raw_state = water_pipe.unnormalize_state(state)
    xyz = raw_state["proprio"][:, :3].detach().cpu().numpy()

    was_training = policy.training
    policy.eval()
    with torch.no_grad():
        if mode == "q":
            value = policy.get_a_q(state, action)
        elif mode == "v":
            value = policy.get_v(state)
        else:
            value = torch.linalg.norm(action, dim=1, keepdim=True)
    if was_training:
        policy.train()

    value = value.detach().cpu().numpy().squeeze()
    print_value_stats(f"{mode}_value", value)

    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, projection="3d")
    sc = ax.scatter(
        xyz[:, 0],
        xyz[:, 1],
        xyz[:, 2],
        c=value,
        cmap="viridis",
        alpha=0.55,
        s=9,
        vmin=value_min,
        vmax=value_max,
    )
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_title(f"{mode.upper()} over XYZ")
    fig.colorbar(sc, ax=ax, label=f"{mode}_value")
    fig.tight_layout()
    return fig


def build_xyz_value_mixed_figure(
    policy,
    expert_pipe,
    random_pipe,
    mode,
    sample_size=1500,
    value_min=None,
    value_max=None,
):
    import torch

    if mode != "v":
        raise ValueError("mixed expert/random value plot is only supported for mode='v'.")

    expert_size = sample_size // 2
    random_size = sample_size - expert_size

    expert_loader = expert_pipe.make_dataloader(
        expert_size,
        epoch_size=1,
        num_workers=0,
    )
    random_loader = random_pipe.make_dataloader(
        random_size,
        epoch_size=1,
        num_workers=0,
    )
    state_expert, _, _, _, _, _ = expert_pipe.batch_to_device(next(iter(expert_loader)))
    state_random, _, _, _, _, _ = random_pipe.batch_to_device(next(iter(random_loader)))

    raw_expert = expert_pipe.unnormalize_state(state_expert)
    raw_random = random_pipe.unnormalize_state(state_random)
    xyz_expert = raw_expert["proprio"][:, :3].detach().cpu().numpy()
    xyz_random = raw_random["proprio"][:, :3].detach().cpu().numpy()

    was_training = policy.training
    policy.eval()
    with torch.no_grad():
        value_expert = policy.get_v(state_expert)
        value_random = policy.get_v(state_random)
    if was_training:
        policy.train()

    value_expert = value_expert.detach().cpu().numpy().squeeze()
    value_random = value_random.detach().cpu().numpy().squeeze()
    xyz = np.concatenate([xyz_expert, xyz_random], axis=0)
    value = np.concatenate([value_expert, value_random], axis=0)

    print_value_stats(f"{mode}_value mixed", value)
    print_value_stats(f"{mode}_value expert_half", value_expert)
    print_value_stats(f"{mode}_value random_half", value_random)

    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, projection="3d")
    sc = ax.scatter(
        xyz[:, 0],
        xyz[:, 1],
        xyz[:, 2],
        c=value,
        cmap="viridis",
        alpha=0.55,
        s=9,
        vmin=value_min,
        vmax=value_max,
    )
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_title(f"{mode.upper()} over XYZ (expert/random)")
    fig.colorbar(sc, ax=ax, label=f"{mode}_value")
    fig.tight_layout()
    return fig


def show_xyz_value_cv2(
    policy,
    water_pipe,
    mode,
    window_name="xyz_value",
    sample_size=1500,
    value_min=None,
    value_max=None,
):
    import cv2

    fig = build_xyz_value_figure(
        policy,
        water_pipe,
        mode,
        sample_size=sample_size,
        value_min=value_min,
        value_max=value_max,
    )
    fig.canvas.draw()
    image = np.asarray(fig.canvas.buffer_rgba())
    image = cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)
    plt.close(fig)

    cv2.imshow(window_name, image)
    print("press any key in the cv2 window to close")
    cv2.waitKey(0)
    cv2.destroyWindow(window_name)


def load_normalization_stats(path):
    with open(path, "r") as f:
        stats = json.load(f)
    return {
        "s_mean": np.asarray(stats["state_mean"], dtype=np.float32),
        "s_std": np.asarray(stats["state_std"], dtype=np.float32),
        "a_mean": np.asarray(stats["action_mean"], dtype=np.float32),
        "a_std": np.asarray(stats["action_std"], dtype=np.float32),
    }


def find_latest_checkpoint_dir(log_dir, exp_id, env_name):
    model_dir = Path(log_dir) / f"Exp{exp_id:04d}" / env_name
    if not model_dir.exists():
        raise FileNotFoundError(f"model directory not found: {model_dir}")

    pth_files = sorted(model_dir.glob("*.pth"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not pth_files:
        raise FileNotFoundError(f"no .pth files found in: {model_dir}")

    return model_dir, pth_files[0]


def build_water_pipe(args, folder_name, state_dim, action_dim):
    from datasets.utils import get_real_dataset
    from replay_buffer.vision_numpy_buffer import WaterPipeDataset

    dataset_path_expert_list, dataset_path_random_list = get_franka_dataset_paths()
    dataset_path_expert_list = choose_dataset_paths(
        dataset_path_expert_list,
        seed=args.seed,
        dataset_index=args.dataset_index,
    )
    dataset_expert = [get_real_dataset(path) for path in dataset_path_expert_list]
    dataset_random = []

    expert_pipe = WaterPipeDataset(args.env_name, state_dim, action_dim, args.device)
    expert_pipe.load(dataset_expert)

    stats_path = Path(folder_name) / "normalization_stats.json"
    if stats_path.exists():
        normalize_stats = load_normalization_stats(stats_path)
    else:
        stats_buffers = [expert_pipe]
        if dataset_random:
            random_pipe = WaterPipeDataset(args.env_name, state_dim, action_dim, args.device)
            random_pipe.load(dataset_random)
            stats_buffers.append(random_pipe)
        normalize_stats = WaterPipeDataset.compute_joint_stats(stats_buffers)

    expert_pipe.apply_stats(normalize_stats)
    return expert_pipe


def build_policy(args, state_dim, action_dim):
    import algos.vision_algos_reference as algos

    min_v = 0
    max_v = 100
    latent_dim = max(4, int(action_dim * 2.0))
    return algos.Latent(
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
        max_latent_action=args.max_latent_action,
        expectile=args.expectile,
        kl_beta=args.kl_beta,
        doubleq_min=args.doubleq_min,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="q", choices=["q", "v", "pi"], type=str)
    parser.add_argument("--ExpID", default=2, type=int)
    parser.add_argument("--log_dir", default="./results/", type=str)
    parser.add_argument("--env_name", default="franka")
    parser.add_argument("--seed", default=789, type=int)
    parser.add_argument("--sample_size", default=1500, type=int)
    parser.add_argument("--dataset_index", default=None, type=int)
    parser.add_argument("--device", default="cuda:0", type=str)

    parser.add_argument("--vae_lr", default=2e-4, type=float)
    parser.add_argument("--actor_lr", default=2e-4, type=float)
    parser.add_argument("--critic_lr", default=2e-4, type=float)
    parser.add_argument("--tau", default=0.003, type=float)
    parser.add_argument("--discount", default=0.99, type=float)
    parser.add_argument("--expectile", default=0.85, type=float)
    parser.add_argument("--kl_beta", default=1.0, type=float)
    parser.add_argument("--max_latent_action", default=0.675, type=float)
    parser.add_argument("--doubleq_min", default=1.0, type=float)
    args = parser.parse_args()

    import torch

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    state_dim = 8
    action_dim = 8
    folder_name, latest_pth = find_latest_checkpoint_dir(args.log_dir, args.ExpID, args.env_name)
    water_pipe = build_water_pipe(args, folder_name, state_dim, action_dim)
    policy = build_policy(args, state_dim, action_dim)
    policy.load("model", str(folder_name))

    value_ranges = {
        "q": (0, 110),
        "v": (0, 100),
        "pi": (0, 100),
    }
    value_min, value_max = value_ranges[args.mode]
    show_xyz_value_cv2(
        policy,
        water_pipe,
        args.mode,
        window_name=f"xyz_{args.mode}_Exp{args.ExpID:04d}",
        sample_size=args.sample_size,
        value_min=value_min,
        value_max=value_max,
    )
    print(f"loaded checkpoint dir: {folder_name}")
    print(f"latest pth: {latest_pth}")


if __name__ == "__main__":
    main()
