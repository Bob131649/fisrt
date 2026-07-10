import os

import numpy as np
import torch
import torch.nn.functional as F

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def safe_mean(values):
    return float(np.mean(values)) if values else float("nan")


def safe_column_mean(values):
    return np.mean(np.stack(values, axis=0), axis=0) if values else np.asarray([])


def eval_ref_metrics(policy, water_pipe, dataloader, num_batches=4):
    rec_list, rec_sum_list, raw_rec_list, raw_rec_sum_list, raw_rec_dim_list, kl_list = [], [], [], [], [], []
    was_training = policy.training
    policy.eval()
    with torch.no_grad():
        for batch_idx, cpu_batch in enumerate(dataloader):
            if batch_idx >= num_batches:
                break
            state, action, _, _, _, _ = water_pipe.batch_to_device(cpu_batch)
            recons_action, mu, log_var = policy.actor_vae(state, action)
            rec_per_dim = F.mse_loss(recons_action, action, reduction="none")
            rec = rec_per_dim.mean()
            rec_sum = rec_per_dim.sum(dim=1).mean()

            raw_action = water_pipe.unnormalize_action(action)
            raw_recons_action = water_pipe.unnormalize_action(recons_action)
            raw_rec_per_dim = F.mse_loss(raw_recons_action, raw_action, reduction="none")
            raw_rec = raw_rec_per_dim.mean()
            raw_rec_sum = raw_rec_per_dim.sum(dim=1).mean()

            free_bits = 0.5
            kl_per_dim = -0.5 * (1 + log_var - mu.pow(2) - log_var.exp())
            kl = torch.maximum(
                kl_per_dim,
                torch.tensor(free_bits, device=kl_per_dim.device),
            ).sum(dim=1).mean()

            rec_list.append(rec.item())
            rec_sum_list.append(rec_sum.item())
            raw_rec_list.append(raw_rec.item())
            raw_rec_sum_list.append(raw_rec_sum.item())
            raw_rec_dim_list.append(raw_rec_per_dim.mean(dim=0).detach().cpu().numpy())
            kl_list.append(kl.item())
    if was_training:
        policy.train()
    raw_rec_dim_mse = safe_column_mean(raw_rec_dim_list)
    return {
        "rec": safe_mean(rec_list),
        "rec_sum": safe_mean(rec_sum_list),
        "raw_rec": safe_mean(raw_rec_list),
        "raw_rec_sum": safe_mean(raw_rec_sum_list),
        "raw_rec_dim_mse": raw_rec_dim_mse.tolist(),
        "raw_rec_dim_rmse": np.sqrt(raw_rec_dim_mse).tolist(),
        "kl": safe_mean(kl_list),
    }


def collect_eval_points(policy, expert_pipe, random_pipe, sample_size):
    expert_size = max(1, sample_size // 2)
    random_size = max(1, sample_size - expert_size)
    expert_loader = expert_pipe.make_dataloader(expert_size, epoch_size=1, num_workers=0)
    random_loader = random_pipe.make_dataloader(random_size, epoch_size=1, num_workers=0)

    rows = []
    was_training = policy.training
    policy.eval()
    with torch.no_grad():
        for label, pipe, loader in (
            ("expert", expert_pipe, expert_loader),
            ("random", random_pipe, random_loader),
        ):
            state, action, _, _, _, _ = pipe.batch_to_device(next(iter(loader)))
            raw_state = pipe.unnormalize_state(state)
            xyz = raw_state["proprio"][:, :3].detach().cpu().numpy()

            latent = policy.actor(state)
            policy_action = policy.actor_vae_target.decode(state, z=latent)
            q1, q2 = policy.critic(state, policy_action)
            q_pi = torch.min(q1, q2).detach().cpu().numpy().reshape(-1)

            recons_action, _, _ = policy.actor_vae(state, action)
            recon = F.mse_loss(recons_action, action, reduction="none").sum(dim=1)
            recon = recon.detach().cpu().numpy().reshape(-1)

            raw_action = pipe.unnormalize_action(action)
            raw_recons_action = pipe.unnormalize_action(recons_action)
            raw_recon = F.mse_loss(raw_recons_action, raw_action, reduction="none").sum(dim=1)
            raw_recon = raw_recon.detach().cpu().numpy().reshape(-1)

            rows.append(
                {
                    "label": label,
                    "xyz": xyz,
                    "q_pi": q_pi,
                    "recon": recon,
                    "raw_recon": raw_recon,
                }
            )
    if was_training:
        policy.train()
    return rows


def plot_eval_heatmap(rows, path):
    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, projection="3d")
    markers = {"expert": "o", "random": "^"}
    all_values = np.concatenate([row["q_pi"] for row in rows], axis=0)
    vmin, vmax = np.percentile(all_values, [5, 95])
    if np.isclose(vmin, vmax):
        vmin, vmax = None, None
    last_scatter = None
    for row in rows:
        xyz = row["xyz"]
        last_scatter = ax.scatter(
            xyz[:, 0],
            xyz[:, 1],
            xyz[:, 2],
            c=row["q_pi"],
            cmap="viridis",
            marker=markers[row["label"]],
            alpha=0.6,
            s=10,
            vmin=vmin,
            vmax=vmax,
            label=row["label"],
        )
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_title("Policy Q heatmap over sampled XYZ")
    ax.legend(loc="best")
    fig.colorbar(last_scatter, ax=ax, label="min Q(policy action)")
    fig.tight_layout()
    fig.savefig(path, dpi=250)
    plt.close(fig)


def plot_eval_recon(rows, path):
    fig = plt.figure(figsize=(12, 5))
    ax_xyz = fig.add_subplot(121, projection="3d")
    ax_hist = fig.add_subplot(122)
    markers = {"expert": "o", "random": "^"}
    all_values = np.concatenate([row["raw_recon"] for row in rows], axis=0)
    vmax = np.percentile(all_values, 95)
    if np.isclose(vmax, 0):
        vmax = None
    last_scatter = None
    for row in rows:
        xyz = row["xyz"]
        last_scatter = ax_xyz.scatter(
            xyz[:, 0],
            xyz[:, 1],
            xyz[:, 2],
            c=row["raw_recon"],
            cmap="magma",
            marker=markers[row["label"]],
            alpha=0.6,
            s=10,
            vmin=0,
            vmax=vmax,
            label=row["label"],
        )
        ax_hist.hist(row["raw_recon"], bins=40, alpha=0.45, label=row["label"])
    ax_xyz.set_xlabel("x")
    ax_xyz.set_ylabel("y")
    ax_xyz.set_zlabel("z")
    ax_xyz.set_title("Raw recon loss over sampled XYZ")
    ax_xyz.legend(loc="best")
    fig.colorbar(last_scatter, ax=ax_xyz, label="raw action recon MSE sum")
    ax_hist.set_title("Raw recon loss distribution")
    ax_hist.set_xlabel("raw action recon MSE sum")
    ax_hist.set_ylabel("count")
    ax_hist.legend(loc="best")
    fig.tight_layout()
    fig.savefig(path, dpi=250)
    plt.close(fig)


def run_visual_eval(policy, expert_pipe, random_pipe, folder_name, epoch_idx, sample_size):
    rows = collect_eval_points(policy, expert_pipe, random_pipe, sample_size)
    heatmap_path = os.path.join(folder_name, f"eval_heatmap_epoch{epoch_idx:04d}.png")
    recon_path = os.path.join(folder_name, f"eval_recon_epoch{epoch_idx:04d}.png")
    plot_eval_heatmap(rows, heatmap_path)
    plot_eval_recon(rows, recon_path)
    return {"heatmap_path": heatmap_path, "recon_path": recon_path}
