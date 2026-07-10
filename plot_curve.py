#!/usr/bin/env python3
import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt


EPOCH_RE = re.compile(r"Training Epochs\s+([-+]?\d+)")
NORM_RE = re.compile(r"NormReturn\s+([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)")


def read_norm_returns(log_path):
    epochs, norm_returns = [], []
    pending_epoch = None

    with log_path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            epoch_match = EPOCH_RE.search(line)
            if epoch_match:
                pending_epoch = int(epoch_match.group(1))
                continue

            norm_match = NORM_RE.search(line)
            if norm_match:
                norm_returns.append(float(norm_match.group(1)))
                epochs.append(pending_epoch if pending_epoch is not None else len(norm_returns) - 1)
                pending_epoch = None

    return epochs, norm_returns


def main():
    parser = argparse.ArgumentParser(description="Plot NormReturn curves from debug.log files.")
    parser.add_argument(
        "--exp_id",
        nargs="+",
        default=[100],
        type=int,
        help="Experiment number(s) after Exp. Default: 100, meaning results/Exp0100",
    )
    parser.add_argument(
        "--exp_dir",
        default=None,
        help="Experiment directory that contains env subfolders. Overrides --exp_id if set.",
    )
    parser.add_argument(
        "--env_name",
        nargs="+",
        default=["maze2d-large-v1"],
        # required=True,
        help="One or more env folder names under exp_dir, e.g. maze2d-large-v1 antmaze-large-diverse-v2",
    )
    parser.add_argument(
        "--x",
        choices=["eval", "epoch"],
        default="eval",
        help="X axis. Use eval index by default because debug.log may contain appended restarted runs.",
    )
    parser.add_argument("--output", default="normreturn_curve.png", help="Output image path.")
    args = parser.parse_args()

    if args.exp_dir is not None:
        exp_dirs = [Path(args.exp_dir)]
    else:
        exp_ids = args.exp_id if isinstance(args.exp_id, list) else [args.exp_id]
        exp_dirs = [Path("results") / f"Exp{exp_id:04d}" for exp_id in exp_ids]

    plt.figure(figsize=(8, 5))

    plotted = 0
    for exp_dir in exp_dirs:
        for env_name in args.env_name:
            log_path = exp_dir / env_name / "debug.log"
            if not log_path.exists():
                print(f"[skip] missing: {log_path}")
                continue

            epochs, norm_returns = read_norm_returns(log_path)
            if not norm_returns:
                print(f"[skip] no NormReturn found in: {log_path}")
                continue

            x_values = list(range(len(norm_returns))) if args.x == "eval" else epochs
            label = f"{exp_dir.name}/{env_name}"
            plt.plot(x_values, norm_returns, marker="o", linewidth=1.8, markersize=3.5, label=label)
            print(f"[ok] {label}: {len(norm_returns)} points, last={norm_returns[-1]:.6g}")
            plotted += 1

    if plotted == 0:
        raise SystemExit("No curves were plotted.")

    plt.xlabel("Evaluation" if args.x == "eval" else "Training Epoch")
    plt.ylabel("NormReturn")
    plt.title("NormReturn Curves")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.output, dpi=200)
    print(f"saved: {args.output}")


if __name__ == "__main__":
    main()
