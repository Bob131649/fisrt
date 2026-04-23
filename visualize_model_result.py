#!/usr/bin/env python3

import argparse
import os
import re

import matplotlib.pyplot as plt


EPOCH_PATTERN = re.compile(r"Training Epochs\s+([-+]?\d+)")
NORM_PATTERN = re.compile(r"NormReturn\s+([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)")


def find_debug_logs(results_dir):
    debug_logs = []
    for root, _, files in os.walk(results_dir):
        if "debug.log" in files:
            debug_logs.append(os.path.join(root, "debug.log"))
    debug_logs.sort()
    return debug_logs


def find_task_debug_log(results_dir, task_name):
    debug_log_path = os.path.join(results_dir, task_name, "debug.log")
    if not os.path.isfile(debug_log_path):
        raise FileNotFoundError(
            f"Could not find debug.log for task '{task_name}' under: {results_dir}"
        )
    return debug_log_path


def parse_debug_log(debug_log_path):
    epochs = []
    norm_returns = []

    current_epoch = None
    with open(debug_log_path, "r", encoding="utf-8") as f:
        for line in f:
            epoch_match = EPOCH_PATTERN.search(line)
            if epoch_match:
                current_epoch = int(epoch_match.group(1))
                continue

            norm_match = NORM_PATTERN.search(line)
            if norm_match and current_epoch is not None:
                epochs.append(current_epoch)
                norm_returns.append(float(norm_match.group(1)))
                current_epoch = None

    return epochs, norm_returns


def collect_task_curves(results_dir):
    task_curves = []
    for debug_log_path in find_debug_logs(results_dir):
        task_name = os.path.basename(os.path.dirname(debug_log_path))
        epochs, norm_returns = parse_debug_log(debug_log_path)
        if not epochs:
            continue
        task_curves.append((task_name, epochs, norm_returns))
    return task_curves


def collect_selected_task_curves(results_dir, task_names):
    task_curves = []
    for task_name in task_names:
        debug_log_path = find_task_debug_log(results_dir, task_name)
        epochs, norm_returns = parse_debug_log(debug_log_path)
        if not epochs:
            raise ValueError(f"No valid NormReturn curve found in: {debug_log_path}")
        task_curves.append((task_name, epochs, norm_returns))
    return task_curves


def plot_task_curves(task_curves, output_path, figsize=(8.5, 5.2)):
    fig, ax = plt.subplots(figsize=figsize)

    for task_name, epochs, norm_returns in task_curves:
        ax.plot(epochs, norm_returns, linewidth=2.0, label=task_name)

    ax.set_title("NormReturn Curves", fontsize=14)
    ax.set_xlabel("epoch number", fontsize=11)
    ax.set_ylabel("normreturn", fontsize=11)
    ax.grid(True, alpha=0.25, linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(fontsize=10)

    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Saved figure to: {output_path}")
    plt.show()


def main():
    parser = argparse.ArgumentParser(
        description="Visualize NormReturn curves from debug.log files under the results directory."
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default="./results",
        help="Path to the results directory.",
    )
    parser.add_argument(
        "task_names",
        nargs="*",
        help="Optional task names under results_dir. Pass multiple names separated by spaces.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="./results/model_result_summary.png",
        help="Output figure path.",
    )
    args = parser.parse_args()

    if args.task_names:
        task_curves = collect_selected_task_curves(args.results_dir, args.task_names)
    else:
        task_curves = collect_task_curves(args.results_dir)

    if not task_curves:
        raise ValueError(f"No valid debug.log files with NormReturn found under: {args.results_dir}")

    os.makedirs(os.path.dirname(os.path.abspath(args.output_path)), exist_ok=True)
    plot_task_curves(task_curves, args.output_path)


if __name__ == "__main__":
    main()
