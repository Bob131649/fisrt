#!/usr/bin/env python3

import argparse
import html
import re
from collections import Counter
from pathlib import Path


EPISODE_START_RE = re.compile(
    r"eval episode (\d+): .*goal=\[([^\]]+)\], start_id=(\d+)"
)


def parse_float_list(text):
    return [float(part.strip()) for part in text.split(",")]


def infer_failure_step(records):
    step_counts = [record["steps"] for record in records]
    if not step_counts:
        raise ValueError("No eval records found in the log.")
    return Counter(step_counts).most_common(1)[0][0]


def parse_eval_records(log_path):
    lines = Path(log_path).read_text(encoding="utf-8").splitlines()
    records = []
    pending = None

    for line_no, raw_line in enumerate(lines, start=1):
        line = raw_line.strip()

        start_match = EPISODE_START_RE.search(line)
        if start_match:
            pending = {
                "line_no": line_no,
                "episode_idx": int(start_match.group(1)),
                "goal": parse_float_list(start_match.group(2)),
                "start_id": int(start_match.group(3)),
            }
            continue

        if pending is None or not line.startswith("---"):
            continue

        tokens = line.split()
        if len(tokens) < 4 or not tokens[-1].isdigit():
            continue

        records.append(
            {
                **pending,
                "final_x": float(tokens[-3]),
                "final_y": float(tokens[-2]),
                "steps": int(tokens[-1]),
            }
        )
        pending = None

    return records


def select_success_records(records, start_id, failure_step):
    start_records = [record for record in records if record["start_id"] == start_id]
    success_records = [
        record for record in start_records if record["steps"] < failure_step
    ]
    return start_records, success_records


def build_default_output_path(log_path, start_id):
    log_path = Path(log_path)
    return log_path.with_name(f"{log_path.stem}_start{start_id}_success_steps.svg")


def svg_text(x, y, text, font_size=12, anchor="start", fill="#111111"):
    safe_text = html.escape(str(text))
    return (
        f'<text x="{x}" y="{y}" font-size="{font_size}" text-anchor="{anchor}" '
        f'fill="{fill}" font-family="Arial, Helvetica, sans-serif">{safe_text}</text>'
    )


def svg_rotated_text(x, y, text, angle=-60, font_size=10, anchor="end", fill="#444444"):
    safe_text = html.escape(str(text))
    return (
        f'<text x="{x}" y="{y}" font-size="{font_size}" text-anchor="{anchor}" '
        f'fill="{fill}" font-family="Arial, Helvetica, sans-serif" '
        f'transform="rotate({angle} {x} {y})">{safe_text}</text>'
    )


def build_svg_chart(success_records, start_id, failure_step):
    width = 1200
    height = 700
    margin_left = 80
    margin_right = 40
    margin_top = 80
    margin_bottom = 170

    chart_width = width - margin_left - margin_right
    chart_height = height - margin_top - margin_bottom
    max_step = max(max(record["steps"] for record in success_records), failure_step)
    step_counter = Counter(record["steps"] for record in success_records)
    sorted_steps = sorted(step_counter.items())
    count = len(sorted_steps)
    slot_width = chart_width / max(count, 1)
    bar_width = max(min(slot_width * 0.72, 36), 6)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        svg_text(width / 2, 36, f"Successful Eval Step Histogram for start_id={start_id}", 24, "middle"),
        svg_text(width / 2, 62, f"success if steps < {failure_step}", 13, "middle", "#666666"),
        f'<line x1="{margin_left}" y1="{margin_top + chart_height}" x2="{margin_left + chart_width}" y2="{margin_top + chart_height}" stroke="#222222" stroke-width="1.5"/>',
        f'<line x1="{margin_left}" y1="{margin_top}" x2="{margin_left}" y2="{margin_top + chart_height}" stroke="#222222" stroke-width="1.5"/>',
    ]

    max_episode_count = max(step_counter.values())
    for tick_idx in range(6):
        value = max_episode_count * tick_idx / 5.0
        y = (
            margin_top + chart_height - (value / max_episode_count) * chart_height
            if max_episode_count
            else margin_top + chart_height
        )
        parts.append(
            f'<line x1="{margin_left}" y1="{y:.2f}" x2="{margin_left + chart_width}" y2="{y:.2f}" stroke="#E6E6E6" stroke-width="1"/>'
        )
        parts.append(svg_text(margin_left - 10, y + 4, int(round(value)), 11, "end", "#555555"))

    for idx, (step_value, episode_count) in enumerate(sorted_steps, start=1):
        x_center = margin_left + (idx - 0.5) * slot_width
        bar_height = (
            (episode_count / max_episode_count) * chart_height if max_episode_count else 0
        )
        x = x_center - bar_width / 2
        y = margin_top + chart_height - bar_height
        tooltip = (
            f"steps={step_value}, successful_episode_count={episode_count}"
        )
        parts.append(
            f'<g><title>{html.escape(tooltip)}</title>'
            f'<rect x="{x:.2f}" y="{y:.2f}" width="{bar_width:.2f}" height="{bar_height:.2f}" '
            f'fill="#4C78A8" stroke="#1F3552" stroke-width="1"/></g>'
        )
        parts.append(
            svg_rotated_text(
                x_center,
                margin_top + chart_height + 54,
                step_value,
                angle=-60,
                font_size=9,
                anchor="end",
                fill="#444444",
            )
        )
        parts.append(svg_text(x_center, y - 6, episode_count, 10, "middle", "#222222"))

    parts.append(svg_text(width / 2, height - 22, "Step Count", 14, "middle"))
    parts.append(
        f'<text x="24" y="{margin_top + chart_height / 2}" font-size="14" fill="#111111" '
        'font-family="Arial, Helvetica, sans-serif" transform="rotate(-90 24 '
        f'{margin_top + chart_height / 2})" text-anchor="middle">Episode Count</text>'
    )
    parts.append("</svg>")
    return "\n".join(parts)


def main():
    parser = argparse.ArgumentParser(
        description="Read an eval log and plot successful step counts for a chosen start_id."
    )
    parser.add_argument("--log_path", required=True, type=str)
    parser.add_argument("--start_id", required=True, type=int)
    parser.add_argument(
        "--output",
        default=None,
        type=str,
        help="Output SVG path. Defaults to <log_stem>_start<id>_success_steps.svg",
    )
    parser.add_argument(
        "--failure_step",
        default=None,
        type=int,
        help="Treat steps < failure_step as success. Defaults to the most common step count in the log.",
    )
    args = parser.parse_args()

    records = parse_eval_records(args.log_path)
    failure_step = args.failure_step or infer_failure_step(records)
    start_records, success_records = select_success_records(
        records, args.start_id, failure_step
    )

    if not start_records:
        raise ValueError(f"No eval records found for start_id={args.start_id}.")
    if not success_records:
        raise ValueError(
            f"No successful eval records found for start_id={args.start_id} "
            f"under failure_step={failure_step}."
        )

    output_path = Path(args.output) if args.output else build_default_output_path(args.log_path, args.start_id)
    svg_content = build_svg_chart(success_records, args.start_id, failure_step)
    output_path.write_text(svg_content, encoding="utf-8")

    steps = [record["steps"] for record in success_records]
    print(f"parsed records for start_id={args.start_id}: {len(start_records)}")
    print(f"successful records: {len(success_records)}")
    print(f"failure_step threshold: {failure_step}")
    print(f"min/mean/max success steps: {min(steps)}/{sum(steps)/len(steps):.2f}/{max(steps)}")
    print(f"saved figure to: {output_path}")


if __name__ == "__main__":
    main()
