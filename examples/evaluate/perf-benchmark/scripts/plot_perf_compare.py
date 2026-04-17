#!/usr/bin/env python3
"""Multi-version performance comparison plots.

Overlays smoothed performance curves for multiple model versions on the
same axes, with one PNG produced per (subset, metric) combination.

Usage:
    python plot_perf_compare.py \\
        --source "No Spec=nospec/results.csv" \\
        --source "DFlash=dflash/results.csv" \\
        --source "Eagle3 k3=eagle3/sweep_HumanEval.json" \\
        --metric latency --metric itl \\
        --output-dir ./plots
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt

from perf_utils import (
    METRICS,
    load_data,
    parse_source_args,
    pretty_subset,
    smooth_curve,
)

COLOR_CYCLE = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
    "#9467bd", "#8c564b", "#e377c2", "#7f7f7f",
    "#bcbd22", "#17becf",
]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Multi-version performance comparison plots."
    )
    parser.add_argument(
        "--source",
        action="append",
        required=True,
        metavar="LABEL=PATH",
        help=(
            "Version data source as 'Label=path/to/file'. "
            "Repeatable; same label with multiple paths pools repetitions."
        ),
    )
    parser.add_argument(
        "--metric",
        action="append",
        choices=list(METRICS.keys()),
        metavar="METRIC",
        help=f"Metric(s) to plot (default: latency). Choices: {', '.join(METRICS)}",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("."),
        help="Directory for output PNGs (default: current directory)",
    )
    parser.add_argument(
        "--subsets",
        type=str,
        default=None,
        help="Comma-separated subset filter (default: all found in data)",
    )
    args = parser.parse_args()

    metrics = args.metric or ["latency"]
    subset_filter = set(args.subsets.split(",")) if args.subsets else None

    try:
        sources = parse_source_args(args.source)
    except (ValueError, FileNotFoundError) as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        sys.exit(1)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    for metric_name in metrics:
        metric_cfg = METRICS[metric_name]
        increasing = metric_cfg["increasing"]

        # {label: {subset: [(rps, y), ...]}}
        all_data: dict[str, dict[str, list[tuple[float, float]]]] = defaultdict(
            lambda: defaultdict(list)
        )

        for label, paths in sources.items():
            for path in paths:
                try:
                    file_data = load_data(path, metric_name)
                except (ValueError, FileNotFoundError) as e:
                    print(f"[WARN] {e}", file=sys.stderr)
                    continue
                for subset, points in file_data.items():
                    all_data[label][subset].extend(points)

        all_subsets: set[str] = set()
        for label_data in all_data.values():
            all_subsets.update(label_data.keys())

        if subset_filter:
            all_subsets &= subset_filter

        if not all_subsets:
            print(f"[WARN] No data found for metric '{metric_name}'", file=sys.stderr)
            continue

        for subset in sorted(all_subsets):
            fig, ax = plt.subplots(figsize=(8, 5))

            for i, label in enumerate(sources.keys()):
                points = all_data[label].get(subset, [])
                if not points:
                    continue

                color = COLOR_CYCLE[i % len(COLOR_CYCLE)]
                points.sort(key=lambda p: p[0])
                xs = [p[0] for p in points]
                ys = [p[1] for p in points]

                ax.scatter(xs, ys, color=color, alpha=0.35, s=25, zorder=3)

                x_smooth, y_smooth = smooth_curve(xs, ys, increasing=increasing)
                ax.plot(
                    x_smooth, y_smooth,
                    color=color, linewidth=2, label=label, zorder=4,
                )

            title = f"{pretty_subset(subset)} - {metric_cfg['label']}"
            ax.set_title(title, fontsize=14, fontweight="bold")
            ax.set_xlabel("Requests per Second", fontsize=12)
            ax.set_ylabel(metric_cfg["label"], fontsize=12)
            ax.legend(framealpha=0.9)
            ax.grid(True, alpha=0.3)
            fig.tight_layout()

            outpath = args.output_dir / f"compare_{subset}_{metric_name}.png"
            fig.savefig(outpath, dpi=150)
            plt.close(fig)
            print(f"[INFO] Saved {outpath}")


if __name__ == "__main__":
    main()
