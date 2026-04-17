#!/usr/bin/env python3
"""Pairwise speedup visualization with gradient-shaded region.

Compares a baseline and target version, plotting both smoothed curves
with a shaded region between them colored by local speedup using the
bwr colormap (blue = faster, red = regression).

Usage:
    python plot_perf_speedup.py \\
        --baseline "No Spec=nospec/results.csv" \\
        --target "DFlash=dflash/results.csv" \\
        --metric latency \\
        --output-dir ./plots
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.cm import ScalarMappable

from perf_utils import (
    METRICS,
    load_data,
    parse_source_args,
    pretty_subset,
    smooth_curve,
)


def _collect_points(
    source_args: list[str], metric_name: str
) -> tuple[str, dict[str, list[tuple[float, float]]]]:
    """Parse source args and collect all points, returning (label, {subset: points})."""
    sources = parse_source_args(source_args)
    if len(sources) != 1:
        raise ValueError(
            f"Expected exactly one label, got {len(sources)}: {list(sources.keys())}"
        )
    label = next(iter(sources))
    combined: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for path in sources[label]:
        file_data = load_data(path, metric_name)
        for subset, points in file_data.items():
            combined[subset].extend(points)
    return label, dict(combined)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pairwise speedup visualization with gradient shading."
    )
    parser.add_argument(
        "--baseline",
        action="append",
        required=True,
        metavar="LABEL=PATH",
        help="Baseline version as 'Label=path'. Repeatable for pooling reps (same label).",
    )
    parser.add_argument(
        "--target",
        action="append",
        required=True,
        metavar="LABEL=PATH",
        help="Target version as 'Label=path'. Repeatable for pooling reps (same label).",
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
    parser.add_argument(
        "--title",
        type=str,
        default=None,
        help="Optional title prefix for plots (e.g. model name)",
    )
    args = parser.parse_args()

    metrics = args.metric or ["latency"]
    subset_filter = set(args.subsets.split(",")) if args.subsets else None
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for metric_name in metrics:
        metric_cfg = METRICS[metric_name]
        increasing = metric_cfg["increasing"]

        try:
            baseline_label, baseline_data = _collect_points(args.baseline, metric_name)
            target_label, target_data = _collect_points(args.target, metric_name)
        except (ValueError, FileNotFoundError) as e:
            print(f"[ERROR] {e}", file=sys.stderr)
            sys.exit(1)

        all_subsets = set(baseline_data.keys()) & set(target_data.keys())
        if subset_filter:
            all_subsets &= subset_filter

        if not all_subsets:
            print(f"[WARN] No common subsets for metric '{metric_name}'", file=sys.stderr)
            continue

        for subset in sorted(all_subsets):
            b_pts = sorted(baseline_data[subset], key=lambda p: p[0])
            t_pts = sorted(target_data[subset], key=lambda p: p[0])

            b_xs, b_ys = [p[0] for p in b_pts], [p[1] for p in b_pts]
            t_xs, t_ys = [p[0] for p in t_pts], [p[1] for p in t_pts]

            b_x_smooth, b_y_smooth = smooth_curve(b_xs, b_ys, increasing=increasing)
            t_x_smooth, t_y_smooth = smooth_curve(t_xs, t_ys, increasing=increasing)

            # Compute shared X range (overlap of both smoothed curves)
            x_lo = max(b_x_smooth.min(), t_x_smooth.min())
            x_hi = min(b_x_smooth.max(), t_x_smooth.max())

            if x_lo >= x_hi:
                print(
                    f"[WARN] No overlapping RPS range for subset '{subset}', skipping",
                    file=sys.stderr,
                )
                continue

            # Interpolate both onto common dense grid
            from scipy.interpolate import PchipInterpolator

            b_interp = PchipInterpolator(b_x_smooth, b_y_smooth)
            t_interp = PchipInterpolator(t_x_smooth, t_y_smooth)

            x_dense = np.linspace(x_lo, x_hi, 200)
            y_baseline = b_interp(x_dense)
            y_target = t_interp(x_dense)

            # Speedup: >1 means target is better
            eps = 1e-12
            if increasing:
                speedup = y_baseline / np.maximum(y_target, eps)
            else:
                speedup = y_target / np.maximum(y_baseline, eps)

            # --- Plot ---
            fig, ax = plt.subplots(figsize=(8, 5))

            # Scatter raw data
            ax.scatter(b_xs, b_ys, color="black", alpha=0.35, s=25, zorder=3)
            ax.scatter(t_xs, t_ys, color="green", alpha=0.35, s=25, zorder=3)

            # Shaded region with per-strip coloring.
            # Use bwr_r throughout: blue = speedup > 1, red = speedup < 1.
            # Truncate the colormap to the relevant half when data is one-sided
            # so the colorbar only shows colors that actually appear.
            sp_min, sp_max = float(speedup.min()), float(speedup.max())
            full_cmap = plt.get_cmap("bwr_r")

            if sp_min < 1.0 < sp_max:
                norm = mcolors.TwoSlopeNorm(vcenter=1.0, vmin=sp_min, vmax=sp_max)
                cmap = full_cmap
            elif sp_min >= 1.0:
                # All speedup >= 1: use upper half of bwr_r (white -> blue)
                cmap = mcolors.LinearSegmentedColormap.from_list(
                    "bwr_r_upper", full_cmap(np.linspace(0.5, 1.0, 256))
                )
                norm = mcolors.Normalize(vmin=1.0, vmax=max(sp_max, 1.01))
            else:
                # All speedup <= 1: use lower half of bwr_r (red -> white)
                cmap = mcolors.LinearSegmentedColormap.from_list(
                    "bwr_r_lower", full_cmap(np.linspace(0.0, 0.5, 256))
                )
                norm = mcolors.Normalize(vmin=min(sp_min, 0.99), vmax=1.0)

            for i in range(len(x_dense) - 1):
                color = cmap(norm(speedup[i]))
                ax.fill_between(
                    x_dense[i : i + 2],
                    y_baseline[i : i + 2],
                    y_target[i : i + 2],
                    color=color,
                    alpha=0.7,
                    edgecolor="none",
                )

            # Smooth curves on top
            ax.plot(
                x_dense, y_baseline,
                color="black", linewidth=2, label=baseline_label, zorder=4,
            )
            ax.plot(
                x_dense, y_target,
                color="green", linewidth=2, label=target_label, zorder=4,
            )

            # Colorbar
            sm = ScalarMappable(cmap=cmap, norm=norm)
            sm.set_array([])
            cbar = fig.colorbar(sm, ax=ax, pad=0.02)
            cbar.set_label("Speedup", fontsize=11)

            # Labels
            title_parts = []
            if args.title:
                title_parts.append(args.title)
            title_parts.append(pretty_subset(subset))
            ax.set_title(", ".join(title_parts), fontsize=14, fontweight="bold")
            ax.set_xlabel("Requests per second (RPS)", fontsize=12)
            ax.set_ylabel(metric_cfg["label"], fontsize=12)
            ax.legend(framealpha=0.9)
            ax.grid(True, alpha=0.3)
            fig.tight_layout()

            outpath = args.output_dir / f"speedup_{subset}_{metric_name}.png"
            fig.savefig(outpath, dpi=150)
            plt.close(fig)
            print(f"[INFO] Saved {outpath}")


if __name__ == "__main__":
    main()
