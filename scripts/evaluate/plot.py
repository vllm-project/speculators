#!/usr/bin/env python3
"""Performance visualization CLI for speculative decoding benchmarks.

Subcommands:
    compare     Multi-version comparison plots (overlay smoothed curves)
    speedup     Pairwise speedup visualization (gradient-shaded region)

Examples:
    python plot.py compare \\
        --source "No Spec=nospec/results.csv" \\
        --source "Eagle3=eagle3/results.csv" \\
        --metric latency --metric itl

    python plot.py speedup \\
        --baseline "No Spec=nospec/results.csv" \\
        --target "Eagle3=eagle3/results.csv" \\
        --metric latency --title "Qwen3-8B"
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

COLOR_CYCLE = [
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
    "#17becf",
]


# ============================================================================
# Compare
# ============================================================================


def _collect_all_data(
    sources: dict[str, list[Path]],
    metric_name: str,
) -> dict[str, dict[str, list[tuple[float, float]]]]:
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
    return all_data


def _plot_compare_subset(
    ax: plt.Axes,
    subset: str,
    all_data: dict[str, dict[str, list[tuple[float, float]]]],
    source_labels: list[str],
) -> None:
    for i, label in enumerate(source_labels):
        points = all_data[label].get(subset, [])
        if not points:
            continue

        color = COLOR_CYCLE[i % len(COLOR_CYCLE)]
        points.sort(key=lambda p: p[0])
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]

        ax.scatter(xs, ys, color=color, alpha=0.35, s=25, zorder=3)

        x_smooth, y_smooth = smooth_curve(xs, ys)
        ax.plot(x_smooth, y_smooth, color=color, linewidth=2.5, label=label, zorder=4)


def run_compare(args: argparse.Namespace) -> None:
    metrics = args.metric or ["latency"]
    subset_filter = set(args.subsets.split(",")) if args.subsets else None

    try:
        sources = parse_source_args(args.source)
    except (ValueError, FileNotFoundError) as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        sys.exit(1)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    source_labels = list(sources.keys())

    for metric_name in metrics:
        metric_cfg = METRICS[metric_name]
        all_data = _collect_all_data(sources, metric_name)

        all_subsets: set[str] = set()
        for label_data in all_data.values():
            all_subsets.update(label_data.keys())
        if subset_filter:
            all_subsets &= subset_filter
        if not all_subsets:
            print(f"[WARN] No data found for metric '{metric_name}'", file=sys.stderr)
            continue

        combined: dict[str, list[tuple[float, float]]] = defaultdict(list)
        for label in source_labels:
            for subset in sorted(all_subsets):
                combined[label].extend(all_data[label].get(subset, []))

        combined_data = {
            label: {"__combined__": pts} for label, pts in combined.items()
        }

        fig, ax = plt.subplots(figsize=(8, 5))
        _plot_compare_subset(
            ax,
            "__combined__",
            combined_data,
            source_labels,
        )

        ax.set_title(metric_cfg["label"], fontsize=14, fontweight="bold")
        ax.set_xlabel("Requests per Second", fontsize=12)
        ax.set_ylabel(metric_cfg["label"], fontsize=12)
        ax.legend(framealpha=0.9)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()

        outpath = args.output_dir / f"compare_{metric_name}.png"
        fig.savefig(outpath, dpi=150)
        plt.close(fig)
        print(f"[INFO] Saved {outpath}")


# ============================================================================
# Speedup
# ============================================================================


def _collect_points(
    source_args: list[str],
    metric_name: str,
) -> tuple[str, dict[str, list[tuple[float, float]]]]:
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


def _build_colormap(
    speedup: np.ndarray,
) -> tuple[mcolors.Colormap, mcolors.Normalize]:
    sp_min, sp_max = float(speedup.min()), float(speedup.max())
    full_cmap = plt.get_cmap("bwr_r")

    if sp_min < 1.0 < sp_max:
        norm = mcolors.TwoSlopeNorm(vcenter=1.0, vmin=sp_min, vmax=sp_max)
        cmap = full_cmap
    elif sp_min >= 1.0:
        cmap = mcolors.LinearSegmentedColormap.from_list(
            "bwr_r_upper",
            full_cmap(np.linspace(0.5, 1.0, 256)),
        )
        norm = mcolors.Normalize(vmin=1.0, vmax=max(sp_max, 1.01))
    else:
        cmap = mcolors.LinearSegmentedColormap.from_list(
            "bwr_r_lower",
            full_cmap(np.linspace(0.0, 0.5, 256)),
        )
        norm = mcolors.Normalize(vmin=min(sp_min, 0.99), vmax=1.0)
    return cmap, norm


def _compute_speedup_curves(
    b_pts: list[tuple[float, float]],
    t_pts: list[tuple[float, float]],
    *,
    increasing: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
    b_x_smooth, b_y_smooth = smooth_curve(
        [p[0] for p in b_pts],
        [p[1] for p in b_pts],
    )
    t_x_smooth, t_y_smooth = smooth_curve(
        [p[0] for p in t_pts],
        [p[1] for p in t_pts],
    )

    x_lo = max(b_x_smooth.min(), t_x_smooth.min())
    x_hi = min(b_x_smooth.max(), t_x_smooth.max())
    if x_lo >= x_hi:
        return None

    x_dense = np.linspace(x_lo, x_hi, 200)
    y_baseline = np.interp(x_dense, b_x_smooth, b_y_smooth)
    y_target = np.interp(x_dense, t_x_smooth, t_y_smooth)

    eps = 1e-12
    if increasing:
        speedup = y_baseline / np.maximum(y_target, eps)
    else:
        speedup = y_target / np.maximum(y_baseline, eps)

    return x_dense, y_baseline, y_target, speedup


def _draw_shaded_region(
    ax: plt.Axes,
    x_dense: np.ndarray,
    y_baseline: np.ndarray,
    y_target: np.ndarray,
    speedup: np.ndarray,
    cmap: mcolors.Colormap,
    norm: mcolors.Normalize,
) -> None:
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


def _plot_speedup_subset(
    fig: plt.Figure,
    ax: plt.Axes,
    subset: str,
    b_pts: list[tuple[float, float]],
    t_pts: list[tuple[float, float]],
    baseline_label: str,
    target_label: str,
    metric_cfg: dict,
    *,
    increasing: bool,
    title_prefix: str | None,
) -> bool:
    result = _compute_speedup_curves(b_pts, t_pts, increasing=increasing)
    if result is None:
        return False

    x_dense, y_baseline, y_target, speedup = result

    bx = [p[0] for p in b_pts]
    by = [p[1] for p in b_pts]
    tx = [p[0] for p in t_pts]
    ty = [p[1] for p in t_pts]
    ax.scatter(bx, by, color="black", alpha=0.35, s=25, zorder=3)
    ax.scatter(tx, ty, color="green", alpha=0.35, s=25, zorder=3)

    cmap, norm = _build_colormap(speedup)
    _draw_shaded_region(ax, x_dense, y_baseline, y_target, speedup, cmap, norm)

    ax.plot(
        x_dense,
        y_baseline,
        color="black",
        linewidth=2,
        label=baseline_label,
        zorder=4,
    )
    ax.plot(x_dense, y_target, color="green", linewidth=2, label=target_label, zorder=4)

    sm = ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, pad=0.02)
    cbar.set_label("Speedup", fontsize=11)

    title_parts = []
    if title_prefix:
        title_parts.append(title_prefix)
    title_parts.append(pretty_subset(subset))
    ax.set_title(", ".join(title_parts), fontsize=14, fontweight="bold")
    ax.set_xlabel("Requests per second (RPS)", fontsize=12)
    ax.set_ylabel(metric_cfg["label"], fontsize=12)
    ax.legend(framealpha=0.9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return True


def run_speedup(args: argparse.Namespace) -> None:
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
            print(
                f"[WARN] No common subsets for metric '{metric_name}'",
                file=sys.stderr,
            )
            continue

        for subset in sorted(all_subsets):
            b_pts = sorted(baseline_data[subset], key=lambda p: p[0])
            t_pts = sorted(target_data[subset], key=lambda p: p[0])

            fig, ax = plt.subplots(figsize=(8, 5))
            ok = _plot_speedup_subset(
                fig,
                ax,
                subset,
                b_pts,
                t_pts,
                baseline_label,
                target_label,
                metric_cfg,
                increasing=increasing,
                title_prefix=args.title,
            )
            if not ok:
                print(
                    f"[WARN] No overlapping RPS range for subset '{subset}', skipping",
                    file=sys.stderr,
                )
                plt.close(fig)
                continue

            outpath = args.output_dir / f"speedup_{subset}_{metric_name}.png"
            fig.savefig(outpath, dpi=150)
            plt.close(fig)
            print(f"[INFO] Saved {outpath}")


# ============================================================================
# CLI
# ============================================================================


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="plot",
        description="Performance visualization for speculative decoding benchmarks.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "examples:\n"
            '  python plot.py compare --source "No Spec=nospec/results.csv" \\\n'
            '      --source "Eagle3=eagle3/results.csv" --metric latency\n\n'
            '  python plot.py speedup --baseline "No Spec=nospec/results.csv" \\\n'
            '      --target "Eagle3=eagle3/results.csv" --metric latency\n'
        ),
    )
    sub = parser.add_subparsers(dest="command", title="commands")

    # --- compare ---
    cmp = sub.add_parser(
        "compare",
        help="Multi-version performance comparison plots",
        description=(
            "Overlay smoothed performance curves for multiple model versions "
            "on the same axes. Produces one PNG per (subset, metric) pair."
        ),
    )
    cmp.add_argument(
        "--source",
        action="append",
        required=True,
        metavar="LABEL=PATH",
        help="Version data as 'Label=path'. Repeatable; same label pools repetitions.",
    )
    cmp.add_argument(
        "--metric",
        action="append",
        choices=list(METRICS.keys()),
        metavar="METRIC",
        help=f"Metric(s) to plot (default: latency). Choices: {', '.join(METRICS)}",
    )
    cmp.add_argument(
        "--output-dir",
        type=Path,
        default=Path(),
        help="Directory for output PNGs (default: current directory)",
    )
    cmp.add_argument(
        "--subsets",
        type=str,
        default=None,
        help="Comma-separated subset filter (default: all found in data)",
    )
    cmp.set_defaults(func=run_compare)

    # --- speedup ---
    spd = sub.add_parser(
        "speedup",
        help="Pairwise speedup visualization with gradient shading",
        description=(
            "Compare baseline and target versions with gradient-shaded region. "
            "Blue = faster, red = regression."
        ),
    )
    spd.add_argument(
        "--baseline",
        action="append",
        required=True,
        metavar="LABEL=PATH",
        help="Baseline version as 'Label=path'. Repeatable for pooling reps.",
    )
    spd.add_argument(
        "--target",
        action="append",
        required=True,
        metavar="LABEL=PATH",
        help="Target version as 'Label=path'. Repeatable for pooling reps.",
    )
    spd.add_argument(
        "--metric",
        action="append",
        choices=list(METRICS.keys()),
        metavar="METRIC",
        help=f"Metric(s) to plot (default: latency). Choices: {', '.join(METRICS)}",
    )
    spd.add_argument(
        "--output-dir",
        type=Path,
        default=Path(),
        help="Directory for output PNGs (default: current directory)",
    )
    spd.add_argument(
        "--subsets",
        type=str,
        default=None,
        help="Comma-separated subset filter (default: all found in data)",
    )
    spd.add_argument(
        "--title",
        type=str,
        default=None,
        help="Optional title prefix for plots (e.g. model name)",
    )
    spd.set_defaults(func=run_speedup)

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    args.func(args)


if __name__ == "__main__":
    main()
