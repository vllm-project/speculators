#!/usr/bin/env python3
"""Performance visualization CLI for speculative decoding benchmarks.

Subcommands:
    compare     Multi-version comparison plots (overlay smoothed curves)
    speedup     Pairwise speedup visualization (gradient-shaded region)
    mrcr        MRCR long-context acceptance length, by request start length
                and by token position (output of `evaluate.py long-context`)

Examples:
    python plot.py compare \\
        --source "No Spec=nospec/results.csv" \\
        --source "Eagle3=eagle3/results.csv" \\
        --metric latency --metric itl

    python plot.py speedup \\
        --baseline "No Spec=nospec/results.csv" \\
        --target "Eagle3=eagle3/results.csv" \\
        --metric latency --title "Qwen3-8B"

    python plot.py mrcr --results-dir Qwen3-8B_20260904_143700
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.cm import ScalarMappable
from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter
from perf_utils import (
    METRICS,
    load_data,
    parse_source_args,
    pretty_subset,
    smooth_curve,
)
from spec_acceptance import RAW_LOG_FILENAME, load_results

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
# MRCR long-context acceptance
# ============================================================================

# Two-hue small-multiples pair (validated for adjacent CVD separation):
# request-start-length panel in blue, token-position panel in orange.
_MRCR_START_COLOR = "#2a78d6"
_MRCR_POSITION_COLOR = "#eb6834"


def _read_bucket_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


_MAX_LABELED_BARS = 40


def _plot_acceptance_bars(ax, rows: list[dict], color: str, title: str) -> None:
    labels = [row["bucket"] for row in rows]
    values = [float(row["mean_acceptance_length"]) for row in rows]
    x = range(len(labels))
    dense = len(labels) > _MAX_LABELED_BARS

    bars = ax.bar(x, values, color=color, width=1.0 if dense else 0.6, zorder=3)
    if not dense:
        for bar, value in zip(bars, values, strict=True):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{value:.2f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    # Too many buckets to label every tick -- thin them out instead of
    # letting them overlap into an unreadable smear.
    tick_step = max(1, -(-len(labels) // 24)) if dense else 1
    tick_idx = list(x)[::tick_step]
    ax.set_xticks(tick_idx)
    ax.set_xticklabels(
        [labels[i] for i in tick_idx],
        rotation=35 if not dense else 90,
        ha="right" if not dense else "center",
        fontsize=9 if not dense else 7,
    )
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlabel("Context length (tokens)", fontsize=11)
    ax.set_ylabel("Mean acceptance length", fontsize=11)
    ax.grid(True, axis="y", alpha=0.3, zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _load_position_scatter(
    raw_log: Path, bin_size: int
) -> tuple[list[int], list[int], list[int], list[float], list[float]]:
    """Reconstruct per-verify-step (position, acceptance length) data from a raw log.

    Returns ``(bubble_x, bubble_y, bubble_count, line_x, line_y)``: the bubble
    arrays aggregate steps into ``(position // bin_size, acceptance_length)``
    counts (one disk per combination); the line arrays are the raw, unbinned
    per-step pairs used for the mean trend line.
    """
    counter: Counter[tuple[int, int]] = Counter()
    line_x: list[float] = []
    line_y: list[float] = []
    for result in load_results(raw_log):
        accepted = result.metrics.get("per_step_accepted")
        drafted = result.metrics.get("per_step_drafted")
        if not accepted or not drafted:
            continue
        pos = result.prompt_tokens
        for accepted_count, _ in zip(accepted, drafted, strict=True):
            acceptance_length = accepted_count + 1
            line_x.append(pos)
            line_y.append(acceptance_length)
            binned_pos = (pos // bin_size) * bin_size
            counter[(binned_pos, acceptance_length)] += 1
            pos += accepted_count + 1

    bubble_x = [key[0] for key in counter]
    bubble_y = [key[1] for key in counter]
    bubble_count = [counter[key] for key in counter]
    return bubble_x, bubble_y, bubble_count, line_x, line_y


def _binned_mean_curve(
    x: list[float], y: list[float], n_bins: int = 40
) -> tuple[np.ndarray, np.ndarray] | None:
    """Plain per-bin mean of y over log-spaced bins of x. No model, no error bars."""
    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    if len(x_arr) < 10:  # noqa: PLR2004
        return None

    log_x = np.log(x_arr)
    edges = np.linspace(log_x.min(), log_x.max(), n_bins + 1)
    bin_idx = np.clip(np.digitize(log_x, edges) - 1, 0, n_bins - 1)

    centers, means = [], []
    for b in range(n_bins):
        mask = bin_idx == b
        if not mask.any():
            continue
        centers.append(np.exp(log_x[mask].mean()))
        means.append(y_arr[mask].mean())
    return np.array(centers), np.array(means)


def _set_log2_ticks(ax) -> None:
    """Label the (already log-scale) x-axis with powers of 2, e.g. "2^12" for 4096.

    Context lengths are naturally thought of in powers of 2 (MRCR's own
    bucket edges double each step); matplotlib's default log-scale ticks
    (4x10^3, 10^4, ...) don't line up with that.
    """
    xmin, xmax = ax.get_xlim()
    if xmin <= 0:
        return
    lo_exp = int(np.floor(np.log2(xmin)))
    hi_exp = int(np.ceil(np.log2(xmax)))
    ticks = [2**e for e in range(lo_exp, hi_exp + 1) if xmin <= 2**e <= xmax]
    if not ticks:
        return
    ax.xaxis.set_major_locator(FixedLocator(ticks))
    ax.xaxis.set_major_formatter(
        FuncFormatter(lambda val, _pos: f"2^{round(np.log2(val))}")
    )
    ax.xaxis.set_minor_formatter(NullFormatter())


_BUBBLE_JITTER_SEED = 0


def _plot_position_scatter(
    ax,
    raw_log: Path,
    *,
    bin_size: int,
    alpha: float,
    min_area: float,
    max_area: float,
    y_jitter: float,
    title: str,
) -> None:
    """Disk-per-(position, acceptance length) scatter, sized by step count.

    Disk *area* -- not radius -- scales with count, so the size comparison
    a reader makes (area) matches the encoded quantity. Bubbles at the same
    integer acceptance length would otherwise sit on an exact horizontal
    line and stack directly on top of each other; a small fixed-seed random
    y-jitter (uniform within +/- `y_jitter`) spreads them out so overlapping
    disks are still visually distinguishable. The trend line below is fit
    on the un-jittered values, so jitter never affects the reported mean.
    """
    bubble_x, bubble_y, bubble_count, line_x, line_y = _load_position_scatter(
        raw_log, bin_size
    )
    if not bubble_x:
        ax.axis("off")
        return

    rng = np.random.default_rng(_BUBBLE_JITTER_SEED)
    jittered_y = [
        y + rng.uniform(-y_jitter, y_jitter) if y_jitter else y for y in bubble_y
    ]

    max_count = max(bubble_count)
    sizes = [min_area + (max_area - min_area) * (c / max_count) for c in bubble_count]
    ax.scatter(
        bubble_x,
        jittered_y,
        s=sizes,
        color=_MRCR_POSITION_COLOR,
        alpha=alpha,
        edgecolors="none",
        zorder=3,
    )

    trend_result = _binned_mean_curve(line_x, line_y)
    if trend_result is not None:
        x_smooth, mean_pred = trend_result
        ax.plot(
            x_smooth,
            mean_pred,
            color="#0d366b",
            linewidth=2.5,
            zorder=4,
            label="Mean acceptance length",
        )
        ax.legend(loc="upper right", framealpha=0.9, fontsize=9)

    ax.set_xscale("log")
    _set_log2_ticks(ax)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlabel("Context length (tokens, token position)", fontsize=11)
    ax.set_ylabel("Acceptance length", fontsize=11)
    ax.grid(True, alpha=0.3, zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def run_mrcr(args: argparse.Namespace) -> None:
    start_rows = _read_bucket_csv(args.results_dir / "acceptance_by_start_length.csv")
    raw_log = args.raw_log or (args.results_dir / RAW_LOG_FILENAME)
    has_raw_log = raw_log.exists()
    position_rows = (
        []
        if has_raw_log
        else _read_bucket_csv(args.results_dir / "acceptance_by_position.csv")
    )

    if not start_rows and not has_raw_log and not position_rows:
        print(
            f"[ERROR] No acceptance data found in {args.results_dir}", file=sys.stderr
        )
        sys.exit(1)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    if start_rows:
        _plot_acceptance_bars(
            axes[0], start_rows, _MRCR_START_COLOR, "By context length at request start"
        )
    else:
        axes[0].axis("off")

    if has_raw_log:
        _plot_position_scatter(
            axes[1],
            raw_log,
            bin_size=args.position_bin_size,
            alpha=args.bubble_alpha,
            min_area=args.bubble_min_area,
            max_area=args.bubble_max_area,
            y_jitter=args.bubble_y_jitter,
            title="By context length at token position",
        )
    elif position_rows:
        _plot_acceptance_bars(
            axes[1],
            position_rows,
            _MRCR_POSITION_COLOR,
            "By context length at token position",
        )
    else:
        axes[1].axis("off")

    fig.suptitle(
        args.title or "MRCR long-context acceptance length",
        fontsize=14,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))

    output_dir = args.output_dir or args.results_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    outpath = output_dir / "mrcr_acceptance_length.png"
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

    # --- mrcr ---
    mrcr = sub.add_parser(
        "mrcr",
        help="MRCR long-context acceptance length bar charts",
        description=(
            "Bar charts of mean acceptance length by context length, both at "
            "request start and at token position (output of "
            "`evaluate.py long-context`)."
        ),
    )
    mrcr.add_argument(
        "--results-dir",
        type=Path,
        required=True,
        help=(
            "Directory containing acceptance_by_start_length.csv and "
            "acceptance_by_position.csv"
        ),
    )
    mrcr.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for the output PNG (default: --results-dir)",
    )
    mrcr.add_argument(
        "--title",
        type=str,
        default=None,
        help="Optional figure title (e.g. model name)",
    )
    mrcr.add_argument(
        "--raw-log",
        type=Path,
        default=None,
        help=(
            "Path to raw_requests.jsonl for the per-token-position disk scatter "
            "(default: <results-dir>/raw_requests.jsonl; falls back to the "
            "coarser acceptance_by_position.csv bar chart if not found)"
        ),
    )
    mrcr.add_argument(
        "--position-bin-size",
        type=int,
        default=32,
        help=(
            "Token width to group positions into for the scatter (default: 32). "
            "At 1 (exact positions), the model's frequent 1-token-per-step "
            "advances in the low-acceptance regime mean adjacent same-row "
            "disks touch and merge into solid bands regardless of opacity; "
            "coarser bins give real, separable count variation."
        ),
    )
    mrcr.add_argument(
        "--bubble-alpha",
        type=float,
        default=0.12,
        help="Disk opacity in the per-token-position scatter (default: 0.12)",
    )
    mrcr.add_argument(
        "--bubble-y-jitter",
        type=float,
        default=0.25,
        help=(
            "Random +/- jitter added to each disk's y-position (acceptance "
            "length) so disks at the same integer value don't stack exactly "
            "on top of each other (default: 0.25; 0 disables)"
        ),
    )
    mrcr.add_argument(
        "--bubble-min-area",
        type=float,
        default=10.0,
        help="Disk area (points^2) for a count-1 cell (default: 10)",
    )
    mrcr.add_argument(
        "--bubble-max-area",
        type=float,
        default=280.0,
        help="Disk area (points^2) for the most frequent cell (default: 280)",
    )
    mrcr.set_defaults(func=run_mrcr)

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    args.func(args)


if __name__ == "__main__":
    main()
