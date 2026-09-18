#!/usr/bin/env python3
"""Performance visualization CLI for speculative decoding benchmarks.

Subcommands:
    compare     Multi-version comparison plots (overlay smoothed curves)
    speedup     Pairwise speedup visualization (gradient-shaded region)
    acceptance  Long-context acceptance-length heatmap from a recorded table

Examples:
    python plot.py compare \\
        --source "No Spec=nospec/results.csv" \\
        --source "Eagle3=eagle3/results.csv" \\
        --metric latency --metric itl

    python plot.py speedup \\
        --baseline "No Spec=nospec/results.csv" \\
        --target "Eagle3=eagle3/results.csv" \\
        --metric latency --title "Qwen3-8B"

    python plot.py acceptance \\
        --table long_context/raw_table \\
        --output long_context/acceptance.png
"""

from __future__ import annotations

import argparse
import math
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.cm import ScalarMappable
from matplotlib.gridspec import GridSpec
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
# Acceptance (long-context spec-decode)
# ============================================================================

ACCEPT_ACCENT = "#1f4e79"  # mean-length line
ACCEPT_BAR = "#3b6fb0"  # tokens-generated bars
INK = "#1a1a1a"
MUTED = "#666666"
TOKENS_PER_K = 1000  # threshold for "k"-suffixed axis labels
MIN_CELL_SHARE = 0.005  # heatmap cells below this share are left blank


def _steps_from_records(records: list) -> tuple[np.ndarray, np.ndarray]:
    """Flatten spec records into (context position, committed length) per step.

    Each verify step is placed at the *running* context length at the moment it
    ran -- the prompt length plus everything committed by earlier steps in that
    same request -- and commits ``accepted drafts + 1`` (the bonus) tokens.
    """
    positions: list[int] = []
    accepts: list[int] = []
    for r in records:
        per_step = r.spec.get("per_step_accepted")
        if not per_step:
            continue
        pos = r.prompt_tokens
        for a in per_step:
            positions.append(pos)
            accepts.append(a + 1)
            pos += a + 1
    return np.array(positions), np.array(accepts)


def _auto_log2_edges(positions: np.ndarray) -> np.ndarray:
    """~2 bins per octave spanning the observed context range."""
    lo, hi = positions.min(), positions.max()
    n_bins = max(4, int(round(math.log2(hi / lo) * 2)))
    return np.logspace(math.log2(lo), math.log2(hi), n_bins + 1, base=2)


def _context_label(v: float) -> str:
    return f"{v / TOKENS_PER_K:.0f}k" if v >= TOKENS_PER_K else f"{v:.0f}"


def _acceptance_grid(
    positions: np.ndarray,
    accepts: np.ndarray,
    edges: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Bin steps by context length; return per-bin (frac, tokens, mean_len, ymax).

    ``frac`` is column-normalized: within each context bin, the share of verify
    steps at each acceptance length (columns sum to 1). ``tokens`` is committed
    tokens per bin (decoupled from acceptance -> a clean sample-size measure) and
    ``mean_len`` is committed tokens per step per bin.
    """
    n_bins = len(edges) - 1
    xidx = np.clip(np.digitize(positions, edges) - 1, 0, n_bins - 1)
    ymax = int(accepts.max())
    counts = np.zeros((ymax, n_bins))
    tokens = np.zeros(n_bins)
    for xi, a in zip(xidx, accepts, strict=True):
        counts[a - 1, xi] += 1
        tokens[xi] += a
    steps = counts.sum(axis=0)
    frac = np.divide(counts, steps, out=np.zeros_like(counts), where=steps > 0)
    mean_len = np.divide(tokens, steps, out=np.zeros_like(tokens), where=steps > 0)
    return frac, tokens, mean_len, ymax


def _annotate_cells(ax: plt.Axes, frac: np.ndarray, ymax: int, n_bins: int) -> None:
    vmax = frac.max()
    for yi in range(ymax):
        for xi in range(n_bins):
            v = frac[yi, xi]
            if v >= MIN_CELL_SHARE:
                ax.text(
                    xi,
                    yi,
                    f"{v * 100:.0f}",
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="white" if v > vmax * 0.55 else INK,
                )


def _draw_acceptance(
    frac: np.ndarray,
    tokens: np.ndarray,
    mean_len: np.ndarray,
    ymax: int,
    edges: np.ndarray,
    *,
    title: str | None,
) -> plt.Figure:
    """Three x-aligned panels sharing one context-length axis.

    A dedicated colorbar column (only under the heatmap) keeps all three left
    panels the same width, so the line, bars, and heatmap columns line up.
    """
    n_bins = len(edges) - 1
    fig = plt.figure(figsize=(11, 8.5))
    gs = GridSpec(
        3,
        2,
        width_ratios=[1, 0.025],
        height_ratios=[1.1, 1.1, 4],
        hspace=0.08,
        wspace=0.02,
    )

    # panel 1: mean acceptance length (summary of the heatmap below)
    ax_line = fig.add_subplot(gs[0, 0])
    ax_line.plot(
        range(n_bins), mean_len, "-o", color=ACCEPT_ACCENT, lw=2, ms=7, zorder=3
    )
    for i, v in enumerate(mean_len):
        if v:
            ax_line.text(
                i, v, f"{v:.2f}", ha="center", va="bottom", fontsize=8, color=MUTED
            )
    ax_line.set_ylabel("Mean\naccept len", color=INK)
    ax_line.grid(axis="y", color="#e6e6e6", lw=0.8)
    ax_line.set_axisbelow(True)
    ax_line.margins(y=0.28)
    ax_line.set_title(
        title or "Speculative-decoding acceptance vs context length",
        fontsize=13,
        fontweight="bold",
        color=INK,
        pad=10,
    )
    for s in ("top", "right"):
        ax_line.spines[s].set_visible(False)

    # panel 2: committed tokens generated per bin (how much data backs each column)
    ax_bar = fig.add_subplot(gs[1, 0], sharex=ax_line)
    ax_bar.bar(
        range(n_bins),
        tokens,
        width=1.0,
        color=ACCEPT_BAR,
        edgecolor="white",
        linewidth=1.2,
    )
    ax_bar.set_ylabel("Tokens\ngenerated", color=INK)
    ax_bar.margins(y=0.18)
    for s in ("top", "right"):
        ax_bar.spines[s].set_visible(False)

    # panel 3: acceptance-length distribution within each context bin
    ax = fig.add_subplot(gs[2, 0], sharex=ax_line)
    im = ax.imshow(
        frac, origin="lower", aspect="auto", cmap="Blues", vmin=0, vmax=frac.max()
    )
    _annotate_cells(ax, frac, ymax, n_bins)
    ax.set_yticks(range(ymax))
    ax.set_yticklabels(range(1, ymax + 1))
    ax.set_ylabel("Acceptance length (committed tokens/step)", color=INK)
    ax.set_xticks(range(n_bins))
    ax.set_xticklabels(
        [
            f"{_context_label(edges[i])}–{_context_label(edges[i + 1])}"
            for i in range(n_bins)
        ],
        fontsize=8,
    )
    ax.set_xlabel("Context length at token position (tokens)", color=INK)
    for shared in (ax_line, ax_bar):  # sharex re-adds labels; keep on heatmap only
        shared.tick_params(labelbottom=False)

    cax = fig.add_subplot(gs[2, 1])
    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label("Share of steps within context bin", color=INK)
    return fig


def run_acceptance(args: argparse.Namespace) -> None:
    from acceptance_report import load_spec_records  # noqa: PLC0415

    records = load_spec_records(args.table)
    if not records:
        print(f"[ERROR] No spec-decode records in {args.table}", file=sys.stderr)
        sys.exit(1)

    positions, accepts = _steps_from_records(records)
    if positions.size == 0:
        print(
            "[ERROR] No per-step data. Start the server with "
            "--per-request-spec-decode-metrics detailed",
            file=sys.stderr,
        )
        sys.exit(1)

    if args.context_bin_edges:
        edges = np.array(
            sorted(int(e) for e in args.context_bin_edges.split(",") if e.strip())
        )
    else:
        edges = _auto_log2_edges(positions)

    frac, tokens, mean_len, ymax = _acceptance_grid(positions, accepts, edges)
    fig = _draw_acceptance(frac, tokens, mean_len, ymax, edges, title=args.title)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)
    print(
        f"[INFO] Saved {args.output} "
        f"({len(accepts)} steps, {len(edges) - 1} context bins)"
    )


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

    # --- acceptance ---
    acc = sub.add_parser(
        "acceptance",
        help="Long-context spec-decode acceptance heatmap",
        description=(
            "Render a 3-panel figure from a recorded request table: mean "
            "acceptance length, tokens generated, and the acceptance-length "
            "distribution -- all sharing one context-length axis."
        ),
    )
    acc.add_argument(
        "--table",
        type=Path,
        required=True,
        help="raw_table directory produced by the long-context runner",
    )
    acc.add_argument(
        "--output",
        type=Path,
        default=Path("acceptance.png"),
        help="Output PNG path (default: acceptance.png)",
    )
    acc.add_argument(
        "--context-bin-edges",
        default=None,
        help="Comma-separated token edges (default: ~2 log2 bins per octave)",
    )
    acc.add_argument("--title", default=None, help="Optional plot title")
    acc.add_argument("--dpi", type=int, default=130, help="Output DPI (default: 130)")
    acc.set_defaults(func=run_acceptance)

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    args.func(args)


if __name__ == "__main__":
    main()
