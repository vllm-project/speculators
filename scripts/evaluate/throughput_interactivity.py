#!/usr/bin/env python3
"""Plot throughput versus interactivity from timed GuideLLM or legacy vLLM results."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

from speedup import read_archive, summarize_benchmark


def interactivity(result: dict) -> float:
    """Use vLLM's TPOT formula; tokenizer-free runs omit mean_tpot_ms."""
    tpots = [
        (latency - ttft) / (tokens - 1)
        for latency, ttft, tokens in zip(
            result["latencies"], result["ttfts"], result["output_lens"], strict=True
        )
        if tokens > 1
    ]
    return 1 / statistics.fmean(tpots)


def aggregate_points(results: list[dict]) -> list[dict]:
    """Average each run's interactivity and throughput at the same concurrency."""
    methods = {
        "guidellm_window" if "benchmarks" in r else "vllm_batch" for r in results
    }
    if len(methods) > 1:
        raise ValueError("Cannot mix timed-window and batch measurements")
    groups = defaultdict(list)
    for result in results:
        if "benchmarks" in result:
            point = summarize_benchmark(result)
            groups[point["streams"]].append(point)
            continue
        if result["completed"] != result["num_prompts"]:
            raise ValueError("Cannot plot a benchmark with failed requests")
        groups[result["max_concurrency"]].append(
            {"x": interactivity(result), "y": result["output_throughput"]}
        )
    return [
        {
            "streams": streams,
            "x": statistics.fmean(r["x"] for r in runs),
            "y": statistics.fmean(r["y"] for r in runs),
        }
        for streams, runs in sorted(groups.items())
    ]


PALETTE = ["#b52513", "#2c6e9b", "#7a8c3f", "#b8b1a8", "#8a5a9e"]
BG, GRID, INK, MUTED = "#f2f2f2", "#cfdae2", "#181818", "#666666"
YLABEL = "Output Token Throughput (tok/s)"
XLABEL = "Interactivity (tok/s/user, 1 / mean time per output token)"
# label slots tried around a marker, in points; nearest first
LABEL_OFFSETS = [
    (8, 6),
    (8, -16),
    (-8, 6),
    (-8, -16),
    (8, 22),
    (-8, 22),
    (8, -32),
    (-8, -32),
]
MARKER_PX = 8  # markers closer than this sit on top of each other


def _archive_reports(path: Path) -> list[dict]:
    header, records, _ = read_archive(path)
    valid = []
    for record in records.values():
        if record.get("status", "ok") != "ok" or "report" not in record:
            print(
                f"Warning: excluding {record['point']} "
                f"({record.get('status', 'missing report')})",
                file=sys.stderr,
            )
            continue
        valid.append(record)
    expected = header.get("config", {}).get("repeats")
    counts: dict[str, int] = defaultdict(int)
    for record in valid:
        counts[record["point"].split("_r")[0]] += 1
    results = []
    for record in valid:
        if expected is not None and counts[record["point"].split("_r")[0]] != expected:
            print(
                f"Warning: excluding partial concurrency at {record['point']}",
                file=sys.stderr,
            )
            continue
        results.append(record["report"])
    return results


def _load_series(args: argparse.Namespace) -> list[dict]:
    series: list[dict] = []
    methods: set[str] = set()
    for spec in args.series:
        # DIRECTORY[:NAME[:#COLOR]]: the color is a trailing `:#...` field, so the name
        # may contain colons ("B200: no speculator").
        path_text, _, name = spec.partition(":")
        color = ""
        head, sep, tail = name.rpartition(":")
        if sep and tail.startswith("#"):
            name, color = head, tail
        path = Path(path_text)
        result_dir = path / "artifacts" if (path / "artifacts").is_dir() else path
        archive_path = path / "benchmarks.jsonl"
        if archive_path.exists():
            results = _archive_reports(archive_path)
        else:
            results = [
                json.loads(p.read_text())
                for p in sorted(result_dir.glob("conc*_r*.json"))
                if not p.name.endswith(".command.json")
            ]
        methods.update(
            "guidellm_window" if "benchmarks" in r else "vllm_batch" for r in results
        )
        if len(methods) > 1:
            raise ValueError(
                "Cannot compare timed-window and batch measurements on one chart"
            )
        points = aggregate_points(results)
        if not points:
            raise SystemExit(f"no benchmark results in {path}")
        series.append(
            {
                "name": name or results[0]["label"],
                "color": color or PALETTE[len(series) % len(PALETTE)],
                "points": points,  # in N order, from aggregate_points
            }
        )
    return series


def _label_points(ax: Any, points: list[dict], taken: list[Any]) -> None:
    """Write `N=...` next to each marker, in the first slot that overlaps nothing.

    Markers drawn on top of each other (a saturated server puts N = 32, 64 and
    128 on one spot) share one label. `taken` holds the boxes already used on the
    canvas, markers included, and grows with every label placed.
    """
    from matplotlib.transforms import Bbox  # noqa: PLC0415

    clusters: list[tuple[list[dict], tuple[float, float]]] = []
    for p in points:
        px, py = ax.transData.transform((p["x"], p["y"]))
        if clusters and all(
            abs(v - w) < MARKER_PX
            for v, w in zip((px, py), clusters[-1][1], strict=True)
        ):
            clusters[-1][0].append(p)
        else:
            clusters.append(([p], (px, py)))
    for _, (px, py) in clusters:
        taken.append(Bbox.from_extents(px - 7, py - 7, px + 7, py + 7))
    for cluster, _ in clusters:
        first, last = cluster[0]["streams"], cluster[-1]["streams"]
        label = f"N={first:.0f}" if len(cluster) == 1 else f"N={first:.0f}..{last:.0f}"
        for dx, dy in LABEL_OFFSETS:
            text = ax.annotate(
                label,
                (cluster[0]["x"], cluster[0]["y"]),
                xytext=(dx, dy),
                textcoords="offset points",
                ha="left" if dx > 0 else "right",
                fontsize=9,
                fontweight="bold",
                color=INK,
            )
            box = text.get_window_extent()
            inside = ax.bbox.contains(box.x0, box.y0) and ax.bbox.contains(
                box.x1, box.y1
            )
            if inside and not any(box.overlaps(other) for other in taken):
                taken.append(box)
                break
            text.remove()


def plot(args: argparse.Namespace) -> int:
    import matplotlib.pyplot as plt  # noqa: PLC0415

    series = _load_series(args)
    every = [p for s in series for p in s["points"]]
    fig, ax = plt.subplots(figsize=(16, 9), dpi=135, facecolor=BG)
    ax.set_facecolor(BG)
    ax.set_xlim(0, 1.08 * max(p["x"] for p in every))
    ax.set_ylim(0, 1.08 * max(p["y"] for p in every))
    ax.grid(color=GRID, linewidth=1)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.yaxis.set_major_formatter(
        lambda value, _: f"{value / 1000:g}k" if value >= 1000 else f"{value:g}"  # noqa: PLR2004
    )
    ax.tick_params(colors=MUTED, labelsize=11)
    for s in series:
        xs, ys = [p["x"] for p in s["points"]], [p["y"] for p in s["points"]]
        ax.plot(
            xs,
            ys,
            color=s["color"],
            linewidth=3,
            marker="o",
            markersize=7,
            label=s["name"],
        )
    ax.set_xlabel(XLABEL, fontsize=12, color=INK)
    ax.set_ylabel(YLABEL, fontsize=12, color=INK)
    fig.suptitle(
        args.title, x=0.05, ha="left", fontsize=22, fontweight="bold", color=INK
    )
    if args.subtitle:
        ax.set_title(
            args.subtitle, loc="left", fontsize=11, color=MUTED, pad=14, wrap=True
        )
    if len(series) > 1:
        ax.legend(loc="upper right", fontsize=11, facecolor="white", framealpha=1)
    if args.note:
        fig.text(0.05, 0.015, args.note, fontsize=9, color=MUTED)
    fig.tight_layout(rect=(0, 0.03, 1, 0.95))
    taken: list[Any] = []  # labels of later series avoid those of earlier ones
    for s in series:
        _label_points(ax, s["points"], taken)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)
    print(f"wrote {out}")
    return 0


# ---------------------------------------------------------------------------
# cli
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="command", required=True)

    g = sub.add_parser("plot", help="render the chart")
    g.add_argument(
        "--series",
        action="append",
        required=True,
        metavar="DIRECTORY[:NAME[:#COLOR]]",
        help="one curve per result directory, fields separated by colons; "
        "repeatable; later "
        "series are drawn over earlier ones",
    )
    g.add_argument("--out", required=True)
    g.add_argument("--title", default="Output Throughput vs. Interactivity")
    g.add_argument("--subtitle", default="")
    g.add_argument("--note", default="", help="one grey line under the x-axis label")
    args = parser.parse_args(argv)
    return plot(args)


if __name__ == "__main__":
    sys.exit(main())
