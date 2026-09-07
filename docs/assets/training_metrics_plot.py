#!/usr/bin/env python3
"""Draw the reference_prefix_acc_i figures in docs/user_guide/training_metrics.md.

Reads ``training_metrics_curves.csv`` from this directory and writes
``training_metrics_prefix_acc_{1,2,3}.png`` beside it. Only matplotlib and the
standard library are needed::

    python docs/assets/training_metrics_plot.py

How the CSV was produced
------------------------
Five drafters were trained on one Qwen3-8B verifier and one offline
hidden-state cache with identical arguments apart from ``--speculator-type``::

    torchrun --standalone --nproc_per_node 4 -m speculators.train \\
        --verifier-name-or-path Qwen/Qwen3-8B \\
        --data-path <preprocessed tutorial_regen, 4993 rows> \\
        --hidden-states-path <offline cache> --on-missing raise \\
        --speculator-type <eagle3|peagle|dflash|dflash2|dspark> \\
        --epochs 3 --lr 3e-4 --total-seq-len 8192 --seed 42 --log-freq 10

No ``--draft-vocab-size`` was passed, so every drafter used the full 151936
verifier vocabulary; no ``--target-layer-ids`` was passed, so every drafter used
the default ``[2, n//2, n-3]``. The CSV holds every ``reference_prefix_acc_i``
value the trainer logged, with the ``sum`` and ``total`` counts behind each rate:
``split=train`` rows are per-``--log-freq`` training steps, ``split=val`` rows are
per-epoch validation recorded at the step the epoch ended on.

Columns: ``drafter, split, position, step, acc, sum, total``.
"""

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import MultipleLocator  # noqa: E402

HERE = Path(__file__).parent
CSV = HERE / "training_metrics_curves.csv"

# Fixed categorical order. Validated as a set on a light surface: passes the
# lightness band, chroma floor, CVD separation and normal-vision floors. Three
# slots fall below 3:1 contrast, so the figures carry a legend and the page
# carries a table of the same numbers.
DRAFTERS = [
    ("eagle3", "#2a78d6"),
    ("peagle", "#eb6834"),
    ("dflash", "#1baf7a"),
    ("dflash2", "#eda100"),
    ("dspark", "#e87ba4"),
]
INK, MUTED, GRID = "#16191c", "#5c6169", "#e3e3e0"
STEPS_PER_EPOCH = 471
TOTAL_STEPS = 1413
YMAX = {1: 0.8, 2: 0.45, 3: 0.30}
SMOOTH = 5


def load():
    """Return (train series, final validation total) keyed by drafter and position."""
    train: dict[tuple[str, int], list[tuple[int, float]]] = {}
    final_total: dict[tuple[str, int], float] = {}
    with CSV.open(newline="") as f:
        for r in csv.DictReader(f):
            key = (r["drafter"], int(r["position"]))
            if r["split"] == "train":
                train.setdefault(key, []).append((int(r["step"]), float(r["acc"])))
            elif r["total"]:
                final_total[key] = float(r["total"])
    for v in train.values():
        v.sort()
    return train, final_total


def rolling(ys, w):
    return [sum(ys[max(0, k - w + 1) : k + 1]) / len(ys[max(0, k - w + 1) : k + 1])
            for k in range(len(ys))]


def main():
    train, final_total = load()
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "axes.edgecolor": GRID,
        "axes.labelcolor": INK,
        "text.color": INK,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "figure.dpi": 200,
    })

    for i in (1, 2, 3):
        fig, ax = plt.subplots(figsize=(6.8, 4.0))
        for b in (1, 2):
            ax.axvline(b * STEPS_PER_EPOCH, color=GRID, lw=1, zorder=1)
            ax.text(b * STEPS_PER_EPOCH, YMAX[i] * 0.02, f" epoch {b + 1}",
                    color=MUTED, fontsize=7.5, va="bottom", ha="left", zorder=2)
        ax.grid(axis="y", color=GRID, lw=0.8, zorder=0)
        ax.set_axisbelow(True)

        for name, color in DRAFTERS:
            series = train[(name, i)]
            xs = [x for x, _ in series]
            ys = [y for _, y in series]
            n = final_total[(name, i)]
            # raw per-batch values behind, as texture; rolling mean in front
            ax.plot(xs, ys, color=color, lw=0.9, alpha=0.20, zorder=2)
            ax.plot(xs, rolling(ys, SMOOTH), color=color, lw=1.9,
                    solid_capstyle="round", zorder=3,
                    label=f"{name}  (N={n / 1000:.0f}k)")

        ax.set_xlim(0, TOTAL_STEPS)
        ax.set_ylim(0, YMAX[i])
        ax.xaxis.set_major_locator(MultipleLocator(300))
        ax.set_xlabel("training step", color=MUTED)
        ax.set_ylabel(f"reference_prefix_acc_{i}", color=INK)
        ax.set_title(
            f"Prefix agreement at position {i}"
            + ("" if i == 1 else f"  (first {i} predictions all match)"),
            color=INK, fontsize=10.5, loc="left", pad=16,
        )
        ax.text(0, 1.015, f"{SMOOTH}-point rolling mean; raw per-batch values shown faint",
                transform=ax.transAxes, color=MUTED, fontsize=7.5, va="bottom")
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        leg = ax.legend(frameon=False, fontsize=8, loc="upper left", handlelength=1.6)
        for t in leg.get_texts():
            t.set_color(INK)
        fig.tight_layout()
        out = HERE / f"training_metrics_prefix_acc_{i}.png"
        fig.savefig(out, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
