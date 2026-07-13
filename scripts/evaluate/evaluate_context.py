#!/usr/bin/env python3
"""Evaluate speculative decoding acceptance rates across context lengths.

Uses LongBench2 prompts truncated to target context lengths (2k-20k tokens),
sent to a running vLLM server with speculative decoding, with acceptance metrics
scraped from the Prometheus endpoint between batches.

Produces:
    - acceptance.csv with per-position acceptance rates per context-length bin
    - Acceptance length vs context length plot
    - Per-position acceptance rate vs context length plot
    - Raw results JSON

Usage:
    python evaluate_context.py \
        --target http://localhost:8000/v1 \
        --label "Eagle3-SW"

    python evaluate_context.py \
        --target http://localhost:8000/v1 \
        --label "Peagle-SW" \
        --append --output-dir context_length_results
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen

from perf_utils import (
    CsvWriter,
    acceptance_csv_columns,
    extract_spec_decode_metrics,
    fetch_metrics,
    parse_prometheus_metrics,
    print_acceptance_report,
)

logger = logging.getLogger("evaluate_context")

TARGET_LENGTHS = [2000, 4000, 6000, 8000, 10000, 12000, 14000, 16000, 18000, 20000]
DEFAULT_SAMPLES_PER_BIN = 10
MAX_NEW_TOKENS = 128

LONGBENCH2_DATASET = "zai-org/LongBench-v2"


def _fetch_model_name(base_url: str) -> str | None:
    url = base_url.rstrip("/") + "/models"
    try:
        with urlopen(url, timeout=10) as resp:  # noqa: S310
            data = json.loads(resp.read())
        models = data.get("data", [])
        if models:
            return models[0].get("id")
    except (URLError, json.JSONDecodeError, OSError) as e:
        logger.warning("Could not fetch model name: %s", e)
    return None


def load_longbench2_prompts(tokenizer) -> dict[str, list[dict]]:
    """Load LongBench2 and return {sub_domain: [{ctx_token_count, ctx_tokens}]}."""
    from collections import defaultdict

    from datasets import load_dataset

    ds = load_dataset(LONGBENCH2_DATASET, split="train")
    by_category: dict[str, list[dict]] = defaultdict(list)
    for row in ds:
        ctx_tokens = tokenizer.encode(row["context"], add_special_tokens=False)
        by_category[row["sub_domain"]].append(
            {
                "ctx_token_count": len(ctx_tokens),
                "ctx_tokens": ctx_tokens,
            }
        )
    for cat in by_category:
        by_category[cat].sort(key=lambda r: r["ctx_token_count"], reverse=True)

    total = sum(len(v) for v in by_category.values())
    logger.info(
        "Loaded %d LongBench2 examples across %d sub-domains",
        total,
        len(by_category),
    )
    for cat, rows in sorted(by_category.items()):
        logger.info(
            "  %s: %d examples (min=%d, max=%d tokens)",
            cat,
            len(rows),
            rows[-1]["ctx_token_count"],
            rows[0]["ctx_token_count"],
        )
    return dict(by_category)


def build_prompts_per_category_and_length(
    rows_by_category: dict[str, list[dict]],
    tokenizer,
    target_lengths: list[int],
    samples_per_bin: int,
) -> dict[str, dict[int, list[str]]]:
    """For each (sub_domain, target_length), create prompts by truncating contexts.

    Returns {sub_domain: {target_length: [prompt_str, ...]}}.
    """
    result: dict[str, dict[int, list[str]]] = {}
    for category, rows in sorted(rows_by_category.items()):
        prompts_by_length: dict[int, list[str]] = {}
        for target_len in target_lengths:
            eligible = [r for r in rows if r["ctx_token_count"] >= target_len]
            if not eligible:
                continue
            if len(eligible) < samples_per_bin:
                logger.warning(
                    "[%s] Only %d examples >= %d tokens (need %d), using all",
                    category,
                    len(eligible),
                    target_len,
                    samples_per_bin,
                )
            selected = eligible[:samples_per_bin]
            prompts = []
            for row in selected:
                truncated_tokens = row["ctx_tokens"][:target_len]
                truncated_text = tokenizer.decode(
                    truncated_tokens, skip_special_tokens=False
                )
                prompt = (
                    "Please read the following text and provide a brief summary.\n\n"
                    f"<text>\n{truncated_text}\n</text>\n\nSummary:"
                )
                prompts.append(prompt)
            prompts_by_length[target_len] = prompts
            actual_lens = [
                len(tokenizer.encode(p, add_special_tokens=False)) for p in prompts
            ]
            logger.info(
                "[%s] Context %dk: %d prompts, actual token range [%d, %d]",
                category,
                target_len // 1000,
                len(prompts),
                min(actual_lens),
                max(actual_lens),
            )
        if prompts_by_length:
            result[category] = prompts_by_length
    return result


def _send_completions(
    base_url: str,
    prompts: list[str],
    model_name: str,
    max_tokens: int = MAX_NEW_TOKENS,
):
    import openai

    client = openai.OpenAI(base_url=base_url, api_key="EMPTY")
    for i, prompt in enumerate(prompts):
        try:
            client.completions.create(
                model=model_name,
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=0.0,
            )
        except Exception as e:
            logger.warning("Request %d/%d failed: %s", i + 1, len(prompts), e)


def evaluate_context_lengths(
    target: str,
    label: str,
    prompts_by_category: dict[str, dict[int, list[str]]],
    output_dir: Path,
) -> dict[str, dict[int, dict]]:
    """Run evaluation against a running vLLM server, writing CSV incrementally.

    Returns {sub_domain: {context_length: spec_metrics}}.
    """
    base_url = target.rstrip("/")
    if not base_url.endswith("/v1"):
        base_url += "/v1"
    metrics_url = base_url.removesuffix("/v1") + "/metrics"
    results: dict[str, dict[int, dict]] = {}

    model_name = _fetch_model_name(base_url)
    if not model_name:
        raise RuntimeError(f"Could not determine model name from {base_url}")
    logger.info("Model: %s", model_name)

    acceptance_csv = None

    for category, prompts_by_length in sorted(prompts_by_category.items()):
        cat_results: dict[int, dict] = {}
        for target_len in sorted(prompts_by_length.keys()):
            prompts = prompts_by_length[target_len]
            logger.info(
                "[%s][%s] Evaluating context length %dk (%d prompts)...",
                label,
                category,
                target_len // 1000,
                len(prompts),
            )

            baseline_text = fetch_metrics(metrics_url)
            baseline = parse_prometheus_metrics(baseline_text) if baseline_text else []

            _send_completions(base_url, prompts, model_name)

            current_text = fetch_metrics(metrics_url)
            current = parse_prometheus_metrics(current_text) if current_text else []

            spec = extract_spec_decode_metrics(current, baseline_metrics=baseline)
            has_spec = spec and spec.get("num_drafts", 0) > 0

            if has_spec:
                spec["sub_domain"] = category
                spec["context_length"] = target_len
                cat_results[target_len] = spec
                print_acceptance_report(spec)

                if acceptance_csv is None:
                    acceptance_csv = CsvWriter(
                        output_dir / "acceptance.csv",
                        ["sub_domain", "context_length"]
                        + acceptance_csv_columns(spec),
                    )
                acceptance_csv.append(spec)

                logger.info(
                    "[%s][%s] %dk: acceptance_length=%.3f, drafts=%.0f",
                    label,
                    category,
                    target_len // 1000,
                    spec.get("acceptance_length", 0),
                    spec.get("num_drafts", 0),
                )
            else:
                logger.warning(
                    "[%s][%s] No spec decode metrics at %dk",
                    label,
                    category,
                    target_len // 1000,
                )

        if cat_results:
            results[category] = cat_results

    return results


def _aggregate_across_categories(
    cat_results: dict[str, dict[int, dict]],
) -> dict[int, dict]:
    """Average spec-decode metrics across sub-domains for each context length."""
    from collections import defaultdict

    by_length: dict[int, list[dict]] = defaultdict(list)
    for cat_data in cat_results.values():
        for ctx_len, spec in cat_data.items():
            by_length[ctx_len].append(spec)

    aggregated: dict[int, dict] = {}
    for ctx_len, specs in sorted(by_length.items()):
        agg: dict[str, float] = {}
        keys = [k for k in specs[0] if isinstance(specs[0][k], (int, float))]
        for k in keys:
            agg[k] = sum(s.get(k, 0) for s in specs) / len(specs)
        aggregated[ctx_len] = agg
    return aggregated


def plot_results(
    all_label_results: dict[str, dict[str, dict[int, dict]]],
    output_dir: Path,
):
    """Plot acceptance metrics.

    all_label_results: {label: {sub_domain: {context_length: spec_metrics}}}

    Produces:
        - Aggregated acceptance length vs context length (one line per label)
        - Per-position acceptance rate vs context length (one subplot per label)
        - Per-category acceptance length vs context length (one subplot per label)
    """
    import matplotlib.pyplot as plt
    import numpy as np

    output_dir.mkdir(parents=True, exist_ok=True)
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
    position_colors = [
        "#2ca02c",
        "#1f77b4",
        "#ff7f0e",
        "#d62728",
        "#9467bd",
        "#8c564b",
        "#e377c2",
        "#7f7f7f",
    ]

    aggregated_by_label = {
        label: _aggregate_across_categories(cat_results)
        for label, cat_results in all_label_results.items()
    }

    # --- Plot 1: Aggregated Acceptance Length vs Context Length ---
    fig, ax = plt.subplots(figsize=(10, 6))
    for i, (label, agg) in enumerate(aggregated_by_label.items()):
        lengths = sorted(agg.keys())
        acc_lens = [agg[l].get("acceptance_length", 0) for l in lengths]
        x = [l / 1000 for l in lengths]
        ax.plot(
            x,
            acc_lens,
            "o-",
            color=colors[i % len(colors)],
            linewidth=2,
            markersize=8,
            label=label,
        )

    ax.set_xlabel("Context Length (k tokens)", fontsize=13)
    ax.set_ylabel("Acceptance Length", fontsize=13)
    ax.set_title(
        "Acceptance Length vs Context Length (aggregated)",
        fontsize=14,
        fontweight="bold",
    )
    ax.legend(fontsize=12, framealpha=0.9)
    ax.grid(True, alpha=0.3)
    ax.set_xticks([l / 1000 for l in TARGET_LENGTHS])
    fig.tight_layout()
    outpath = output_dir / "acceptance_length_vs_context.png"
    fig.savefig(outpath, dpi=150)
    plt.close(fig)
    logger.info("Saved %s", outpath)

    # --- Plot 2: Per-position acceptance rates (aggregated) ---
    fig, axes = plt.subplots(
        1,
        len(aggregated_by_label),
        figsize=(8 * len(aggregated_by_label), 6),
        squeeze=False,
    )

    for col, (label, agg) in enumerate(aggregated_by_label.items()):
        ax = axes[0, col]
        lengths = sorted(agg.keys())
        x = [l / 1000 for l in lengths]

        max_pos = 0
        for r in agg.values():
            pos = 0
            while f"acceptance_at_pos_{pos}" in r:
                pos += 1
            max_pos = max(max_pos, pos)

        for pos in range(max_pos):
            key = f"acceptance_at_pos_{pos}"
            rates = [agg[l].get(key, float("nan")) for l in lengths]
            if any(not np.isnan(r) for r in rates):
                ax.plot(
                    x,
                    rates,
                    "o-",
                    color=position_colors[pos % len(position_colors)],
                    linewidth=2,
                    markersize=6,
                    label=f"Position {pos}",
                )

        ax.set_xlabel("Context Length (k tokens)", fontsize=12)
        ax.set_ylabel("Acceptance Rate", fontsize=12)
        ax.set_title(f"{label}", fontsize=13, fontweight="bold")
        ax.legend(fontsize=10, framealpha=0.9)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1)
        ax.set_xticks([l / 1000 for l in TARGET_LENGTHS])

    fig.suptitle(
        "Per-Position Acceptance Rate vs Context Length (aggregated)",
        fontsize=14,
        fontweight="bold",
        y=1.02,
    )
    fig.tight_layout()
    outpath = output_dir / "acceptance_rate_per_position.png"
    fig.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s", outpath)

    # --- Plot 3: Per-category acceptance length (one subplot per label) ---
    for label, cat_results in all_label_results.items():
        categories = sorted(cat_results.keys())
        n_cats = len(categories)
        if n_cats == 0:
            continue
        ncols = min(4, n_cats)
        nrows = (n_cats + ncols - 1) // ncols
        fig, axes = plt.subplots(
            nrows, ncols, figsize=(6 * ncols, 5 * nrows), squeeze=False
        )
        for idx, category in enumerate(categories):
            ax = axes[idx // ncols, idx % ncols]
            results = cat_results[category]
            lengths = sorted(results.keys())
            acc_lens = [results[l].get("acceptance_length", 0) for l in lengths]
            x = [l / 1000 for l in lengths]
            ax.plot(x, acc_lens, "o-", color=colors[0], linewidth=2, markersize=6)
            ax.set_xlabel("Context Length (k tokens)", fontsize=10)
            ax.set_ylabel("Acceptance Length", fontsize=10)
            ax.set_title(category, fontsize=11, fontweight="bold")
            ax.grid(True, alpha=0.3)

        for idx in range(n_cats, nrows * ncols):
            axes[idx // ncols, idx % ncols].set_visible(False)

        fig.suptitle(
            f"{label} — Acceptance Length by Sub-Domain",
            fontsize=14,
            fontweight="bold",
        )
        fig.tight_layout()
        safe_label = label.replace("/", "_").replace(" ", "_")
        outpath = output_dir / f"acceptance_by_category_{safe_label}.png"
        fig.savefig(outpath, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info("Saved %s", outpath)

    # --- Save raw results as JSON ---
    json_results = {}
    for label, cat_results in all_label_results.items():
        json_results[label] = {
            cat: {str(k): v for k, v in lengths.items()}
            for cat, lengths in cat_results.items()
        }
    json_path = output_dir / "results.json"
    with json_path.open("w") as f:
        json.dump(json_results, f, indent=2)
    logger.info("Saved %s", json_path)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate speculative decoding acceptance across context lengths",
    )
    parser.add_argument(
        "--target",
        required=True,
        help="vLLM server endpoint (e.g. http://localhost:8000/v1)",
    )
    parser.add_argument(
        "--label",
        required=True,
        help="Label for this model run (used in plots, CSV, and results JSON)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory (default: scripts/evaluate/context_length_results)",
    )
    parser.add_argument(
        "--samples-per-bin",
        type=int,
        default=DEFAULT_SAMPLES_PER_BIN,
        help=f"Number of prompts per context length bin (default: {DEFAULT_SAMPLES_PER_BIN})",
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        default="Qwen/Qwen3-8B",
        help="Tokenizer for context length measurement (default: Qwen/Qwen3-8B)",
    )
    parser.add_argument(
        "--append",
        action="store_true",
        help="Append to existing results.json instead of overwriting",
    )
    return parser.parse_args()


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="[%(levelname)s] %(message)s",
        stream=sys.stderr,
    )
    args = parse_args()

    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else (Path(__file__).parent / "context_length_results")
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)

    logger.info("Loading LongBench2 data...")
    rows_by_category = load_longbench2_prompts(tokenizer)

    logger.info("Building prompts for %d context lengths...", len(TARGET_LENGTHS))
    prompts_by_category = build_prompts_per_category_and_length(
        rows_by_category,
        tokenizer,
        TARGET_LENGTHS,
        args.samples_per_bin,
    )

    logger.info("=" * 60)
    logger.info("Evaluating: %s (target=%s)", args.label, args.target)
    logger.info("=" * 60)
    results = evaluate_context_lengths(
        target=args.target,
        label=args.label,
        prompts_by_category=prompts_by_category,
        output_dir=output_dir,
    )

    # {label: {sub_domain: {context_length: metrics}}}
    all_results: dict[str, dict[str, dict[int, dict]]] = {}
    json_path = output_dir / "results.json"
    if args.append and json_path.exists():
        with json_path.open() as f:
            prev = json.load(f)
        for lbl, cat_data in prev.items():
            all_results[lbl] = {
                cat: {int(k): v for k, v in lengths.items()}
                for cat, lengths in cat_data.items()
            }
        logger.info("Loaded previous results for: %s", list(all_results.keys()))

    all_results[args.label] = results

    logger.info("Plotting results...")
    plot_results(all_results, output_dir)
    logger.info("Done! Results saved to %s", output_dir)


if __name__ == "__main__":
    main()
