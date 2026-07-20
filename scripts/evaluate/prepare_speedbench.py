#!/usr/bin/env python3
"""Split SPEED-Bench flat files into per-category/subcategory JSONL files.

**Step 1** — run NVIDIA's prepare.py to materialise prompts from their
original sources (fetched at runtime due to redistribution restrictions)::

    curl -LsSf https://raw.githubusercontent.com/NVIDIA-NeMo/Skills/ \\
        refs/heads/main/nemo_skills/dataset/speed-bench/prepare.py \\
        | python3 - --output_dir ./speedbench_data

.. note::
    The URL above points to the ``main`` branch of an external repository
    maintained by NVIDIA.  Save a local copy of the script if you anticipate
    running data preparation again::

        curl -LsSf <url> -o prepare_nvidia_speedbench.py

    The output files produced by prepare.py contain data fetched from
    third-party sources that carry their own licences.  Do not redistribute
    the materialised JSONL files.

**Step 2** — run this script to split the flat files into per-category files
that evaluate.py can consume directly::

    python prepare_speedbench.py --data-dir ./speedbench_data

Output files follow the naming convention ``{config}_{key}.jsonl`` where
``key`` is the category (qualitative) or ``{entropy}__{subcategory}``
(throughput configs).  Each file has a single ``turns`` column.

Usage::

    python prepare_speedbench.py --data-dir ./speedbench_data \
        [--configs qualitative,throughput_1k]
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
import urllib.request
from pathlib import Path

logger = logging.getLogger("prepare_speedbench")

_PLACEHOLDER_MARKERS = (
    "FULL BENCHMARK DATA SHOULD BE FETCHED FROM THE SOURCE",
    "{article}",
)

_ALL_CONFIGS = [
    "qualitative",
    "throughput_1k",
    "throughput_2k",
    "throughput_8k",
    "throughput_32k",
]


def _extract_text(row: dict) -> str:
    """Return prompt text — supports both ``turns`` and ``messages`` columns.

    .. note::
        For multi-turn rows only the first turn is extracted.  Full multi-turn
        conversation support is out of scope for this initial implementation.
    """
    text = row.get("turns") or ""
    if not text:
        msgs = row.get("messages") or []
        # TODO: support multi-turn by concatenating all turns
        text = msgs[0].get("content", "") if msgs else ""
    return text


def split_config(flat_file: Path, out_dir: Path, config: str) -> int:
    """Split *flat_file* into per-category/subcategory files in *out_dir*.

    Returns the number of output files written.
    """
    logger.info("Loading %s ...", flat_file)
    with flat_file.open() as f:
        rows = [json.loads(line) for line in f if line.strip()]

    use_category_only = config == "qualitative"
    buckets: dict[str, list[str]] = {}

    for row in rows:
        text = _extract_text(row)
        if not text:
            continue
        for marker in _PLACEHOLDER_MARKERS:
            if marker in text:
                cat = row.get("category", "?")
                logger.error(
                    "Placeholder text in %s/%s — re-run NVIDIA prepare.py first.",
                    config,
                    cat,
                )
                sys.exit(1)

        cat = (row.get("category") or "unknown").replace(" ", "_").replace("/", "_")
        sub = (row.get("sub_category") or "unknown").replace(" ", "_").replace("/", "_")
        key = cat if use_category_only else f"{cat}__{sub}"
        buckets.setdefault(key, []).append(text)

    written = 0
    for key, texts in sorted(buckets.items()):
        out_path = out_dir / f"{config}_{key}.jsonl"
        with out_path.open("w") as f:
            for t in texts:
                f.write(json.dumps({"turns": t}) + "\n")
        logger.info("  wrote %s (%d rows)", out_path.name, len(texts))
        written += 1

    return written


_NVIDIA_PREPARE_URL = (
    "https://raw.githubusercontent.com/NVIDIA-NeMo/Skills/"
    "refs/heads/main/nemo_skills/dataset/speed-bench/prepare.py"
)


def _run_nvidia_prepare(data_dir: Path, configs: list[str]) -> None:
    """Download and run NVIDIA's prepare.py to materialise prompts."""
    logger.info("Downloading NVIDIA prepare.py from %s ...", _NVIDIA_PREPARE_URL)
    with urllib.request.urlopen(_NVIDIA_PREPARE_URL) as resp:  # noqa: S310
        script_bytes = resp.read()

    for config in configs:
        logger.info("Running prepare.py for config=%s ...", config)
        subprocess.run(  # noqa: S603
            [sys.executable, "-", "--config", config, "--output_dir", str(data_dir)],
            input=script_bytes,
            check=True,
        )
        logger.info("prepare.py done for config=%s", config)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

    parser = argparse.ArgumentParser(
        description="Split SPEED-Bench flat files into per-category JSONL files.",
    )
    parser.add_argument(
        "--data-dir",
        required=True,
        type=Path,
        help=(
            "Directory with flat JSONL files from NVIDIA prepare.py"
            " (e.g. qualitative.jsonl)."
        ),
    )
    parser.add_argument(
        "--configs",
        default=",".join(_ALL_CONFIGS),
        help=(f"Comma-separated configs to split (default: {','.join(_ALL_CONFIGS)})"),
    )
    parser.add_argument(
        "--download",
        action="store_true",
        default=False,
        help=(
            "Download and run NVIDIA's prepare.py to materialise prompts before "
            "splitting. Fetches from the NVIDIA-NeMo/Skills repository."
        ),
    )
    args = parser.parse_args()

    data_dir: Path = args.data_dir
    data_dir.mkdir(parents=True, exist_ok=True)
    configs = [c.strip() for c in args.configs.split(",") if c.strip()]

    if args.download:
        _run_nvidia_prepare(data_dir, configs)

    if not data_dir.exists():
        logger.error("--data-dir '%s' does not exist.", data_dir)
        sys.exit(1)

    total = 0
    for config in configs:
        flat_file = data_dir / f"{config}.jsonl"
        if not flat_file.exists():
            logger.warning("Skipping %s — %s not found.", config, flat_file)
            continue
        n = split_config(flat_file, data_dir, config)
        total += n
        logger.info("%s: %d files written.", config, n)

    logger.info("Done. %d total files written to %s", total, data_dir)


if __name__ == "__main__":
    main()
