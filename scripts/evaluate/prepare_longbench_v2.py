#!/usr/bin/env python3
"""Prepare LongBench-v2 prompts for the performance evaluator."""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
from pathlib import Path
from urllib.request import urlopen

logger = logging.getLogger("prepare_longbench_v2")

_DATA_URL = (
    "https://huggingface.co/datasets/zai-org/LongBench-v2/resolve/"
    "2b48e494f2c7a2f0af81aae178e05c7e1dde0fe9/data.json"
)
_PROMPT = """Please read the following text and answer the question below.

<text>
{context}
</text>

What is the correct answer to this question: {question}
Choices:
(A) {choice_A}
(B) {choice_B}
(C) {choice_C}
(D) {choice_D}

Format your response as follows: "The correct answer is (insert answer here)"."""


def _download(url: str, path: Path) -> None:
    logger.info("Downloading %s ...", url)
    partial = path.with_suffix(f"{path.suffix}.part")
    # Bounds each socket read, not the transfer: a stall raises instead of
    # hanging a 465 MB download indefinitely.
    with (
        urlopen(url, timeout=30) as source,  # noqa: S310
        partial.open("wb") as destination,
    ):
        shutil.copyfileobj(source, destination)
    partial.replace(path)


def prepare(source_path: Path, output_path: Path) -> int:
    """Render the official zero-shot prompt into a GuideLLM JSONL file."""
    with source_path.open(encoding="utf-8") as source:
        rows = json.load(source)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as output:
        for row in rows:
            prompt = _PROMPT.format(
                **{
                    key: row[key].strip()
                    for key in (
                        "context",
                        "question",
                        "choice_A",
                        "choice_B",
                        "choice_C",
                        "choice_D",
                    )
                }
            )
            output.write(json.dumps({"prompt": prompt}, ensure_ascii=False) + "\n")

    return len(rows)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    parser = argparse.ArgumentParser(
        description="Prepare prompt-only LongBench-v2 JSONL data.",
    )
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument(
        "--download",
        action="store_true",
        help="Download the pinned official data file if missing",
    )
    args = parser.parse_args()

    data_dir: Path = args.data_dir
    data_dir.mkdir(parents=True, exist_ok=True)
    source_path = data_dir / "data.json"
    if args.download and not source_path.exists():
        _download(_DATA_URL, source_path)
    if not source_path.is_file():
        logger.error("Missing %s; rerun with --download.", source_path)
        sys.exit(1)

    output_path = data_dir / "longbench_v2.jsonl"
    rows = prepare(source_path, output_path)
    logger.info("Wrote %d prompts to %s", rows, output_path)


if __name__ == "__main__":
    main()
