#!/usr/bin/env python3
"""Prepare official LongBench prompts for the performance evaluator.

The source archive and its component datasets retain their original licenses.
This script downloads them at runtime and writes prompt-only JSONL files; do not
redistribute the prepared files without checking those licenses.

Usage::

    python prepare_longbench.py --data-dir ./longbench_data --download
"""

from __future__ import annotations

import argparse
import io
import json
import logging
import shutil
import sys
from pathlib import Path
from urllib.request import urlopen
from zipfile import ZipFile

logger = logging.getLogger("prepare_longbench")

# Pin both sources so the rendered benchmark prompts are reproducible.
_DATA_URL = (
    "https://huggingface.co/datasets/zai-org/LongBench/resolve/"
    "5e628be450b7e67fb7ae6e201bd6d8f7056f7672/data.zip"
)
_PROMPTS_URL = (
    "https://raw.githubusercontent.com/THUDM/LongBench/"
    "2e00731f8d0bff23dc4325161044d0ed8af94c1e/"
    "LongBench/config/dataset2prompt.json"
)


def _download(url: str, path: Path) -> None:
    logger.info("Downloading %s ...", url)
    partial = path.with_suffix(f"{path.suffix}.part")
    with urlopen(url) as source, partial.open("wb") as destination:  # noqa: S310
        shutil.copyfileobj(source, destination)
    partial.replace(path)


def prepare(
    archive_path: Path,
    prompts_path: Path,
    output_dir: Path,
) -> int:
    """Render official ``context``/``input`` templates into prompt JSONL files."""
    templates = json.loads(prompts_path.read_text(encoding="utf-8"))
    output_dir.mkdir(parents=True, exist_ok=True)

    with ZipFile(archive_path) as archive:
        for task, template in templates.items():
            source_name = f"data/{task}.jsonl"
            output_path = output_dir / f"longbench_{task}.jsonl"
            rows = 0
            with (
                archive.open(source_name) as source,
                io.TextIOWrapper(source, encoding="utf-8") as lines,
                output_path.open("w", encoding="utf-8") as output,
            ):
                for line in lines:
                    if not line.strip():
                        continue
                    prompt = template.format(**json.loads(line))
                    output.write(json.dumps({"prompt": prompt}, ensure_ascii=False))
                    output.write("\n")
                    rows += 1
            logger.info("  wrote %s (%d rows)", output_path.name, rows)

    return len(templates)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    parser = argparse.ArgumentParser(
        description="Prepare prompt-only LongBench JSONL files.",
    )
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument(
        "--download",
        action="store_true",
        help="Download the official data archive and prompt templates if missing",
    )
    args = parser.parse_args()

    data_dir: Path = args.data_dir
    data_dir.mkdir(parents=True, exist_ok=True)
    archive_path = data_dir / "data.zip"
    prompts_path = data_dir / "dataset2prompt.json"
    if args.download:
        if not archive_path.exists():
            _download(_DATA_URL, archive_path)
        if not prompts_path.exists():
            _download(_PROMPTS_URL, prompts_path)

    missing = [path for path in (archive_path, prompts_path) if not path.is_file()]
    if missing:
        logger.error("Missing %s; rerun with --download.", ", ".join(map(str, missing)))
        sys.exit(1)

    written = prepare(archive_path, prompts_path, data_dir)
    logger.info("Done. %d files written to %s", written, data_dir)


if __name__ == "__main__":
    main()
