"""Prepare and cache SPEED-Bench locally, or load existing prepared files."""

import argparse
import json
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from urllib.request import urlretrieve

from datasets import load_dataset
from huggingface_hub.constants import HF_HOME

CONFIGS = (
    "qualitative",
    "throughput_1k",
    "throughput_2k",
    "throughput_8k",
    "throughput_16k",
    "throughput_32k",
)
PREPARE_COMMIT = "5ac8609a56ac941540b10c92e68d556e6343cd4c"
PREPARE_URL = (
    f"https://raw.githubusercontent.com/NVIDIA-NeMo/Skills/{PREPARE_COMMIT}/"
    "nemo_skills/dataset/speed-bench/prepare.py"
)


def _prepare(config: str | None) -> list[Path]:
    """Run NVIDIA's preparation script once and reuse complete local files."""
    cache = Path(HF_HOME) / "speculators" / "speedbench"
    configs = (config,) if config else CONFIGS
    files = [cache / f"{name}.jsonl" for name in configs]
    if not all(path.is_file() for path in files):
        cache.mkdir(parents=True, exist_ok=True)
        print(f"Preparing SPEED-Bench locally in {cache}", flush=True)
        with TemporaryDirectory(dir=cache) as temporary:
            script = Path(temporary) / "prepare.py"
            urlretrieve(PREPARE_URL, script)  # noqa: S310
            subprocess.run(  # noqa: S603
                [
                    sys.executable,
                    str(script),
                    "--config",
                    config or "all",
                    "--output_dir",
                    temporary,
                ],
                check=True,
            )
            for path in files:
                (Path(temporary) / path.name).replace(path)
    return files


def _rows(source: str | Path):
    """Read prepared JSONL without inferring a schema for optional metadata."""
    if Path(source).suffix == ".parquet":
        yield from load_dataset(
            "parquet", data_files=str(source), split="train", streaming=True
        )
    else:
        with Path(source).open() as handle:
            for line in handle:
                if line.strip():
                    yield json.loads(line)


def _conversations(source: str | Path):
    for row in _rows(source):
        # NeMo Skills writes messages; Model Optimizer writes turns.
        yield {
            "messages": (
                row["messages"][:1]
                if "messages" in row
                else [{"role": "user", "content": row["turns"][0]}]
            ),
            "category": row.get("category"),
        }


def load(args: argparse.Namespace) -> dict:
    """Use explicit prepared files, or prepare the selected configurations."""
    sources = args.data_files or _prepare(args.dataset_config)
    return {
        f"speedbench/{Path(source).stem}": _conversations(source) for source in sources
    }
