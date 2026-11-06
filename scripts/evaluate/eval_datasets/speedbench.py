"""Build SPEED-Bench arguments and prepare cached inputs for AIPerf."""

import gzip
import json
import os
import subprocess
import sys
from argparse import Namespace
from collections import Counter
from functools import cache
from pathlib import Path
from tempfile import TemporaryDirectory
from urllib.request import urlopen

SPEEDBENCH_PREPARE_COMMIT = "5ac8609a56ac941540b10c92e68d556e6343cd4c"
SPEEDBENCH_PREPARE_URL = (
    f"https://raw.githubusercontent.com/NVIDIA-NeMo/Skills/{SPEEDBENCH_PREPARE_COMMIT}/"
    "nemo_skills/dataset/speed-bench/prepare.py"
)


def prepare_speedbench(dataset_type: str) -> Path:
    """Cache NVIDIA's prepared split; AIPerf still loads and filters the rows."""
    config = "qualitative"
    if dataset_type.startswith("speed_bench_throughput_"):
        config = "throughput_" + dataset_type.split("_")[3]
    cache = (
        Path(os.environ.get("HF_HOME", str(Path.home() / ".cache/huggingface")))
        / "speculators"
        / "speedbench"
        / SPEEDBENCH_PREPARE_COMMIT
    )
    prepared = cache / f"{config}.jsonl"
    if prepared.is_file():
        return prepared
    cache.mkdir(parents=True, exist_ok=True)
    script = cache / "prepare.py"
    if not script.is_file():
        with urlopen(SPEEDBENCH_PREPARE_URL, timeout=60) as response:  # noqa: S310
            script.write_bytes(response.read())
    with TemporaryDirectory(dir=cache) as temporary:
        subprocess.run(  # noqa: S603
            [
                sys.executable,
                str(script),
                "--config",
                config,
                "--output_dir",
                temporary,
            ],
            check=True,
        )
        (Path(temporary) / prepared.name).replace(prepared)
    return prepared


@cache
def conversation_counts(source: Path) -> Counter[str]:
    """Count source conversations per category for native acceptance runs."""
    opener = gzip.open if source.suffix == ".gz" else open
    with opener(source, "rt", encoding="utf-8") as handle:
        return Counter(json.loads(line)["category"] for line in handle if line.strip())


def speedbench_args(name: str, args: Namespace) -> list[str]:
    """Prepare SPEED-Bench inputs and add the default acceptance budget."""
    source = args.input_file or prepare_speedbench(name)
    arguments = ["--custom-dataset-type", name]
    if args.input_file is None:
        arguments.extend(["--input-file", str(source)])
    if args.mode == "acceptance" and not any(
        (args.request_count, args.num_conversations, args.benchmark_duration)
    ):
        counts = conversation_counts(source.resolve())
        category = name.removeprefix("speed_bench_")
        count = sum(counts.values()) if category == "qualitative" else counts[category]
        if count:
            arguments.extend(["--num-conversations", str(count)])
    return arguments


if __name__ == "__main__":
    print(prepare_speedbench("speed_bench_qualitative"))
