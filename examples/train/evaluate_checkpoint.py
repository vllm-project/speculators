#!/usr/bin/env python3
"""Checkpoint evaluator example using the repository's vLLM and AIPerf tools."""

import argparse
import csv
import json
import math
import os
import socket
import subprocess
import sys
import time
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen
from uuid import uuid4

REPO_ROOT = Path(__file__).resolve().parents[2]


def read_acceptance(path: Path) -> dict[str, float]:
    """Keep pooled metrics separate for each dataset; never average row means."""
    metrics: dict[str, float] = {}
    datasets: set[str] = set()
    with path.open() as handle:
        for row in csv.DictReader(handle):
            dataset = row.get("dataset")
            if not dataset or dataset in datasets or not row.get("acceptance_length"):
                raise ValueError("Missing or duplicate dataset acceptance results")
            datasets.add(dataset)
            for name, text in row.items():
                if name == "dataset" or not text:
                    continue
                value = float(text)
                if not math.isfinite(value):
                    raise ValueError(f"Non-finite acceptance metric {name}")
                metrics[f"{dataset}/{name}"] = value
    if not metrics:
        raise ValueError("No acceptance metrics found")
    return metrics


def wait_ready(
    process: subprocess.Popen, url: str, timeout: float, expected_model: str
) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(
                f"Evaluation server exited with code {process.returncode}"
            )
        try:
            with urlopen(f"{url}/v1/models", timeout=1) as response:
                if response.status == 200 and process.poll() is None:
                    models = json.load(response)
                    if [model["id"] for model in models["data"]] != [expected_model]:
                        raise RuntimeError("Evaluation server model identity mismatch")
                    return
        except (URLError, TimeoutError):
            pass
        time.sleep(0.2)
    raise TimeoutError("Evaluation server did not become ready")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verifier", required=True)
    parser.add_argument("--cuda-visible-devices", required=True)
    parser.add_argument("--port", type=int, default=9145)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--spec-tokens", type=int, default=7)
    parser.add_argument("--spec-method", help="Defaults to the checkpoint's algorithm")
    parser.add_argument(
        "--no-hash-checkpoints",
        action="store_true",
        help="Record file sizes and mtimes instead of hashing local model weights",
    )
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--dataset", nargs="+", default=["gsm8k"])
    parser.add_argument("--request-count", type=int, default=8)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--startup-timeout", type=float, default=180)
    parser.add_argument(
        "--extra-inputs", default='{"temperature":0,"max_completion_tokens":256}'
    )
    args = parser.parse_args()

    checkpoint = Path(os.environ["SPECULATORS_EVAL_CHECKPOINT"])
    spec_method = (
        args.spec_method
        or json.loads((checkpoint / "config.json").read_text())["speculators_config"][
            "algorithm"
        ]
    )
    output = Path(os.environ["SPECULATORS_EVAL_OUTPUT_DIR"])
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=args.cuda_visible_devices)
    url = f"http://127.0.0.1:{args.port}"
    served_model = f"speculators-eval-{uuid4().hex}"
    # Do not accidentally benchmark a pre-existing server at the requested port.
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", args.port))

    serve = [
        sys.executable,
        str(REPO_ROOT / "scripts/launch_vllm.py"),
        "eval",
        args.verifier,
        "--spec-model",
        str(checkpoint),
        "--spec-tokens",
        str(args.spec_tokens),
        "--spec-method",
        spec_method,
        "--provenance-dir",
        str(output / "server"),
        *(["--no-hash-checkpoints"] if args.no_hash_checkpoints else []),
        "--",
        "--host",
        "127.0.0.1",
        "--port",
        str(args.port),
        "--tensor-parallel-size",
        str(args.tensor_parallel_size),
        "--max-model-len",
        str(args.max_model_len),
        "--gpu-memory-utilization",
        str(args.gpu_memory_utilization),
        "--served-model-name",
        served_model,
        "--per-request-spec-decode-metrics",
        "summary",
        "--no-enable-prefix-caching",
    ]
    if args.enforce_eager:
        serve.append("--enforce-eager")
    evaluate = [
        sys.executable,
        str(REPO_ROOT / "scripts/evaluate/evaluate.py"),
        "acceptance",
        "--output-artifact-dir",
        str(output / "acceptance"),
        "--dataset",
        *args.dataset,
        "--model",
        served_model,
        "--tokenizer",
        args.verifier,
        "--url",
        url,
        "--request-count",
        str(args.request_count),
        "--concurrency",
        str(args.concurrency),
        "--extra-inputs",
        args.extra_inputs,
    ]
    with (output / "vllm.log").open("w") as log:
        # Inherit the evaluator's session so its timeout also kills vLLM workers.
        server = subprocess.Popen(  # noqa: S603
            serve, env=env, cwd=REPO_ROOT, stdout=log, stderr=subprocess.STDOUT
        )
        try:
            wait_ready(server, url, args.startup_timeout, served_model)
            subprocess.run(evaluate, env=env, cwd=REPO_ROOT, check=True)  # noqa: S603
            if server.poll() is not None:
                raise RuntimeError("Evaluation server exited during profiling")
            metrics = read_acceptance(output / "acceptance/acceptance.csv")
            (output / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
        finally:
            server.terminate()
            try:
                server.wait(timeout=10)
            except subprocess.TimeoutExpired:
                server.kill()
                server.wait()


if __name__ == "__main__":
    main()
