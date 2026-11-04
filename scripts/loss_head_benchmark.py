#!/usr/bin/env python3
r"""Loss-head benchmark for the DFlash and DSpark drafters.

The loss head is everything between the drafter's final (post-norm) hidden
states and the scalar loss: the ``lm_head`` projection (plus DSpark's Markov
logit bias), the no-grad verifier projection that produces the targets, and
the model's own ``compute_metrics``. Measuring it alone isolates the
``[N, V]`` logits cost from the decoder layers. The head itself is
``speculators.models.loss_head``; this script times it. Full training steps are
covered by ``scripts/benchmark.py run --synthetic``.

Subcommand:
    run    Peak memory and forward/backward time over a matrix of models, rows,
           vocab sizes, losses, weightings and implementations

Examples:
    python scripts/loss_head_benchmark.py run --model dflash,dspark \
        --rows 8192,16384 --vocab 110000,151936 --output loss_head.json

    python scripts/loss_head_benchmark.py run --model dspark --rows 16384 \
        --vocab 151936 --loss '{"kl_div": 0.5, "tv": 0.5}' --weighting dpace
"""

from __future__ import annotations

import argparse
import dataclasses
import gc
import itertools
import json
import socket
import statistics
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch

from speculators.models.loss_head import (
    IMPLEMENTATIONS,
    MODELS,
    WEIGHTINGS,
    HeadConfig,
    cached_loss_config,
    loss_head,
    make_inputs,
)
from speculators.provenance import find_repo_root, git_sha, pkg_version

if TYPE_CHECKING:
    from collections.abc import Callable

BENCHMARK_VERSION = "1.0"
MIB = 1024**2


# ---------------------------------------------------------------------------
# Measurement helpers
# ---------------------------------------------------------------------------


def _device() -> torch.device:
    if not torch.cuda.is_available():
        sys.exit("loss_head_benchmark.py needs a CUDA or ROCm device")
    return torch.device("cuda", torch.cuda.current_device())


def _free_device_memory() -> None:
    gc.collect()
    torch.cuda.empty_cache()


def collect_provenance() -> dict:
    """Versions and device metadata for reproducibility."""

    triton = pkg_version("triton")
    if triton == "not installed":
        triton = pkg_version("triton-rocm")
    device: dict[str, Any] = {}
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(torch.cuda.current_device())
        device = {
            "name": props.name,
            "arch": getattr(props, "gcnArchName", None),
            "total_memory_gb": round(props.total_memory / 2**30, 2),
        }
    return {
        "benchmark_version": BENCHMARK_VERSION,
        "git_sha": git_sha(find_repo_root(Path(__file__).resolve().parent)),
        "timestamp": datetime.now(UTC).isoformat(),
        "hostname": socket.gethostname(),
        "argv": sys.argv,
        "python_version": sys.version.split()[0],
        "pytorch_version": torch.__version__,
        "hip_version": torch.version.hip,
        "cuda_version": torch.version.cuda,
        "triton_version": triton,
        "speculators_version": pkg_version("speculators"),
        "device": device,
    }


def _write_json(path: str | None, payload: dict) -> None:
    if path is None:
        return
    Path(path).write_text(json.dumps(payload, indent=2) + "\n")
    print(f"wrote {path}")


def _csv(value: str, cast: Callable[[str], Any] = str) -> list[Any]:
    return [cast(item) for item in value.split(",") if item]


def _int_at_least(minimum: int) -> Callable[[str], int]:
    def parse(value: str) -> int:
        number = int(value)
        if number < minimum:
            raise argparse.ArgumentTypeError(f"must be >= {minimum}, got {number}")
        return number

    return parse


# ---------------------------------------------------------------------------
# run
# ---------------------------------------------------------------------------


def measure_head(cfg: HeadConfig, iters: int, warmup: int) -> dict[str, Any]:
    """Peak memory above the inputs and fwd/bwd device time of one config."""
    device = _device()
    inputs = make_inputs(cfg, device)

    def step() -> tuple[float, float]:
        for tensor in inputs.trainable():
            tensor.grad = None
        start = torch.cuda.Event(enable_timing=True)
        mid = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        loss, _ = loss_head(cfg, inputs)
        mid.record()
        loss.backward()
        end.record()
        end.synchronize()
        return start.elapsed_time(mid), mid.elapsed_time(end)

    # Warmup also triggers Triton autotuning, whose trial buffers must not
    # count towards the peak.
    for _ in range(warmup):
        step()
    for tensor in inputs.trainable():
        tensor.grad = None
    torch.cuda.synchronize()
    baseline = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    fwd, bwd = zip(*(step() for _ in range(iters)), strict=True)
    peak = torch.cuda.max_memory_allocated() - baseline

    nv_bytes = cfg.num_rows * cfg.vocab * inputs.weight.element_size()
    return {
        "status": "ok",
        "peak_above_inputs_mb": round(peak / MIB, 1),
        "nv_tensor_mb": round(nv_bytes / MIB, 1),
        "peak_in_nv_tensors": round(peak / nv_bytes, 2),
        "fwd_ms": round(statistics.median(fwd), 3),
        "bwd_ms": round(statistics.median(bwd), 3),
        "total_ms": round(
            statistics.median(f + b for f, b in zip(fwd, bwd, strict=True)), 3
        ),
        "fwd_ms_all": [round(t, 3) for t in fwd],
        "bwd_ms_all": [round(t, 3) for t in bwd],
    }


def cmd_run(args: argparse.Namespace) -> dict:
    configs = [
        HeadConfig(
            model=model,
            rows=rows,
            vocab=vocab,
            hidden=args.hidden,
            block_size=args.block_size,
            markov_rank=args.markov_rank,
            loss=loss,
            weighting=weighting,
            implementation=impl,
            dtype=args.dtype,
        )
        for model, rows, vocab, loss, weighting, impl in itertools.product(
            _csv(args.model),
            _csv(args.rows, int),
            _csv(args.vocab, int),
            args.loss or ["kl_div"],
            _csv(args.weighting),
            _csv(args.implementation),
        )
    ]
    # Validate the whole matrix before measuring anything.
    for cfg in configs:
        for name, value, choices in (
            ("model", cfg.model, MODELS),
            ("weighting", cfg.weighting, WEIGHTINGS),
            ("implementation", cfg.implementation, IMPLEMENTATIONS),
        ):
            if value not in choices:
                sys.exit(f"unknown {name} {value!r}; choose from {choices}")
        if cfg.vocab < 1:
            sys.exit(f"--vocab must be positive, got {cfg.vocab}")
        if cfg.num_rows <= 0:
            sys.exit(f"--rows {cfg.rows} is smaller than --block-size {cfg.block_size}")
        try:
            # fused_linear* compute the fused losses from hidden states.
            impl = cfg.implementation
            loss_impl = "fused" if impl.startswith("fused_linear") else impl
            cached_loss_config(cfg.loss, loss_impl)
        except ValueError as err:
            sys.exit(f"invalid --loss {cfg.loss!r}: {err}")
    # fused_linear covers DFlash only; skip the other models' combinations.
    configs = [
        cfg
        for cfg in configs
        if not (cfg.implementation == "fused_linear" and cfg.model != "dflash")
    ]
    _device()

    results = []
    header = (
        f"{'model':7} {'rows':>6} {'vocab':>7} {'impl':6} {'weighting':16} "
        f"{'peak MiB':>9} {'xNV':>5} {'fwd ms':>8} {'bwd ms':>8}  loss"
    )
    print(header)
    for cfg in configs:
        try:
            measured = measure_head(cfg, args.iters, args.warmup)
        except torch.OutOfMemoryError:
            measured = {"status": "oom"}
        _free_device_memory()
        results.append({"config": dataclasses.asdict(cfg), **measured})
        if measured["status"] == "ok":
            row = (
                f"{measured['peak_above_inputs_mb']:9.0f} "
                f"{measured['peak_in_nv_tensors']:5.2f} "
                f"{measured['fwd_ms']:8.2f} {measured['bwd_ms']:8.2f}"
            )
        else:
            row = f"{'OOM':>9} {'':5} {'':8} {'':8}"
        print(
            f"{cfg.model:7} {cfg.num_rows:6d} {cfg.vocab:7d} {cfg.implementation:6} "
            f"{cfg.weighting:16} {row}  {cfg.loss}"
        )
    return {"subcommand": "run", "provenance": collect_provenance(), "results": results}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="subcommand", required=True)

    run = sub.add_parser("run", help="loss-head peak memory and time")
    run.add_argument(
        "--model", default="dflash,dspark", help="comma list of dflash,dspark"
    )
    run.add_argument(
        "--rows", default="8192,16384", help="comma list; rows N = anchors x block"
    )
    run.add_argument("--vocab", default="110000,151936", help="comma list")
    run.add_argument("--hidden", type=_int_at_least(1), default=4096)
    run.add_argument("--block-size", type=_int_at_least(1), default=8)
    run.add_argument("--markov-rank", type=_int_at_least(1), default=256)
    run.add_argument(
        "--loss",
        action="append",
        help="loss spec as accepted by --loss-fn (repeatable); default kl_div",
    )
    run.add_argument(
        "--weighting",
        default="fixed-exp-decay",
        help="comma list of fixed-exp-decay,dpace",
    )
    run.add_argument(
        "--implementation",
        default="fused",
        help=f"comma list of {','.join(IMPLEMENTATIONS)} (fused_linear: dflash only)",
    )
    run.add_argument(
        "--dtype", default="bfloat16", choices=("bfloat16", "float16", "float32")
    )
    run.add_argument("--iters", type=_int_at_least(1), default=5)
    run.add_argument("--warmup", type=_int_at_least(0), default=2)
    run.add_argument("--output", help="write results as JSON to this path")
    run.set_defaults(func=cmd_run)

    return parser


def main(argv: list[str] | None = None) -> dict:
    args = build_parser().parse_args(argv)
    payload = args.func(args)
    _write_json(args.output, payload)
    return payload


if __name__ == "__main__":
    main()
