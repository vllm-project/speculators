#!/usr/bin/env python3
"""Training benchmark harness for speculators.

Measures training loop throughput, timing breakdown, and GPU memory usage
with full provenance tracking. Results are stored as JSON for comparison
across code changes.

Uses the real Trainer class so measurements stay in sync with the actual
training code path.

Subcommands:
    run        Run a training benchmark
    compare    Compare two benchmark result files
    report     Generate an interactive HTML report from results
    comment    Render results as markdown for a GitHub PR comment

Examples:
    # Synthetic benchmark (no dataset / vLLM needed)
    python scripts/benchmark.py run --synthetic \\
        -- --verifier-name-or-path Qwen/Qwen3-8B --total-seq-len 4096

    # Real data benchmark
    python scripts/benchmark.py run \\
        -- --verifier-name-or-path Qwen/Qwen3-8B --data-path ./output \\
        --on-missing skip

    # Multi-GPU
    torchrun --standalone --nproc_per_node 2 scripts/benchmark.py run \\
        --synthetic -- --verifier-name-or-path Qwen/Qwen3-8B

    # With torch.profiler trace (outputs Chrome trace to profile_traces/)
    python scripts/benchmark.py run --synthetic --profile \\
        -- --verifier-name-or-path Qwen/Qwen3-8B --total-seq-len 4096

    # Compare two runs
    python scripts/benchmark.py compare baseline.json candidate.json

    # Generate interactive HTML report
    python scripts/benchmark.py report benchmark_20260818.json

    # Post a before/after summary on a PR (run both on the same machine first)
    python scripts/benchmark.py comment after.json --baseline before.json \\
        --output comment.md
    gh pr comment 1234 --body-file comment.md
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import logging
import math
import os
import re
import shlex
import socket
import statistics
import subprocess
import sys
from datetime import datetime, timezone
from html import escape as html_escape
from pathlib import Path

import torch
import torch.distributed as dist

from hs_connectors import HiddenStatesBackend
from speculators.model import SpeculatorModel
from speculators.models.eagle3.data import shift_batch
from speculators.models.mtp.data import shift_batch_mtp
from speculators.train.cli import (
    build_draft_model,
    parse_vocab_mappings,
    set_seed,
)
from speculators.train.config import TrainConfig
from speculators.train.dataloader import BatchType, create_train_val_loaders
from speculators.train.distributed import (
    get_rank,
    maybe_destroy_distributed,
    maybe_setup_distributed,
)
from speculators.train.logger import setup_root_logger
from speculators.train.trainer import Trainer, TrainerConfig

BENCHMARK_VERSION = "1.3"

# Rows shown in the --profile kernel breakdown tables and the compare diff.
KERNEL_TOP_N = 20
# Mangled C++ and cuBLAS kernel names run to several hundred characters and
# differ only near their ends, so the tables elide their middle at this width
# (see _short_kernel_name). Pie legend entries have less room than a table
# cell, but not enough less to warrant a second width.
KERNEL_NAME_WIDTH = 52
# Kernels below this share of device time are merged into one "Other" slice.
# A profiled step launches dozens of kernels and all but a handful are under a
# millisecond; drawn individually they are an unreadable fringe of hairlines.
KERNEL_PIE_MIN_PCT = 2.0


def _short_kernel_name(name: str, width: int = KERNEL_NAME_WIDTH) -> str:
    """Shorten a kernel name to ``width`` characters by eliding its middle.

    Mangled C++ and cuBLAS names share long prefixes and differ only near the
    end (``..._stages_32x3_nn`` against ``..._stages_64x3_tn``), so clipping the
    tail renders distinct kernels as identical-looking rows. Keeping both ends
    does not.
    """
    if len(name) <= width:
        return name
    tail = (width - 1) // 3
    return name[: width - 1 - tail] + "…" + name[len(name) - tail :]


def _unique_labels(labels: list[str]) -> list[str]:
    """Suffix repeated labels: plotly merges pie slices that share a label.

    Elision can still collide on kernels that differ only in the elided middle,
    and a silent merge would misattribute their time.
    """
    seen: dict[str, int] = {}
    out = []
    for label in labels:
        seen[label] = seen.get(label, 0) + 1
        out.append(label if seen[label] == 1 else f"{label} #{seen[label]}")
    return out


def _wrap_kernel_name(name: str, width: int = 60) -> str:
    """Break a kernel name across lines so a hover tooltip stays on screen.

    Escaped chunk by chunk: the result is interpolated into a plotly hover
    template, which renders it as HTML, and the breaks are the only markup
    this should contribute.
    """
    return "<br>".join(
        html_escape(name[i : i + width]) for i in range(0, len(name), width)
    )


def _invocation() -> str:
    """The current command line, in the form someone reproducing it would type.

    The script path is made repo-relative: an absolute one leaks the layout of
    whichever machine happened to run the benchmark and is not what a reader
    would paste back into their own checkout.

    Under torchrun, ``sys.argv`` is the worker's -- the launcher and its process
    count are in the environment, not the argv -- so a bare ``python ...`` here
    would claim a single-process run reproduces a distributed one. The launcher
    is put back from the variables torchelastic sets. Multi-node runs also need
    a rendezvous the environment cannot supply from one worker, so those are
    marked for the reader rather than guessed at.
    """
    argv0 = sys.argv[0]
    try:
        argv0 = str(Path(argv0).resolve().relative_to(Path.cwd()))
    except ValueError:
        argv0 = Path(argv0).name

    local_world_size = int(os.environ.get("LOCAL_WORLD_SIZE", "0"))
    if "TORCHELASTIC_RUN_ID" not in os.environ or local_world_size < 1:
        return shlex.join(["python", argv0, *sys.argv[1:]])

    nnodes = int(os.environ.get("GROUP_WORLD_SIZE", "1"))
    launcher = ["torchrun"]
    if nnodes > 1:
        launcher += [f"--nnodes={nnodes}", "--rdzv-endpoint=<host:port>"]
    else:
        launcher.append("--standalone")
    launcher.append(f"--nproc-per-node={local_world_size}")
    return shlex.join([*launcher, argv0, *sys.argv[1:]])


# ---------------------------------------------------------------------------
# Metric capture
# ---------------------------------------------------------------------------


class _MetricCapture(logging.Handler):
    """Captures profile dicts emitted by the Trainer via metric_logger."""

    def __init__(self):
        super().__init__()
        self.profiles: list[dict] = []

    def emit(self, record):
        msg = record.msg
        if isinstance(msg, dict) and "profile" in msg and msg["profile"] is not None:
            self.profiles.append(msg["profile"])


# ---------------------------------------------------------------------------
# Synthetic data loader
# ---------------------------------------------------------------------------


class _SyntheticLoader:
    """DataLoader-like wrapper that yields the same batch repeatedly.

    Satisfies the Trainer's expectations: ``__len__``, ``__iter__``, and a
    ``batch_sampler`` with ``set_epoch``.
    """

    class _BatchSampler:
        def set_epoch(self, _epoch):
            pass

    def __init__(self, batch: BatchType, num_steps: int):
        self._batch = batch
        self._num_steps = num_steps
        self.batch_sampler = self._BatchSampler()

    def __len__(self):
        return self._num_steps

    def __iter__(self):
        for _ in range(self._num_steps):
            yield self._batch


# ---------------------------------------------------------------------------
# Provenance
# ---------------------------------------------------------------------------


def collect_provenance() -> dict:
    """Gather system and version metadata for reproducibility."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],  # noqa: S607
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        git_sha = result.stdout.strip() if result.returncode == 0 else "unknown"
    except OSError:
        git_sha = "unknown"

    gpu_info = []
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            gpu_info.append(
                {
                    "name": props.name,
                    "total_memory_gb": round(props.total_memory / 2**30, 2),
                    "compute_capability": [props.major, props.minor],
                }
            )

    def _version(pkg: str) -> str:
        try:
            return importlib.metadata.version(pkg)
        except importlib.metadata.PackageNotFoundError:
            return "unknown"

    return {
        "git_sha": git_sha,
        # The invocation is the one piece of provenance a reader cannot
        # reconstruct from the config dump, because the train args after '--'
        # are forwarded verbatim. Mirrors train_command.txt, which records argv
        # for the same reason.
        "command": _invocation(),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "hostname": socket.gethostname(),
        "python_version": sys.version.split()[0],
        "pytorch_version": torch.__version__,
        "cuda_version": torch.version.cuda or "none",
        "speculators_version": _version("speculators"),
        "transformers_version": _version("transformers"),
        "gpu_info": gpu_info,
        "num_gpus": (torch.cuda.device_count() if torch.cuda.is_available() else 0),
    }


# ---------------------------------------------------------------------------
# Synthetic data
# ---------------------------------------------------------------------------


def create_synthetic_batch(
    total_seq_len: int,
    hidden_size: int,
    num_target_layers: int,
    vocab_size: int = 32000,
    dtype: torch.dtype = torch.bfloat16,
    device: torch.device | int = 0,
) -> BatchType:
    """Create a random batch matching the post-collation training shape."""
    hs_dim = num_target_layers * hidden_size
    return {
        "hidden_states": torch.randn(
            1, total_seq_len, hs_dim, dtype=dtype, device=device
        ),
        "input_ids": torch.randint(0, vocab_size, (1, total_seq_len), device=device),
        "verifier_last_hidden_states": torch.randn(
            1, total_seq_len, hidden_size, dtype=dtype, device=device
        ),
        "loss_mask": torch.ones(1, total_seq_len, dtype=torch.bool, device=device),
        "position_ids": torch.arange(
            1, total_seq_len + 1, device=device, dtype=torch.long
        ).unsqueeze(0),
        "document_ids": torch.zeros(1, total_seq_len, dtype=torch.long, device=device),
        "error_records": 0,
    }


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------


def compute_statistics(values: list[float]) -> dict[str, float]:
    """Compute summary statistics with 95% confidence interval."""
    from scipy import stats as sp_stats  # noqa: PLC0415

    n = len(values)
    mean = statistics.mean(values)
    std = statistics.stdev(values) if n > 1 else 0.0
    sem = std / math.sqrt(n) if n > 1 else 0.0
    ci_half = sp_stats.t.ppf(0.975, n - 1) * sem if n > 1 else 0.0
    return {
        "mean": mean,
        "std": std,
        "min": min(values),
        "max": max(values),
        "median": statistics.median(values),
        "count": n,
        "ci95_lower": mean - ci_half,
        "ci95_upper": mean + ci_half,
    }


def measured_step_window(warmup_steps: int, measured_steps: int) -> range:
    """Which trainer steps the run measures — the single source of truth.

    The same 0-based window drives the profiler schedule, the per-step
    capture slice, and the kernel-time normalization, so the per-step
    timings and the kernel breakdown always describe the same steps.
    """
    return range(warmup_steps, warmup_steps + measured_steps)


def select_measured_profiles(all_profiles: list[dict], window: range) -> list[dict]:
    """Validate the sample count and discard warmup profiles."""
    if len(all_profiles) < window.stop:
        raise RuntimeError(
            "Benchmark dataset exhausted before the requested number of steps: "
            f"got {len(all_profiles)}, requested {window.stop}. Use more "
            "samples or request fewer benchmark steps."
        )
    return list(all_profiles[window.start : window.stop])


def compute_aggregate_throughput(profiles: list[dict]) -> dict[str, float]:
    """Compute time-weighted throughput for the measured window.

    ``mean(tokens_per_s)`` overweights short, fast steps. Reconstructing each
    profile's token count and dividing by total elapsed time gives the effective
    throughput observed by rank 0 over the complete measurement window.
    """
    elapsed_s = sum(profile["step_ms"] for profile in profiles) / 1000
    rank0_tokens = sum(
        profile["tokens_per_s"] * profile["step_ms"] / 1000 for profile in profiles
    )
    return {
        "measured_time_s": elapsed_s,
        "rank0_tokens": rank0_tokens,
        "effective_rank0_tokens_per_s": rank0_tokens / elapsed_s,
    }


def shutdown_dataloader_workers(loader) -> None:
    """Stop persistent workers before Mooncake/distributed teardown."""
    iterator = getattr(loader, "_iterator", None)
    shutdown = getattr(iterator, "_shutdown_workers", None)
    if callable(shutdown):
        shutdown()
        loader._iterator = None


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------


def _build_train_loader(
    bench_args,
    train_args,
    hidden_size,
    num_target_layers,
    vocab_size,
    hidden_states_dtype,
    total_steps,
):
    """Build either a synthetic or real data loader for benchmarking."""
    if bench_args.synthetic:
        synth_batch = create_synthetic_batch(
            total_seq_len=train_args.total_seq_len,
            hidden_size=hidden_size,
            num_target_layers=num_target_layers,
            vocab_size=vocab_size,
            dtype=hidden_states_dtype,
            device="cpu",
        )
        return _SyntheticLoader(synth_batch, total_steps), True

    preprocess_fns = {
        "eagle3": shift_batch,
        "peagle": shift_batch,
        "mtp": shift_batch_mtp,
    }
    preprocess = preprocess_fns.get(train_args.speculator_type)

    backend_registry = HiddenStatesBackend.registry
    backend_cls = backend_registry[train_args.hidden_states_backend]
    transfer = backend_cls.from_train_args(train_args, train_args.data_path)

    train_loader, _ = create_train_val_loaders(
        data_path=train_args.data_path,
        total_seq_len=train_args.total_seq_len,
        hidden_states_dtype=hidden_states_dtype,
        transfer=transfer,
        vllm_endpoint=train_args.vllm_endpoint,
        on_missing=train_args.on_missing,
        verifier_name_or_path=train_args.verifier_name_or_path,
        request_timeout=train_args.request_timeout,
        max_retries=train_args.max_retries,
        generation_validation_retries=train_args.generation_validation_retries,
        max_consecutive_generation_failures=1,
        hidden_size=hidden_size,
        num_target_layers=num_target_layers,
        num_workers=train_args.num_workers,
        prefetch_factor=train_args.prefetch_factor,
        preprocess=preprocess,
        train_data_ratio=train_args.train_data_ratio,
        max_train_batches=total_steps,
        seed=train_args.seed,
    )
    return train_loader, False


def _aggregate_timing(measured_profiles: list[dict]) -> dict:
    """Compute statistics for all timing keys across measured profiles."""
    agg = {}
    for key, value in measured_profiles[0].items():
        if isinstance(value, dict):
            continue
        agg[key] = compute_statistics([s[key] for s in measured_profiles])
    return agg


def _is_device_kernel(evt) -> bool:
    """True for launched GPU kernels among ``key_averages()`` entries.

    ``key_averages()`` emits one entry per launched device kernel alongside
    the ATen op entries; the CUDA-device-type rows are the kernels
    themselves, so the breakdown needs no extra instrumentation inside the
    model. This is how a change to one kernel (an attention backend, a
    fused loss tile) shows up attributed to that kernel instead of being
    averaged into ``fwd_ms``.

    User annotations (``ProfilerStep*``, ``Optimizer.step#...``, the
    inductor ``## Call CompiledFxGraph <hash>`` spans) are also CUDA-typed
    but are enclosing ranges, not kernels: keeping them would double-count
    the total, and the graph hashes change whenever the compiled graph
    does, so they turn a before/after diff into noise.
    """
    return (
        evt.device_type == torch.autograd.DeviceType.CUDA
        and not getattr(evt, "is_user_annotation", False)
        and evt.self_device_time_total > 0
    )


def _local_kernel_rows(prof) -> list[dict]:
    """This rank's kernel rows as plain dicts, safe to send in a collect."""
    return [
        {
            "name": evt.key,
            "self_device_us": evt.self_device_time_total,
            "count": evt.count,
        }
        for evt in prof.key_averages()
        if _is_device_kernel(evt)
    ]


def _aggregate_kernel_rows(per_rank_rows: list[list[dict]], active_steps: int) -> dict:
    """Aggregate per-rank kernel rows into a per-step breakdown.

    Device time is summed across ranks (total GPU time per step over the
    run's devices) and divided by ``active_steps`` — the same window
    ``select_measured_profiles`` slices for the per-step timings, so both
    tables describe the same steps.
    """
    steps = max(active_steps, 1)
    totals: dict[str, list] = {}
    for rows in per_rank_rows:
        for row in rows:
            acc = totals.setdefault(row["name"], [0.0, 0])
            acc[0] += row["self_device_us"]
            acc[1] += row["count"]
    kernels = [
        {
            "name": name,
            "ms_per_step": us / 1000.0 / steps,
            "calls_per_step": count / steps,
        }
        for name, (us, count) in totals.items()
        if us > 0
    ]
    kernels.sort(key=lambda k: -k["ms_per_step"])
    return {
        "total_device_ms_per_step": sum(k["ms_per_step"] for k in kernels),
        "kernels": kernels,
        "num_ranks": len(per_rank_rows),
    }


def _collect_kernel_breakdown(prof, active_steps: int) -> dict:
    """Kernel breakdown aggregated across every rank, normalized per step."""
    rows = _local_kernel_rows(prof)
    if dist.is_initialized() and dist.get_world_size() > 1:
        gathered: list = [None] * dist.get_world_size()
        dist.all_gather_object(gathered, rows)
    else:
        gathered = [rows]
    return _aggregate_kernel_rows(gathered, active_steps)


def _run_training(bench_args, trainer, rank, window: range) -> dict | None:
    """Run the training loop, optionally with torch.profiler.

    The profiler is active exactly over ``window`` (``wait=0``, warmup for
    the first ``window.start`` steps, then ``len(window)`` active steps),
    so the per-step capture and the kernel breakdown cover the same steps.
    Timing fields measured inside the window do include profiling overhead;
    the device-side kernel durations do not.

    Returns the aggregated per-kernel breakdown when profiling, else None.
    """
    if not bench_args.profile:
        trainer.train_epoch(0)
        return None

    profile_dir = Path(bench_args.profile_dir)
    profile_dir.mkdir(parents=True, exist_ok=True)
    # Timestamped per (run, rank) so concurrent or repeated runs do not
    # collide in a shared trace directory.
    worker_name = f"{Path(bench_args.output).stem}-rank{rank}"
    steps_completed = 0

    def _profiler_step() -> None:
        nonlocal steps_completed
        steps_completed += 1
        prof.step()

    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        schedule=torch.profiler.schedule(
            wait=0,
            warmup=window.start,
            active=len(window),
            repeat=1,
        ),
        on_trace_ready=torch.profiler.tensorboard_trace_handler(
            str(profile_dir), worker_name=worker_name
        ),
        record_shapes=True,
        profile_memory=True,
        with_stack=bench_args.profile_stacks,
    ) as prof:
        trainer.train_epoch(0, step_callback=_profiler_step)

    # The window is only meaningful if the callback fired exactly once per
    # step; catch loop/profiler drift as an error, not a silent skew.
    if steps_completed != window.stop:
        raise RuntimeError(
            f"profiler stepped {steps_completed} times but the run had "
            f"{window.stop} steps; step_callback must fire once per trainer "
            "step"
        )
    if rank == 0:
        print(f"Profile traces written to {profile_dir}")
    return _collect_kernel_breakdown(prof, len(window))


def _gather_per_rank(entry: dict) -> list[dict]:
    """Collect one small per-rank dict (e.g. peak memory) onto every rank."""
    if dist.is_initialized() and dist.get_world_size() > 1:
        gathered: list = [None] * dist.get_world_size()
        dist.all_gather_object(gathered, entry)
        return sorted(gathered, key=lambda e: e["rank"])
    return [entry]


def _phase_peaks(measured_profiles: list[dict]) -> dict[str, float]:
    """Peak allocated MiB at each timer phase, over the measured steps.

    The per-step ``memory_mb`` marks say where in the step memory peaks;
    this collapses them into one row per phase.
    """
    peaks: dict[str, float] = {}
    for profile in measured_profiles:
        for phase, mb in profile.get("memory_mb", {}).items():
            if mb > peaks.get(phase, 0.0):
                peaks[phase] = mb
    return {phase: round(mb, 2) for phase, mb in peaks.items()}


def run_benchmark(bench_args, train_args) -> dict:
    """Execute the benchmark using the real Trainer and return results."""
    set_seed(
        train_args.seed,
        getattr(train_args, "deterministic_cuda", False),
    )
    setup_root_logger()
    maybe_setup_distributed()

    rank = get_rank()
    total_steps = bench_args.warmup_steps + bench_args.measured_steps
    hidden_states_dtype = getattr(torch, train_args.hidden_states_dtype)

    # --- Build model ---
    if train_args.speculator_type == "mtp":
        d2t, t2d, draft_vocab_size = None, None, None
        train_args.mask_token_id = None
    else:
        d2t, t2d, draft_vocab_size = parse_vocab_mappings(train_args)

    model_class = SpeculatorModel.registry[train_args.speculator_type]
    draft_model = build_draft_model(train_args, model_class, t2d, d2t, draft_vocab_size)

    num_target_layers = len(draft_model.target_layer_ids)
    hidden_size = draft_model.config.transformer_layer_config.hidden_size
    vocab_size = draft_model.config.transformer_layer_config.vocab_size

    # --- Build data loader ---
    train_loader, is_synthetic = _build_train_loader(
        bench_args,
        train_args,
        hidden_size,
        num_target_layers,
        vocab_size,
        hidden_states_dtype,
        total_steps,
    )

    # --- Get forward kwargs ---
    train_call_kwargs, _ = model_class.get_trainer_kwargs(**vars(train_args))

    # --- Build TrainerConfig ---
    trainer_config = TrainerConfig(
        lr=train_args.lr,
        num_epochs=1,
        save_path="benchmark_unused",
        resume_from_checkpoint=False,
        train_call_kwargs=train_call_kwargs,
        optimizer=train_args.optimizer,
        weight_decay=train_args.weight_decay,
        muon_lr=train_args.muon_lr,
        muon_momentum=train_args.muon_momentum,
        muon_weight_decay=train_args.muon_weight_decay,
        muon_ns_steps=train_args.muon_ns_steps,
        muon_adjust_lr_fn=train_args.muon_adjust_lr_fn,
        scheduler_type="none",
        hidden_states_dtype=hidden_states_dtype,
        log_freq=1,
        fsdp_shard=train_args.fsdp_shard,
        max_steps=total_steps,
    )

    # --- Construct Trainer (handles GPU placement, DDP, optimizer) ---
    trainer = Trainer(draft_model, trainer_config, train_loader)

    if rank == 0:
        print(
            f"Benchmarking: {bench_args.warmup_steps} warmup + "
            f"{bench_args.measured_steps} measured steps"
        )

    # --- Attach metric capture ---
    metric_logger = logging.getLogger("speculators.metrics")
    capture = _MetricCapture()
    metric_logger.addHandler(capture)

    # --- Reset memory tracking before the run ---
    local_rank = trainer.local_rank
    torch.accelerator.reset_peak_memory_stats(local_rank)

    window = measured_step_window(bench_args.warmup_steps, bench_args.measured_steps)

    # --- Run the real training loop ---
    kernel_breakdown = _run_training(bench_args, trainer, rank, window)

    # --- Remove capture handler ---
    metric_logger.removeHandler(capture)

    # --- Split warmup / measured profiles ---
    measured_profiles = select_measured_profiles(capture.profiles, window)

    # --- Collect memory: every rank reports its own device ---
    own_memory = {
        "rank": rank,
        "peak_allocated_mb": round(
            torch.accelerator.max_memory_allocated(local_rank) / (1024**2), 2
        ),
        "peak_reserved_mb": round(
            torch.accelerator.max_memory_reserved(local_rank) / (1024**2), 2
        ),
    }
    per_rank_memory = _gather_per_rank(own_memory)

    # --- Aggregate ---
    timing_agg = _aggregate_timing(measured_profiles)
    aggregate = compute_aggregate_throughput(measured_profiles)
    phase_peaks = _phase_peaks(measured_profiles)

    num_gpus_used = dist.get_world_size() if dist.is_initialized() else 1

    results = {
        "benchmark_version": BENCHMARK_VERSION,
        "provenance": collect_provenance(),
        "config": {
            "speculator_type": train_args.speculator_type,
            "verifier_name_or_path": train_args.verifier_name_or_path,
            "total_seq_len": train_args.total_seq_len,
            "hidden_size": hidden_size,
            "num_target_layers": num_target_layers,
            "optimizer": train_args.optimizer,
            "lr": train_args.lr,
            "hidden_states_dtype": train_args.hidden_states_dtype,
            "synthetic_data": is_synthetic,
            "data_path": train_args.data_path,
            "hidden_states_backend": train_args.hidden_states_backend,
            "num_workers": train_args.num_workers,
            "prefetch_factor": train_args.prefetch_factor,
            "fsdp_shard": train_args.fsdp_shard,
            "num_gpus_used": num_gpus_used,
            "warmup_steps": bench_args.warmup_steps,
            "measured_steps": bench_args.measured_steps,
            "seed": train_args.seed,
            "profiled": bool(bench_args.profile),
        },
        "memory": {
            "per_rank": per_rank_memory,
            "phases": phase_peaks,
        },
        "timing": timing_agg,
        "aggregate": aggregate,
    }

    if not bench_args.no_per_step:
        results["per_step"] = [
            {"step": i, **p} for i, p in enumerate(measured_profiles)
        ]

    if kernel_breakdown:
        results["kernels"] = kernel_breakdown

    # --- Write results (rank 0 only) ---
    if rank == 0:
        output_path = Path(bench_args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults written to {output_path}")
        _print_summary(results)

    # --- Cleanup ---
    shutdown_dataloader_workers(train_loader)
    del trainer, train_loader, draft_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    maybe_destroy_distributed()

    return results


def _print_summary(results: dict) -> None:
    """Print a compact summary of benchmark results to stdout."""
    timing = results["timing"]
    memory = results["memory"]

    hdr = (
        f"{'Metric':<16} {'Mean':>10} {'Std':>10} "
        f"{'95% CI':>20} {'Min':>10} {'Max':>10}"
    )
    print(f"\n{hdr}")
    print("-" * len(hdr))
    for key, stats in timing.items():
        ci = f"[{stats['ci95_lower']:.2f}, {stats['ci95_upper']:.2f}]"
        print(
            f"{key:<16} {stats['mean']:>10.2f} "
            f"{stats['std']:>10.2f} "
            f"{ci:>20} "
            f"{stats['min']:>10.2f} {stats['max']:>10.2f}"
        )

    aggregate = results["aggregate"]
    print(
        "\nEffective rank-0 throughput: "
        f"{aggregate['effective_rank0_tokens_per_s']:.2f} tokens/s "
        f"over {aggregate['measured_time_s']:.2f} s"
    )
    print("\nPeak memory (per rank):")
    for r in memory["per_rank"]:
        print(
            f"  rank {r['rank']}: {r['peak_allocated_mb']:.1f} MB allocated, "
            f"{r['peak_reserved_mb']:.1f} MB reserved"
        )
    if memory["phases"]:
        phases = ", ".join(f"{p}={mb:.1f}" for p, mb in memory["phases"].items())
        print(f"  phase peaks (MB allocated): {phases}")


# ---------------------------------------------------------------------------
# Compare
# ---------------------------------------------------------------------------


def _load_result(path: str) -> dict:
    """Read a result file, refusing one this script would misread.

    The schema moves between versions -- which timing marks exist, how memory
    is shaped -- and a reader that guesses at an older one prints numbers that
    look right and are not.
    """
    with open(path) as f:
        results = json.load(f)
    version = results.get("benchmark_version")
    if version != BENCHMARK_VERSION:
        raise SystemExit(
            f"Incompatible benchmark version: {path} reports {version}, this "
            f"script reads {BENCHMARK_VERSION}. Re-run the benchmark with this "
            "version of the script to report on it."
        )
    return results


def _nested_get(d: dict, key_path: str):
    """Traverse a dotted key path into a nested dict."""
    for part in key_path.split("."):
        if not isinstance(d, dict):
            return None
        d = d.get(part)  # type: ignore[assignment]
    return d


def _get_gpu_name(result: dict) -> str:
    """Extract the first GPU name from a result dict."""
    info = result.get("provenance", {}).get("gpu_info")
    if info:
        return info[0].get("name", "unknown")
    return "unknown"


def _get_effective_throughput(result: dict) -> float | None:
    aggregate = result.get("aggregate", {})
    value = aggregate.get("effective_rank0_tokens_per_s")
    if value is not None:
        return float(value)
    profiles = result.get("per_step")
    if profiles:
        return compute_aggregate_throughput(profiles)["effective_rank0_tokens_per_s"]
    return None


def _welch_test(per_step_a, per_step_b, key):
    """Welch's t-test and Cohen's d for a metric between two runs.

    Returns (t_stat, p_value, cohens_d) or None if per-step data is missing.
    """
    from scipy import stats as sp_stats  # noqa: PLC0415

    if not per_step_a or not per_step_b:
        return None
    a = [s[key] for s in per_step_a]
    b = [s[key] for s in per_step_b]
    min_samples = 2
    if len(a) < min_samples or len(b) < min_samples:
        return None
    t_stat, p_value = sp_stats.ttest_ind(a, b, equal_var=False)
    # Two runs that each reported one constant value leave the test undefined
    # and scipy says so with a nan. Report it as no test rather than let the
    # callers print "p=nan" as though it were a result.
    if not math.isfinite(p_value):
        return None
    # Cohen's d (pooled)
    na, nb = len(a), len(b)
    va = statistics.variance(a)
    vb = statistics.variance(b)
    sp = math.sqrt(((na - 1) * va + (nb - 1) * vb) / (na + nb - 2))
    cohens_d = (statistics.mean(b) - statistics.mean(a)) / sp if sp > 0 else 0.0
    return t_stat, p_value, cohens_d


def _fmt_welch(per_step_a, per_step_b, key):
    """Format Welch's t-test result for a single metric."""
    result = _welch_test(per_step_a, per_step_b, key)
    if not result:
        return f" {'n/a':>10} {'n/a':>10}"
    _, p_val, d_val = result
    p_threshold = 0.001
    p_fmt = f"{p_val:.2e}" if p_val < p_threshold else f"{p_val:.4f}"
    return f" {p_fmt:>10} {d_val:>+10.3f}"


# Config differences that make a before/after number mean something other than
# what it looks like. Two runs that disagree on any of these are measuring
# different work, so the delta between them is not a speedup.
_COMPARABILITY_CHECKS = (
    ("config.speculator_type", "Speculator type"),
    ("config.hidden_size", "Hidden size"),
    ("config.total_seq_len", "Sequence length"),
    ("config.num_gpus_used", "GPU count"),
    ("config.fsdp_shard", "FSDP shard"),
    ("config.optimizer", "Optimizer"),
    ("config.hidden_states_dtype", "Dtype"),
)


def _comparability_warnings(baseline, candidate) -> list[tuple[str, str, str]]:
    """(label, baseline value, candidate value) for each disagreement."""
    found = [
        (label, _nested_get(baseline, key_path), _nested_get(candidate, key_path))
        for key_path, label in _COMPARABILITY_CHECKS
        if _nested_get(baseline, key_path) != _nested_get(candidate, key_path)
    ]
    gpu_a, gpu_b = _get_gpu_name(baseline), _get_gpu_name(candidate)
    if gpu_a != gpu_b:
        found.append(("GPU", gpu_a, gpu_b))
    return found


def compare_benchmarks(baseline_path: str, candidate_path: str) -> None:
    """Load two result files and print a comparison table."""
    with open(baseline_path) as f:
        baseline = json.load(f)
    with open(candidate_path) as f:
        candidate = json.load(f)

    vb = baseline["benchmark_version"]
    vc = candidate["benchmark_version"]
    if vb != vc:
        raise SystemExit(
            f"Incompatible benchmark versions: baseline={vb}, candidate={vc}. "
            "Results are only comparable across runs of the same version of "
            "the benchmark script."
        )

    # --- Comparability warnings ---
    warnings_found = _comparability_warnings(baseline, candidate)
    if warnings_found:
        print("COMPARABILITY WARNINGS:")
        for label, val_a, val_b in warnings_found:
            print(f"  {label}: {val_a} vs {val_b}")

    # --- Header ---
    sha_a = baseline.get("provenance", {}).get("git_sha", "unknown")[:12]
    sha_b = candidate.get("provenance", {}).get("git_sha", "unknown")[:12]
    print(f"\nBaseline:  {baseline_path}")
    print(f"  Git SHA: {sha_a}")
    print(f"Candidate: {candidate_path}")
    print(f"  Git SHA: {sha_b}")

    _print_timing_comparison(baseline, candidate)

    effective_a = _get_effective_throughput(baseline)
    effective_b = _get_effective_throughput(candidate)
    if effective_a is not None and effective_b is not None:
        delta = effective_b - effective_a
        pct = (delta / effective_a * 100) if effective_a != 0 else 0
        print(
            "\nEffective rank-0 throughput: "
            f"{effective_a:.2f} -> {effective_b:.2f} tokens/s "
            f"({delta:+.2f}, {pct:+.1f}%)"
        )

    _print_memory_comparison(baseline, candidate)
    _print_kernel_comparison(baseline, candidate)


def _kernel_diff_rows(baseline, candidate) -> list[tuple[str, float, float]]:
    """(name, baseline ms/step, candidate ms/step) ranked by absolute change.

    This is what localizes a regression or a win to the kernel that caused it:
    the phase timings only say "forward got slower", the kernel rows say which
    launch did it. Kernels present on only one side (a different attention
    backend, a renamed Triton kernel) show up with the missing side at zero.
    Empty unless both runs were profiled.
    """
    ka = (baseline.get("kernels") or {}).get("kernels")
    kb = (candidate.get("kernels") or {}).get("kernels")
    if not ka or not kb:
        return []
    base_ms = {k["name"]: k["ms_per_step"] for k in ka}
    cand_ms = {k["name"]: k["ms_per_step"] for k in kb}
    names = sorted(
        set(base_ms) | set(cand_ms),
        key=lambda n: -abs(cand_ms.get(n, 0.0) - base_ms.get(n, 0.0)),
    )
    return [(n, base_ms.get(n, 0.0), cand_ms.get(n, 0.0)) for n in names]


def _print_kernel_comparison(baseline, candidate):
    """Per-kernel GPU time diff, when both runs were profiled."""
    rows = _kernel_diff_rows(baseline, candidate)
    if not rows:
        return

    print(
        f"\n{'GPU kernel (ms/step, self time)':<54}"
        f"{'Baseline':>10}{'Candidate':>11}{'Delta':>10}{'Delta %':>10}"
    )
    print("-" * 95)
    for name, a, b in rows[:KERNEL_TOP_N]:
        delta = b - a
        pct = f"{delta / a * 100:>+9.1f}%" if a else f"{'n/a':>10}"
        label = _short_kernel_name(name)
        print(f"{label:<54}{a:>10.2f}{b:>11.2f}{delta:>+10.2f}{pct}")

    total_a = baseline["kernels"].get("total_device_ms_per_step", 0)
    total_b = candidate["kernels"].get("total_device_ms_per_step", 0)
    delta = total_b - total_a
    pct = (delta / total_a * 100) if total_a else 0
    print(
        f"{'TOTAL device time':<54}{total_a:>10.2f}{total_b:>11.2f}"
        f"{delta:>+10.2f}{pct:>+9.1f}%"
    )
    print(
        "\n  Ranked by absolute change. Device-side durations, so the "
        "synchronisation overhead --profile adds does not distort them."
    )


def _print_timing_comparison(baseline, candidate):
    """Print the timing comparison table with optional significance tests."""
    per_step_a = baseline.get("per_step", [])
    per_step_b = candidate.get("per_step", [])
    has_stats = bool(per_step_a and per_step_b)

    col_w = 26
    stat_cols = f"{'p-value':>10} {'Cohen d':>10}" if has_stats else ""
    print(
        f"\n{'Metric':<16} "
        f"{'Baseline (mean +/- std)':<{col_w}} "
        f"{'Candidate (mean +/- std)':<{col_w}} "
        f"{'Delta':>10} {'Delta %':>10} {stat_cols}"
    )
    line_w = 16 + col_w * 2 + 22 + (22 if has_stats else 0)
    print("-" * line_w)

    for key, ba in baseline["timing"].items():
        ca = candidate["timing"][key]
        delta = ca["mean"] - ba["mean"]
        pct = (delta / ba["mean"] * 100) if ba["mean"] != 0 else 0

        ba_str = f"{ba['mean']:>8.2f} +/- {ba['std']:<6.2f}"
        ca_str = f"{ca['mean']:>8.2f} +/- {ca['std']:<6.2f}"
        stat_str = _fmt_welch(per_step_a, per_step_b, key) if has_stats else ""
        print(
            f"{key:<16} {ba_str:<{col_w}} {ca_str:<{col_w}} "
            f"{delta:>+10.2f} {pct:>+9.1f}%{stat_str}"
        )

    if has_stats:
        print(
            "\n  p-value: Welch's t-test (two-tailed). "
            "Cohen's d: pooled effect size "
            "(|d|<0.2 negligible, 0.2-0.5 small, "
            "0.5-0.8 medium, >0.8 large)."
        )


def _print_memory_comparison(baseline, candidate):
    """Memory comparison: per-rank peaks and per-phase allocated peaks."""
    print(f"\n{'Memory':<24} {'Baseline':>12} {'Candidate':>12} {'Delta':>12}")
    print("-" * 62)
    b_ranks = {r["rank"]: r for r in baseline["memory"]["per_rank"]}
    c_ranks = {r["rank"]: r for r in candidate["memory"]["per_rank"]}
    rows: list[tuple[str, float, float]] = []
    for rank in sorted(b_ranks.keys() & c_ranks.keys()):
        for key in ("peak_allocated_mb", "peak_reserved_mb"):
            rows.append(
                (f"rank{rank} {key[:-3]}", b_ranks[rank][key], c_ranks[rank][key])
            )
    b_phases = baseline["memory"].get("phases", {})
    c_phases = candidate["memory"].get("phases", {})
    for phase in sorted(b_phases.keys() & c_phases.keys()):
        rows.append((f"phase {phase}", b_phases[phase], c_phases[phase]))
    for label, ba_val, ca_val in rows:
        print(
            f"{label:<24} {ba_val:>9.1f} MB {ca_val:>9.1f} MB "
            f"{ca_val - ba_val:>+9.1f} MB"
        )


# ---------------------------------------------------------------------------
# Visualize
# ---------------------------------------------------------------------------

# The phases a step is drawn as, in the order they happen. ``fetch_ms`` is
# deliberately absent: it spans queue + H2D, so charting it alongside them
# would double-count the wait. ``bwd_ms`` likewise contains ``clip_ms``, which
# the builders below subtract out before plotting.
_PHASE_COLORS = {
    "queue_ms": "#636EFA",
    "h2d_ms": "#AB63FA",
    "fwd_ms": "#00CC96",
    "bwd_ms": "#FFA15A",
    "clip_ms": "#FECB52",
    "opt_ms": "#EF553B",
    "other_ms": "#B6B6B6",
}

_PHASE_LABELS = {
    "queue_ms": "DataLoader wait",
    "h2d_ms": "H2D transfer",
    "fwd_ms": "Forward",
    "bwd_ms": "Backward",
    "clip_ms": "Grad clip",
    "opt_ms": "Optimizer",
    "other_ms": "Other",
}

_MEMORY_MARK_LABELS = {
    "queue": "After queue",
    "fetch": "After H2D",
    "fwd": "After forward",
    "pre_clip": "After backward",
    "bwd": "After clip",
    "opt": "After optimizer",
}


def _phase_series(per_step) -> list[tuple[str, str, list[float]]]:
    """Per-step series for every drawn phase, with grad clip split out of bwd.

    The trainer times ``bwd_ms`` from the end of forward to the end of the
    clip, so the clip sits inside it; stacking both as measured would count it
    twice.
    """
    series = []
    for key, label in _PHASE_LABELS.items():
        if key == "bwd_ms":
            values = [s["bwd_ms"] - s["clip_ms"] for s in per_step]
        elif key == "other_ms":
            # The named phases do not tile the step: the trainer times
            # ``queue_ms`` to the end of the queue wait but starts ``h2d_ms``
            # at ``pre_h2d``, so the recovery bookkeeping in between belongs to
            # no phase. Draw it explicitly so the stack sums to ``step_ms``.
            values = [
                max(
                    s["step_ms"]
                    - (
                        s["queue_ms"]
                        + s["h2d_ms"]
                        + s["fwd_ms"]
                        + s["bwd_ms"]
                        + s["opt_ms"]
                    ),
                    0.0,
                )
                for s in per_step
            ]
        else:
            values = [s[key] for s in per_step]
        series.append((key, label, values))
    return series


def _build_timing_traces(per_step):
    """Build stacked area traces for timing breakdown."""
    import plotly.graph_objects as go  # noqa: PLC0415

    steps = list(range(len(per_step)))
    return [
        go.Scatter(
            x=steps,
            y=values,
            name=label,
            mode="lines",
            stackgroup="timing",
            line={"width": 0.5, "color": _PHASE_COLORS[key]},
        )
        for key, label, values in _phase_series(per_step)
    ]


def _build_memory_traces(per_step):
    """Build line traces for memory at phase boundaries."""
    import plotly.graph_objects as go  # noqa: PLC0415

    steps = list(range(len(per_step)))
    traces = []
    for mark, label in _MEMORY_MARK_LABELS.items():
        values = [s.get("memory_mb", {}).get(mark) for s in per_step]
        if not any(v is not None for v in values):
            continue
        traces.append(go.Scatter(x=steps, y=values, mode="lines", name=label))
    return traces


def _get_phase_means(per_step):
    """Mean and std of each phase duration over the measured steps."""
    return [
        (
            key,
            label,
            statistics.mean(values),
            statistics.stdev(values) if len(values) > 1 else 0.0,
        )
        for key, label, values in _phase_series(per_step)
    ]


def _build_timing_fig(per_step):
    """Mean time per training phase as a horizontal bar chart."""
    import plotly.graph_objects as go  # noqa: PLC0415

    phase_stats = _get_phase_means(per_step)
    total_ms = sum(mean for _, _, mean, _ in phase_stats)

    fig = go.Figure()
    for key, label, mean_val, std_val in reversed(phase_stats):
        pct = mean_val / total_ms * 100 if total_ms > 0 else 0
        fig.add_trace(
            go.Bar(
                y=[label],
                x=[mean_val],
                error_x={"type": "data", "array": [std_val], "visible": True},
                orientation="h",
                name=label,
                marker_color=_PHASE_COLORS.get(key, "#999"),
                text=f"{mean_val:.1f}ms ({pct:.1f}%)",
                textposition="auto",
                showlegend=False,
            )
        )
    fig.update_layout(barmode="stack", yaxis={"categoryorder": "array"})
    return fig


def _timing_sub(per_step) -> str:
    """Key numbers for the timing section: step time, throughput, spread."""
    step_stats = compute_statistics([s["step_ms"] for s in per_step])
    # Use the same time-weighted figure the Markdown summary and ``compare``
    # report. ``mean(tokens_per_s)`` would overweight short, fast steps and
    # make the HTML headline disagree with every other view of the same run.
    eff_tps = compute_aggregate_throughput(per_step)["effective_rank0_tokens_per_s"]
    return (
        f"{step_stats['mean']:.1f} ms/step "
        f"[{step_stats['ci95_lower']:.1f}, {step_stats['ci95_upper']:.1f}] · "
        f"{eff_tps:.0f} tok/s effective · "
        f"n={step_stats['count']} steps · brackets are 95% CI, "
        "error bars ±1 std"
    )


def _build_memory_fig(per_step):
    """Mean allocated memory at each phase boundary, with std error bars."""
    import plotly.graph_objects as go  # noqa: PLC0415

    means = []
    stds = []
    for mark in _MEMORY_MARK_LABELS:
        vals = [
            s["memory_mb"][mark]
            for s in per_step
            if "memory_mb" in s and mark in s["memory_mb"]
        ]
        means.append(statistics.mean(vals) if vals else 0)
        stds.append(statistics.stdev(vals) if len(vals) > 1 else 0.0)

    fig = go.Figure(
        go.Bar(
            x=list(_MEMORY_MARK_LABELS.values()),
            y=means,
            error_y={"type": "data", "array": stds, "visible": True},
            marker_color="#636EFA",
            text=[f"{m:.0f}" for m in means],
            textposition="outside",
            # The section margins are tight; without this the label on the
            # tallest bar is clipped at the plot edge.
            cliponaxis=False,
            showlegend=False,
        )
    )
    fig.update_yaxes(rangemode="tozero")
    return fig


def _memory_sub(results) -> str:
    """Key numbers for the memory section: the peaks torch reports per rank.

    The bars come from rank 0's per-step marks, so the peaks are qualified
    with the rank they describe rather than presented as a run-wide figure.
    """
    per_rank = results["memory"]["per_rank"]
    alloc = max(r["peak_allocated_mb"] for r in per_rank)
    reserved = max(r["peak_reserved_mb"] for r in per_rank)
    scope = "rank 0" if len(per_rank) == 1 else f"max over {len(per_rank)} ranks"
    return (
        f"{alloc:.0f} MB peak allocated · {reserved:.0f} MB peak reserved "
        f"({scope}) · bars are rank 0's mean at each phase boundary, "
        "error bars ±1 std"
    )


def _build_appendix_fig(per_step):
    """Build per-step raw data charts for the appendix."""
    import plotly.graph_objects as go  # noqa: PLC0415
    from plotly.subplots import make_subplots  # noqa: PLC0415

    steps = list(range(len(per_step)))
    has_memory = any("memory_mb" in s for s in per_step)
    n_rows = 4 if has_memory else 3

    titles = [
        "Per-Step Timing Breakdown (ms)",
        "Per-Step Throughput (tokens/s)",
        "Per-Step Time (ms)",
    ]
    if has_memory:
        titles.insert(2, "Per-Step GPU Memory Allocated (logical) (MB)")

    fig = make_subplots(
        rows=n_rows,
        cols=1,
        subplot_titles=titles,
        vertical_spacing=0.06,
    )

    for trace in _build_timing_traces(per_step):
        fig.add_trace(trace, row=1, col=1)

    fig.add_trace(
        go.Scatter(
            x=steps,
            y=[s["tokens_per_s"] for s in per_step],
            mode="lines+markers",
            name="tokens/s",
            marker={"size": 3},
            showlegend=False,
        ),
        row=2,
        col=1,
    )
    fig.update_yaxes(rangemode="tozero", row=2, col=1)

    if has_memory:
        for trace in _build_memory_traces(per_step):
            fig.add_trace(trace, row=3, col=1)
        fig.update_yaxes(rangemode="tozero", row=3, col=1)

    step_row = n_rows
    fig.add_trace(
        go.Scatter(
            x=steps,
            y=[s["step_ms"] for s in per_step],
            mode="lines+markers",
            name="step_ms",
            marker={"size": 3},
            line={"color": "#19D3F3"},
            showlegend=False,
        ),
        row=step_row,
        col=1,
    )
    fig.update_yaxes(rangemode="tozero", row=step_row, col=1)
    fig.update_xaxes(title_text="Step", row=n_rows, col=1)

    return fig, n_rows


def _kernel_markdown_lines(kernels: dict | None) -> list[str]:
    """Markdown table of the top GPU kernels, or nothing when not profiled."""
    if not kernels or not kernels.get("kernels"):
        return []
    total = kernels.get("total_device_ms_per_step", 0)
    lines = [
        f"## Appendix: Top {KERNEL_TOP_N} GPU Kernels",
        "",
        (
            f"{total:.1f} ms GPU time/step · {len(kernels['kernels'])} kernels"
            " · self time, averaged over the profiled steps"
        ),
        "",
        "| Kernel | ms/step | % GPU | calls/step |",
        "|--------|---------|-------|------------|",
    ]
    for k in kernels["kernels"][:KERNEL_TOP_N]:
        pct = k["ms_per_step"] / total * 100 if total else 0
        lines.append(
            f"| `{_short_kernel_name(k['name'])}` | {k['ms_per_step']:.2f}"
            f" | {pct:.1f}% | {k['calls_per_step']:.1f} |"
        )
    lines.append("")
    return lines


def _write_markdown_summary(results: dict, md_path: Path) -> None:
    """Write a markdown summary of benchmark results."""
    cfg = results["config"]
    prov = results["provenance"]
    mem = results["memory"]
    timing = results["timing"]
    per_step = results.get("per_step")
    aggregate = results["aggregate"]

    spec = cfg.get("speculator_type", "?")
    verifier = cfg.get("verifier_name_or_path", "?")
    gpu_desc = f"{cfg.get('num_gpus_used', 1)} x {_get_gpu_name(results)}"
    steps_desc = (
        f"{cfg.get('warmup_steps', '?')} warmup"
        f" + {cfg.get('measured_steps', '?')} measured"
    )
    lines = [
        f"# Benchmark: {spec} / {verifier}",
        "",
        f"- **seq_len**: {cfg.get('total_seq_len', '?')}",
        f"- **GPUs**: {gpu_desc}",
        f"- **optimizer**: {cfg.get('optimizer', '?')}",
        f"- **dtype**: {cfg.get('hidden_states_dtype', '?')}",
        f"- **synthetic**: {cfg.get('synthetic_data', '?')}",
        f"- **steps**: {steps_desc}",
        f"- **git**: `{prov.get('git_sha', 'unknown')[:12]}`",
        f"- **date**: {prov.get('timestamp', '?')[:10]}",
        "",
        "## Timing",
        "",
        "| Metric | Mean | Std | 95% CI | Min | Max |",
        "|--------|------|-----|--------|-----|-----|",
    ]

    for key, stats in timing.items():
        lines.append(
            f"| {key} | {stats['mean']:.2f} | {stats['std']:.2f}"
            f" | [{stats['ci95_lower']:.2f}, {stats['ci95_upper']:.2f}]"
            f" | {stats['min']:.2f} | {stats['max']:.2f} |"
        )

    eff = aggregate["effective_rank0_tokens_per_s"]
    dur = aggregate["measured_time_s"]
    lines += [
        "",
        f"**Effective throughput**: {eff:.0f} tokens/s over {dur:.1f}s",
        "",
        "## Memory",
        "",
        "| Rank | Peak allocated (MB) | Peak reserved (MB) |",
        "|------|---------------------|--------------------|",
    ]
    for r in mem["per_rank"]:
        lines.append(
            f"| {r['rank']} | {r['peak_allocated_mb']:.0f}"
            f" | {r['peak_reserved_mb']:.0f} |"
        )
    if mem["phases"]:
        phases = ", ".join(f"{p}={mb:.0f}" for p, mb in mem["phases"].items())
        lines += ["", f"**Phase peaks** (MB allocated, rank 0): {phases}"]
    lines.append("")

    # Phase breakdown percentages
    if per_step:
        phase_stats = _get_phase_means(per_step)
        total_ms = sum(m for _, _, m, _ in phase_stats)
        if total_ms > 0:
            lines += ["## Phase Breakdown", ""]
            lines.append("| Phase | Mean (ms) | % |")
            lines.append("|-------|-----------|---|")
            for _, label, mean_val, _ in phase_stats:
                pct = mean_val / total_ms * 100
                lines.append(f"| {label} | {mean_val:.1f} | {pct:.1f}% |")
            lines.append("")

    lines += _kernel_markdown_lines(results.get("kernels"))

    md_path.write_text("\n".join(lines))
    print(f"Markdown summary written to {md_path}")


def _build_kernel_pie(kernels: dict):
    """Pie of device time per kernel, with the sub-2% tail merged into "Other"."""
    import plotly.graph_objects as go  # noqa: PLC0415

    total = kernels.get("total_device_ms_per_step", 0) or 1.0
    labels: list[str] = []
    values: list[float] = []
    hovers: list[str] = []
    tail_ms = 0.0
    tail_n = 0

    for k in kernels["kernels"]:
        if k["ms_per_step"] / total * 100 < KERNEL_PIE_MIN_PCT:
            tail_ms += k["ms_per_step"]
            tail_n += 1
            continue
        # Plotly renders legend text as HTML, so the name is escaped here too.
        labels.append(html_escape(_short_kernel_name(k["name"])))
        values.append(k["ms_per_step"])
        # The legend label is clipped, so the full name lives in the hover.
        hovers.append(
            f"<b>{_wrap_kernel_name(k['name'])}</b><br>"
            f"{k['ms_per_step']:.2f} ms/step · "
            f"{k['calls_per_step']:.1f} calls/step"
        )

    if tail_n:
        labels.append(f"Other ({tail_n} kernels)")
        values.append(tail_ms)
        hovers.append(
            f"<b>{tail_n} kernels</b>, each under {KERNEL_PIE_MIN_PCT:.0f}% "
            f"of device time<br>{tail_ms:.2f} ms/step combined"
        )

    fig = go.Figure(
        go.Pie(
            labels=_unique_labels(labels),
            values=values,
            customdata=hovers,
            hovertemplate="%{customdata}<extra></extra>",
            texttemplate="%{percent}",
            textposition="inside",
            insidetextorientation="horizontal",
            sort=False,  # already sorted by time; keep "Other" last
            direction="clockwise",
            marker={"line": {"color": "#fff", "width": 1}},
        )
    )
    fig.update_layout(
        legend={"font": {"family": "ui-monospace, monospace", "size": 10}}
    )
    return fig


def _kernel_name_cell(name: str) -> str:
    """Kernel name as HTML, carrying the full name in a tooltip when clipped.

    Deliberately a ``span.kname`` rather than a ``<code>``: viewers and browser
    extensions routinely restyle ``code`` -- a dark chip behind every cell is
    the common one -- and a class we style ourselves cannot be hijacked that
    way. Elided names get a dotted underline and a help cursor so it is visible
    that there is more to hover for; whole ones stay plain.
    """
    short = _short_kernel_name(name)
    if short == name:
        return f'<span class="kname">{html_escape(name)}</span>'
    return (
        f'<span class="kname trunc" title="{html_escape(name)}">'
        f"{html_escape(short)}</span>"
    )


# ---------------------------------------------------------------------------
# GitHub PR comment
# ---------------------------------------------------------------------------

# A CI job finds its own previous comment by this marker and edits it in place,
# rather than stacking a new comment on every push.
COMMENT_MARKER_ID = "speculators-benchmark"
# Kernel rows in the comment. Fewer than the HTML report: a PR comment is read
# in a scroll, and GitHub hard-truncates very long ones.
COMMENT_KERNEL_N = 8


def _marker_label(label: str) -> str:
    """The label reduced to what is safe inside an HTML comment and a grep.

    A label is free text from whoever set up the job, and ``-->`` in it would
    close the marker early -- leaving the rest of it as visible text and the
    comment unfindable on the next run.
    """
    return re.sub(r"[^A-Za-z0-9._]+", "-", label).strip("-") or "unlabeled"


def _comment_marker(label: str | None = None) -> str:
    """The hidden marker a job greps for to find the comment it should update."""
    suffix = f":{_marker_label(label)}" if label else ""
    return f"<!-- {COMMENT_MARKER_ID}{suffix} -->"


def _md_code(value) -> str:
    """A value as a code span, where markdown renders it as literal text.

    This is what keeps the paths, hostnames and model names out of the result
    file from being read as markup. A backtick in the value would close the
    span early and let the rest back out, so it is the one character that
    cannot be carried through as written.
    """
    text = str(value).replace("`", "'")
    return f"`{text}`"


# Changes smaller than this are reported without a win/loss marker. A green
# tick on a 0.5% move reads as a result when it is run-to-run noise.
MD_NEUTRAL_PCT = 1.0


def _md_delta(before: float, after: float, *, lower_is_better=True) -> str:
    """Signed percentage change, marked when it is big enough to mean anything."""
    if not before:
        return "n/a"
    pct = (after - before) / before * 100
    if abs(pct) < MD_NEUTRAL_PCT:
        return f"{pct:+.1f}%"
    improved = (pct < 0) if lower_is_better else (pct > 0)
    return f"{'✅' if improved else '🔴'} **{pct:+.1f}%**"


def _md_significance(per_step_a, per_step_b, key: str) -> str:
    """Welch p-value and Cohen's d as a compact cell."""
    result = _welch_test(per_step_a, per_step_b, key)
    if not result:
        return "—"
    _, p_val, d_val = result
    p_txt = "p<0.001" if p_val < 0.001 else f"p={p_val:.3f}"  # noqa: PLR2004
    return f"{p_txt}, d={d_val:+.1f}"


def _comment_timing_table(base: dict, cand: dict | None) -> list[str]:
    """Timing table: before/after with significance, or a single-run summary.

    The rows are whatever the run reported rather than a list kept here, so a
    timing mark added to the trainer reaches the comment without this having to
    learn its name.
    """
    if cand is None:
        lines = [
            "| Metric | Mean | 95% CI | Std |",
            "|---|--:|:-:|--:|",
        ]
        for key, stats in base["timing"].items():
            ci = f"[{stats['ci95_lower']:.2f}, {stats['ci95_upper']:.2f}]"
            lines.append(
                f"| `{key}` | {stats['mean']:.2f} | {ci} | {stats['std']:.2f} |"
            )
        return lines

    per_a = base.get("per_step", [])
    per_b = cand.get("per_step", [])
    lines = [
        "| Metric | Before | After | Change | Significance |",
        "|---|--:|--:|--:|:-:|",
    ]
    for key, sa in base["timing"].items():
        sb = cand["timing"].get(key)
        if not sb:
            continue
        # Throughput is the one metric where up is the win.
        lower_better = key != "tokens_per_s"
        lines.append(
            f"| `{key}` | {sa['mean']:,.1f} | {sb['mean']:,.1f} |"
            f" {_md_delta(sa['mean'], sb['mean'], lower_is_better=lower_better)} |"
            f" {_md_significance(per_a, per_b, key)} |"
        )
    return lines


# Appended to every kernel table: the reason these numbers can be trusted even
# though the wall-clock timings from the same run cannot.
_KERNEL_CAVEAT = (
    "Device-side durations, so the synchronisation `--profile` adds does not "
    "distort them."
)


def _comment_kernel_lines(base: dict, cand: dict | None) -> list[str]:
    """Kernel section: a before/after diff table, or a single-run table and pie.

    A diff needs the numbers side by side, and a pie of one of the two sides
    would only invite reading the wrong one. With a single run there is no such
    ambiguity and the mix is the whole point, so the pie comes back.
    """
    rows = _kernel_diff_rows(base, cand) if cand else []
    if rows:
        total_a = base["kernels"].get("total_device_ms_per_step", 0)
        total_b = cand["kernels"].get("total_device_ms_per_step", 0)
        return [
            "",
            "### GPU kernel time (ms/step, self time)",
            "",
            "| Kernel | Before | After | Change |",
            "|---|--:|--:|--:|",
            *[
                f"| `{_short_kernel_name(name, 44)}` | {a:.2f} | {b:.2f} |"
                f" {_md_delta(a, b)} |"
                for name, a, b in rows[:COMMENT_KERNEL_N]
            ],
            (
                f"| **Total device time** | **{total_a:.2f}** | **{total_b:.2f}** |"
                f" {_md_delta(total_a, total_b)} |"
            ),
            "",
            f"<sub>Ranked by absolute change. {_KERNEL_CAVEAT}</sub>",
        ]

    # No baseline to diff against: show where the time goes in the one run.
    kernels = base.get("kernels") if cand is None else None
    if not kernels or not kernels.get("kernels"):
        return []
    total = kernels.get("total_device_ms_per_step", 0)
    return [
        "",
        f"### GPU kernel time — {total:.1f} ms/step total",
        "",
        "| Kernel | ms/step | % GPU | calls/step |",
        "|---|--:|--:|--:|",
        *[
            f"| `{_short_kernel_name(k['name'], 44)}` | {k['ms_per_step']:.2f} |"
            f" {(k['ms_per_step'] / total * 100 if total else 0):.1f}% |"
            f" {k['calls_per_step']:.1f} |"
            for k in kernels["kernels"][:COMMENT_KERNEL_N]
        ],
        "",
        f"<sub>{_KERNEL_CAVEAT}</sub>",
        "",
        *_mermaid_pie(kernels),
    ]


def _mermaid_pie(kernels: dict) -> list[str]:
    """Kernel mix as a mermaid pie, the chart dialect GitHub renders natively.

    Capped at ``COMMENT_KERNEL_N`` slices on top of the percentage floor: after
    a large win the surviving kernels bunch up just over the threshold, and a
    dozen near-equal wedges say nothing.
    """
    total = kernels.get("total_device_ms_per_step", 0) or 1.0
    slices: list[tuple[str, float]] = []
    tail_ms = 0.0
    tail_n = 0
    for k in kernels["kernels"]:
        over_floor = k["ms_per_step"] / total * 100 >= KERNEL_PIE_MIN_PCT
        if over_floor and len(slices) < COMMENT_KERNEL_N:
            # Mermaid delimits labels with double quotes and offers no escape.
            slices.append(
                (_short_kernel_name(k["name"], 40).replace('"', "'"), k["ms_per_step"])
            )
        else:
            tail_ms += k["ms_per_step"]
            tail_n += 1
    if tail_n:
        slices.append((f"Other ({tail_n} kernels)", tail_ms))

    # Mermaid sums same-named wedges just as plotly does, and elision can
    # collide two cuBLAS kernels onto one label.
    labels = _unique_labels([label for label, _ in slices])
    return [
        "```mermaid",
        "pie showData",
        f"    title Device time per kernel — {total:.1f} ms/step",
        *[
            f'    "{label}" : {ms:.2f}'
            for label, (_, ms) in zip(labels, slices, strict=True)
        ],
        "```",
    ]


def _comment_provenance(base: dict, cand: dict | None) -> list[str]:
    """Collapsed provenance: the commands, the environment, the config."""
    prov = (cand or base).get("provenance", {})
    cfg = (cand or base).get("config", {})

    commands = []
    for label, result in (("Before", base), ("After", cand)):
        cmd = (result or {}).get("provenance", {}).get("command")
        if cmd:
            prefix = f"# {label}\n" if cand else ""
            commands.append(f"{prefix}{cmd}")
    cmd_block = ["**Command**", "", "```bash", *commands, "```", ""] if commands else []

    env = [
        ("commit", _md_code(prov.get("git_sha", "unknown")[:12])),
        (
            "GPU",
            _md_code(f"{cfg.get('num_gpus_used', 1)} x {_get_gpu_name(cand or base)}"),
        ),
        ("torch", _md_code(prov.get("pytorch_version", "?"))),
        ("CUDA", _md_code(prov.get("cuda_version", "?"))),
        ("transformers", _md_code(prov.get("transformers_version", "?"))),
        ("speculators", _md_code(prov.get("speculators_version", "?"))),
        ("host", _md_code(prov.get("hostname", "?"))),
        (
            "run at",
            _md_code(prov.get("timestamp", "?")[:19].replace("T", " ") + " UTC"),
        ),
    ]
    if cand:
        sha_a = base.get("provenance", {}).get("git_sha", "unknown")[:12]
        sha_b = prov.get("git_sha", "unknown")[:12]
        if sha_a != sha_b:
            env[0] = ("commit", f"{_md_code(sha_a)} → {_md_code(sha_b)}")

    cfg_keys = (
        "speculator_type",
        "verifier_name_or_path",
        "total_seq_len",
        "hidden_size",
        "optimizer",
        "lr",
        "hidden_states_dtype",
        "synthetic_data",
        "fsdp_shard",
        "warmup_steps",
        "measured_steps",
        "seed",
    )
    return [
        "",
        "<details>",
        "<summary>Provenance</summary>",
        "",
        *cmd_block,
        "**Environment**",
        "",
        "| | |",
        "|---|---|",
        *[f"| {k} | {v} |" for k, v in env],
        "",
        "**Config**",
        "",
        "| | |",
        "|---|---|",
        *[f"| {k} | {_md_code(cfg[k])} |" for k in cfg_keys if k in cfg],
        "",
        "</details>",
    ]


def _peak_allocated_mb(result: dict) -> tuple[float, str]:
    """Peak allocated memory and the scope that one number covers.

    torch reports a peak per rank; the one a comment should carry is the worst
    of them, since that is the rank that decides whether the run fits. Which
    rank it came from is said out loud rather than left to be assumed run-wide.
    """
    per_rank = result["memory"]["per_rank"]
    peak = max(r["peak_allocated_mb"] for r in per_rank)
    scope = "rank 0" if len(per_rank) == 1 else f"max over {len(per_rank)} ranks"
    return peak, scope


def build_comment(base: dict, cand: dict | None, label: str | None = None) -> str:
    """Render benchmark results as markdown for a GitHub PR comment.

    GitHub strips scripts, so the interactive report cannot be embedded; this
    emits the same findings in what a comment does render -- tables and a
    collapsed provenance block.

    ``label`` names the variant. Config and commit are identical whenever the
    thing being measured is an uncommitted patch, so without it two comments on
    the same PR are indistinguishable. It also scopes the marker, giving each
    variant its own comment to update instead of one they fight over.
    """
    cfg = (cand or base).get("config", {})
    # Heading text comes from the result file and GitHub renders inline HTML in
    # a comment, so it is escaped rather than trusted the way the structural
    # markup below is.
    title = " · ".join(
        html_escape(str(part))
        for part in (
            (
                f"{cfg.get('speculator_type', '?')} / "
                f"{cfg.get('verifier_name_or_path', '?')}"
            ),
            f"seq_len {cfg.get('total_seq_len', '?')}",
            f"{cfg.get('num_gpus_used', 1)}x {_get_gpu_name(cand or base)}",
        )
    )
    named = f" — {html_escape(label)}" if label else ""
    lines = [
        _comment_marker(label),
        f"## Training benchmark{named}: {title}",
        "",
    ]

    # Before the headline, because the headline is the part that misleads: a
    # run against half the sequence length is "2x faster" and looks like a win.
    mismatches = _comparability_warnings(base, cand) if cand else []
    if mismatches:
        lines += [
            "> [!WARNING]",
            "> These runs measure different work, so the deltas below are not",
            "> a speedup:",
            *[f"> - {label_}: `{a}` vs `{b}`" for label_, a, b in mismatches],
            "",
        ]

    if cand:
        before = base["timing"]["step_ms"]["mean"]
        after = cand["timing"]["step_ms"]["mean"]
        tps_a = base["timing"]["tokens_per_s"]["mean"]
        tps_b = cand["timing"]["tokens_per_s"]["mean"]
        ratio = before / after if after else 0.0
        faster = ratio >= 1
        factor = ratio if faster else (1 / ratio if ratio else 0)
        lines.append(
            f"### {'⚡' if faster else '⚠️'} "
            f"{factor:.2f}× {'faster' if faster else 'slower'} — "
            f"{before:,.1f} → {after:,.1f} ms/step, "
            f"{tps_a:,.0f} → {tps_b:,.0f} tok/s"
        )
    else:
        step = base["timing"]["step_ms"]["mean"]
        tps = base["timing"]["tokens_per_s"]["mean"]
        lines.append(f"### {step:,.1f} ms/step · {tps:,.0f} tok/s")
    lines.append("")

    lines += _comment_timing_table(base, cand)

    mem_a, scope = _peak_allocated_mb(base)
    if cand:
        mem_b, scope = _peak_allocated_mb(cand)
        lines += [
            "",
            (
                f"Peak allocated ({scope}): {mem_a:,.0f} → {mem_b:,.0f} MB "
                f"({_md_delta(mem_a, mem_b)})"
            ),
        ]
    else:
        lines += ["", f"Peak allocated ({scope}): {mem_a:,.0f} MB"]

    lines += _comment_kernel_lines(base, cand)
    lines += _comment_provenance(base, cand)
    # The comment is itself an artifact, and a reader who wants to regenerate
    # or re-post it should not have to guess which invocation produced it.
    lines += ["", f"<sub>Generated by {_md_code(_invocation())}</sub>"]
    return "\n".join(lines) + "\n"


def comment_benchmark(
    result_path: str,
    baseline_path: str | None,
    output_path: str | None,
    label: str | None = None,
) -> None:
    """Write (or print) the PR-comment markdown for one or two result files."""
    cand_or_base = _load_result(result_path)
    base = cand_or_base
    cand = None
    if baseline_path:
        base = _load_result(baseline_path)
        cand = cand_or_base

    text = build_comment(base, cand, label)
    if output_path:
        Path(output_path).write_text(text)
        print(f"Comment markdown written to {output_path}", file=sys.stderr)
        print(
            f"Post it with:  gh pr comment <N> --body-file {output_path}",
            file=sys.stderr,
        )
    else:
        print(text)


def _section_html(title: str, sub: str, fig, height: int, *, with_js=False) -> str:
    """One report section: heading, a line of key data, then its chart.

    Every body section goes through here so the three of them stay visually
    identical; plotly's own subplot titles and annotations cannot match the
    surrounding page, which is why the figures carry neither.

    ``with_js`` inlines the plotly bundle and must be set on exactly one
    section. Inlined rather than pulled from the CDN: reports are read from
    air-gapped clusters and out of CI artifact archives, where a remote script
    tag leaves a page of blank boxes.
    """
    fig.update_layout(
        height=height,
        template="plotly_white",
        margin={"t": 30, "b": 40, "l": 20, "r": 20},
    )
    fig_html = fig.to_html(full_html=False, include_plotlyjs=with_js)
    return f"""
<h2 class="sec">{title}</h2>
<p class="sub">{sub}</p>
{fig_html}"""


def _kernel_html(kernels: dict | None) -> tuple[str, str]:
    """Kernel-time pie for the body and top-N table for the appendix.

    Returns ``("", "")`` when the run was not profiled.
    """
    if not kernels or not kernels.get("kernels"):
        return "", ""
    entries = kernels["kernels"]
    total = kernels.get("total_device_ms_per_step", 0)
    shown = sum(
        1
        for k in entries
        if k["ms_per_step"] / (total or 1) * 100 >= KERNEL_PIE_MIN_PCT
    )
    body = _section_html(
        "GPU Kernel Breakdown",
        f"{total:.1f} ms GPU time/step · {len(entries)} kernels · "
        f"{shown} at or above {KERNEL_PIE_MIN_PCT:.0f}%, rest grouped as "
        '"Other" · hover for full name, ms/step and calls/step',
        _build_kernel_pie(kernels),
        height=420,
    )

    rows = "".join(
        f"<tr><td>{_kernel_name_cell(k['name'])}</td>"
        f"<td>{k['ms_per_step']:.2f}</td>"
        f"<td>{(k['ms_per_step'] / total * 100 if total else 0):.1f}%</td>"
        f"<td>{k['calls_per_step']:.1f}</td></tr>"
        for k in entries[:KERNEL_TOP_N]
    )
    appendix = f"""
<details>
<summary>Appendix: Top {KERNEL_TOP_N} GPU Kernels</summary>
<table class="kernels">
<thead><tr><th>Kernel</th><th>ms/step</th><th>% GPU</th><th>calls/step</th></tr>
</thead><tbody>{rows}</tbody></table>
</details>"""
    return body, appendix


def _report_paths(result_path: str, output_path: str | None) -> tuple[Path, Path]:
    """Resolve the input, HTML and Markdown paths, refusing to collide.

    The two outputs are derived from one name, so a ``--output`` that already
    ends in ``.md`` -- or a result file named ``.html`` -- would otherwise have
    one write silently clobber another file the command was asked to read or
    produce.
    """
    json_path = Path(result_path)
    html_path = (
        Path(output_path)
        if output_path
        else json_path.with_name(f"{json_path.stem}_report.html")
    )
    md_path = html_path.with_suffix(".md")
    paths = (json_path, html_path, md_path)
    if len({p.resolve() for p in paths}) != len(paths):
        raise SystemExit(
            f"Report paths collide: input {json_path}, HTML {html_path}, "
            f"Markdown {md_path}. Pass --output with a distinct .html name."
        )
    return html_path, md_path


def report_benchmark(result_path: str, output_path: str | None = None) -> None:
    """Generate an interactive HTML report from benchmark results."""
    try:
        import plotly  # noqa: PLC0415, F401
    except ImportError:
        print("plotly is required: pip install plotly", file=sys.stderr)
        sys.exit(1)

    results = _load_result(result_path)

    per_step = results.get("per_step")
    if not per_step:
        print(
            "No per-step data found. Re-run benchmark without --no-per-step.",
            file=sys.stderr,
        )
        sys.exit(1)

    html_path, md_path = _report_paths(result_path, output_path)

    # Everything below comes from the result file, which the report renders as
    # HTML; escape it rather than trust whoever produced the JSON.
    cfg = results["config"]
    prov = results["provenance"]
    heading = " &middot; ".join(
        html_escape(str(part))
        for part in (
            cfg.get("speculator_type", "?"),
            cfg.get("verifier_name_or_path", "?"),
            f"seq_len={cfg.get('total_seq_len', '?')}",
            f"{cfg.get('num_gpus_used', 1)} GPU(s)",
        )
    )
    sub = " &middot; ".join(
        html_escape(str(part))
        for part in (
            f"{cfg.get('measured_steps', '?')} measured steps",
            _get_gpu_name(results),
            prov.get("git_sha", "unknown")[:12],
        )
    )

    # --- Body sections: timing, memory, kernels ---
    timing_html = _section_html(
        "Step Timing Breakdown",
        _timing_sub(per_step),
        _build_timing_fig(per_step),
        height=340,
        with_js=True,  # first figure on the page carries the plotly bundle
    )
    memory_html = ""
    if any("memory_mb" in s for s in per_step):
        memory_html = _section_html(
            "GPU Memory Allocated (logical)",
            _memory_sub(results),
            _build_memory_fig(per_step),
            height=340,
        )

    # --- Appendix: per-step raw data ---
    appendix_fig, n_rows = _build_appendix_fig(per_step)
    appendix_fig.update_layout(
        height=300 * n_rows,
        template="plotly_white",
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "x": 0},
    )
    appendix_html = appendix_fig.to_html(full_html=False, include_plotlyjs=False)

    # --- Kernel breakdown (only present when the run used --profile) ---
    kernel_html, kernel_appendix_html = _kernel_html(results.get("kernels"))

    html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8">
<title>Benchmark Report</title>
<!-- The plotly figures render on a white template, so the page is light-only.
     Without this the browser paints a dark canvas in dark mode and leaves the
     text at its (dark grey) authored colour. -->
<meta name="color-scheme" content="light">
<style>
  body {{ font-family: system-ui, sans-serif; max-width: 1200px;
         margin: 0 auto; padding: 20px; color: #24292f; background: #fff; }}
  h1 {{ font-size: 1.3em; margin-bottom: 0; }}
  h2.sec {{ font-size: 1.1em; font-weight: 600; margin: 34px 0 0; }}
  .sub {{ color: #656d76; font-size: 0.85em; max-width: 900px;
          margin: 4px 0 12px; }}
  details {{ margin-top: 30px; }}
  summary {{ cursor: pointer; font-size: 1.1em; font-weight: 600;
             padding: 8px 0; }}
  table.kernels {{ border-collapse: collapse; font-size: 0.85em; width: 100%; }}
  table.kernels th, table.kernels td {{ text-align: right; padding: 5px 8px;
                                        border-bottom: 1px solid #d8dee4; }}
  table.kernels th {{ color: #656d76; font-weight: 600; }}
  table.kernels tbody tr:nth-child(odd) {{ background: #f6f8fa; }}
  table.kernels th:first-child, table.kernels td:first-child
      {{ text-align: left; }}
  .kname {{ font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
            font-size: 0.95em; color: #0550ae; background: none;
            white-space: nowrap; padding: 0; border-radius: 0; }}
  .kname.trunc {{ border-bottom: 1px dotted #8c959f; cursor: help; }}
</style>
</head><body>
<h1>{heading}</h1>
<div class="sub">{sub}</div>
{timing_html}
{memory_html}
{kernel_html}
{kernel_appendix_html}
<details>
<summary>Appendix: Per-Step Raw Data</summary>
{appendix_html}
</details>
</body></html>"""

    html_path.parent.mkdir(parents=True, exist_ok=True)
    html_path.write_text(html)
    print(f"Report written to {html_path}")

    _write_markdown_summary(results, md_path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser():
    """Build the top-level argument parser."""
    parser = argparse.ArgumentParser(
        description="Training benchmark harness for speculators.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Pass train.py flags after '--'. Example:\n"
            "  python scripts/benchmark.py run --synthetic "
            "-- --verifier-name-or-path Qwen/Qwen3-8B"
        ),
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # --- run ---
    run_parser = subparsers.add_parser("run", help="Run a training benchmark")
    run_parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Use synthetic random data instead of a real dataset.",
    )
    run_parser.add_argument(
        "--warmup-steps",
        type=int,
        default=10,
        help="Warmup steps (not measured). Default: 10.",
    )
    run_parser.add_argument(
        "--measured-steps",
        type=int,
        default=50,
        help="Number of measured steps. Default: 50.",
    )
    run_parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JSON path. Default: benchmark_<ts>.json.",
    )
    run_parser.add_argument(
        "--no-per-step",
        action="store_true",
        help="Omit per-step timing data from the output JSON.",
    )
    run_parser.add_argument(
        "--profile",
        action="store_true",
        help=(
            "Enable torch.profiler to emit a Chrome trace over the measured "
            "steps (warmup steps are profiler warmup). Timing fields for "
            "those steps include profiling overhead; device-side kernel "
            "times do not. View traces in chrome://tracing or TensorBoard."
        ),
    )
    run_parser.add_argument(
        "--profile-dir",
        type=str,
        default="profile_traces",
        help="Directory for torch.profiler trace output. Default: profile_traces/.",
    )
    run_parser.add_argument(
        "--profile-stacks",
        action="store_true",
        help="Capture Python call stacks in the trace (adds overhead).",
    )

    # --- compare ---
    cmp_parser = subparsers.add_parser(
        "compare", help="Compare two benchmark result files"
    )
    cmp_parser.add_argument("baseline", help="Path to baseline result JSON.")
    cmp_parser.add_argument("candidate", help="Path to candidate result JSON.")

    # --- report ---
    viz_parser = subparsers.add_parser(
        "report", help="Generate interactive HTML report from results"
    )
    viz_parser.add_argument("result", help="Path to benchmark result JSON.")
    viz_parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output HTML path. Default: <result>_report.html.",
    )

    # --- comment ---
    comment_parser = subparsers.add_parser(
        "comment", help="Render results as markdown for a GitHub PR comment"
    )
    comment_parser.add_argument("result", help="Path to benchmark result JSON.")
    comment_parser.add_argument(
        "--baseline",
        type=str,
        default=None,
        help="Baseline result JSON. Given, the comment becomes a before/after.",
    )
    comment_parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Write markdown here instead of stdout.",
    )
    comment_parser.add_argument(
        "--label",
        type=str,
        default=None,
        help=(
            "Name for this variant, shown in the heading. Needed to tell apart "
            "several comments on one PR, whose config and commit are otherwise "
            "identical; also scopes the marker so each gets its own comment."
        ),
    )

    return parser


def main():
    parser = build_parser()

    # Split argv on '--' to separate benchmark / train.py args.
    argv = sys.argv[1:]
    if "--" in argv:
        sep_idx = argv.index("--")
        bench_argv = argv[:sep_idx]
        train_argv = argv[sep_idx + 1 :]
    else:
        bench_argv = argv
        train_argv = []

    bench_args = parser.parse_args(bench_argv)

    if bench_args.command == "compare":
        compare_benchmarks(bench_args.baseline, bench_args.candidate)
        return

    if bench_args.command == "report":
        report_benchmark(bench_args.result, bench_args.output)
        return

    if bench_args.command == "comment":
        comment_benchmark(
            bench_args.result,
            bench_args.baseline,
            bench_args.output,
            bench_args.label,
        )
        return

    # --- run command ---
    if bench_args.output is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        bench_args.output = f"benchmark_{ts}.json"

    # Resolve train.py args through the shared TrainConfig, flattened to the
    # same argparse.Namespace the model layer consumes in train.main().
    train_args = argparse.Namespace(**TrainConfig.resolve(train_argv).flatten())

    run_benchmark(bench_args, train_args)


if __name__ == "__main__":
    main()
