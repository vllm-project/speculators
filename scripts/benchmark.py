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
    visualize  Generate interactive HTML report from results

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
    python scripts/benchmark.py visualize benchmark_20260818.json
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import logging
import math
import socket
import statistics
import subprocess
import sys
from datetime import datetime, timezone
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
from speculators.train.dataloader import create_train_val_loaders
from speculators.train.distributed import (
    get_rank,
    maybe_destroy_distributed,
    maybe_setup_distributed,
)
from speculators.train.logger import setup_root_logger
from speculators.train.trainer import Trainer, TrainerConfig

BENCHMARK_VERSION = "1.1"

TIMING_KEYS = (
    "step_ms",
    "fwd_ms",
    "bwd_ms",
    "opt_ms",
    "fetch_ms",
    "tokens_per_s",
)

DETAIL_TIMING_KEYS = (
    "queue_ms",
    "h2d_ms",
    "clip_ms",
)


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

    def __init__(self, batch: dict[str, torch.Tensor], num_steps: int):
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
) -> dict[str, torch.Tensor]:
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


def select_measured_profiles(
    all_profiles: list[dict], warmup_steps: int, measured_steps: int
) -> list[dict]:
    """Validate the sample count and discard warmup profiles."""
    total_steps = warmup_steps + measured_steps
    if len(all_profiles) < total_steps:
        raise RuntimeError(
            "Benchmark dataset exhausted before the requested number of steps: "
            f"got {len(all_profiles)}, requested {total_steps} "
            f"({warmup_steps} warmup + {measured_steps} measured). Use more "
            "samples or request fewer benchmark steps."
        )
    return all_profiles[warmup_steps:total_steps]


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
        noise_std=train_args.noise_std,
        transfer=transfer,
        vllm_endpoint=train_args.vllm_endpoint,
        on_missing=train_args.on_missing,
        on_generate=train_args.on_generate,
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
    )
    return train_loader, False


def _aggregate_timing(measured_profiles: list[dict]) -> dict:
    """Compute statistics for all timing keys across measured profiles."""
    agg = {}
    for key in TIMING_KEYS:
        values = [s[key] for s in measured_profiles]
        agg[key] = compute_statistics(values)
    for key in DETAIL_TIMING_KEYS:
        values = [s[key] for s in measured_profiles if key in s]
        if values:
            agg[key] = compute_statistics(values)
    return agg


def _run_training(bench_args, trainer, rank) -> None:
    """Run the training loop, optionally with torch.profiler."""
    if not bench_args.profile:
        trainer.train_epoch(0)
        return

    profile_dir = Path(bench_args.profile_dir)
    profile_dir.mkdir(parents=True, exist_ok=True)
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        schedule=torch.profiler.schedule(
            wait=0,
            warmup=bench_args.warmup_steps,
            active=bench_args.measured_steps,
            repeat=1,
        ),
        on_trace_ready=torch.profiler.tensorboard_trace_handler(str(profile_dir)),
        record_shapes=True,
        profile_memory=True,
        with_stack=bench_args.profile_stacks,
    ) as prof:
        trainer.profiler = prof
        trainer.train_epoch(0)
    if rank == 0:
        print(f"Profile traces written to {profile_dir}")


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
    torch.cuda.reset_peak_memory_stats(local_rank)

    # --- Run the real training loop ---
    _run_training(bench_args, trainer, rank)

    # --- Remove capture handler ---
    metric_logger.removeHandler(capture)

    # --- Collect memory ---
    peak_allocated_mb = torch.cuda.max_memory_allocated(local_rank) / (1024**2)
    peak_reserved_mb = torch.cuda.max_memory_reserved(local_rank) / (1024**2)

    # --- Split warmup / measured profiles ---
    all_profiles = capture.profiles
    measured_profiles = select_measured_profiles(
        all_profiles,
        bench_args.warmup_steps,
        bench_args.measured_steps,
    )

    # --- Aggregate ---
    timing_agg = _aggregate_timing(measured_profiles)
    aggregate = compute_aggregate_throughput(measured_profiles)

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
        },
        "memory": {
            "peak_allocated_mb": round(peak_allocated_mb, 2),
            "peak_reserved_mb": round(peak_reserved_mb, 2),
        },
        "timing": timing_agg,
        "aggregate": aggregate,
    }

    if not bench_args.no_per_step:
        results["per_step"] = [
            {"step": i, **p} for i, p in enumerate(measured_profiles)
        ]

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


def _get_ci(stats: dict, per_step: list[dict] | None, key: str) -> str:
    """Format 95% CI, recomputing from per-step data if needed."""
    lo = stats.get("ci95_lower")
    hi = stats.get("ci95_upper")
    if lo is not None and hi is not None:
        return f"[{lo:.2f}, {hi:.2f}]"
    if per_step:
        recomputed = compute_statistics([s[key] for s in per_step if key in s])
        return f"[{recomputed['ci95_lower']:.2f}, {recomputed['ci95_upper']:.2f}]"
    return "n/a"


def _print_summary(results: dict) -> None:
    """Print a compact summary of benchmark results to stdout."""
    timing = results["timing"]
    memory = results["memory"]
    per_step = results.get("per_step")

    hdr = (
        f"{'Metric':<16} {'Mean':>10} {'Std':>10} "
        f"{'95% CI':>20} {'Min':>10} {'Max':>10}"
    )
    print(f"\n{hdr}")
    print("-" * len(hdr))
    for key in TIMING_KEYS:
        stats = timing[key]
        print(
            f"{key:<16} {stats['mean']:>10.2f} "
            f"{stats['std']:>10.2f} "
            f"{_get_ci(stats, per_step, key):>20} "
            f"{stats['min']:>10.2f} {stats['max']:>10.2f}"
        )

    detail_keys = [k for k in DETAIL_TIMING_KEYS if k in timing]
    if detail_keys:
        print(
            f"\n{'Detail':<16} {'Mean':>10} {'Std':>10} "
            f"{'95% CI':>20} {'Min':>10} {'Max':>10}"
        )
        print("-" * len(hdr))
        for key in detail_keys:
            stats = timing[key]
            print(
                f"  {key:<14} {stats['mean']:>10.2f} "
                f"{stats['std']:>10.2f} "
                f"{_get_ci(stats, per_step, key):>20} "
                f"{stats['min']:>10.2f} {stats['max']:>10.2f}"
            )

    aggregate = results.get("aggregate")
    if aggregate:
        print(
            "\nEffective rank-0 throughput: "
            f"{aggregate['effective_rank0_tokens_per_s']:.2f} tokens/s "
            f"over {aggregate['measured_time_s']:.2f} s"
        )
    print(
        f"\nPeak memory: {memory['peak_allocated_mb']:.1f} MB "
        f"allocated, {memory['peak_reserved_mb']:.1f} MB reserved"
    )


# ---------------------------------------------------------------------------
# Compare
# ---------------------------------------------------------------------------


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
    a = [s[key] for s in per_step_a if key in s]
    b = [s[key] for s in per_step_b if key in s]
    min_samples = 2
    if len(a) < min_samples or len(b) < min_samples:
        return None
    t_stat, p_value = sp_stats.ttest_ind(a, b, equal_var=False)
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


def compare_benchmarks(baseline_path: str, candidate_path: str) -> None:
    """Load two result files and print a comparison table."""
    with open(baseline_path) as f:
        baseline = json.load(f)
    with open(candidate_path) as f:
        candidate = json.load(f)

    # --- Comparability warnings ---
    comparability_checks = [
        ("config.speculator_type", "Speculator type"),
        ("config.hidden_size", "Hidden size"),
        ("config.total_seq_len", "Sequence length"),
        ("config.num_gpus_used", "GPU count"),
        ("config.fsdp_shard", "FSDP shard"),
        ("config.optimizer", "Optimizer"),
        ("config.hidden_states_dtype", "Dtype"),
    ]
    warnings_found = False
    for key_path, label in comparability_checks:
        val_a = _nested_get(baseline, key_path)
        val_b = _nested_get(candidate, key_path)
        if val_a != val_b:
            if not warnings_found:
                print("COMPARABILITY WARNINGS:")
                warnings_found = True
            print(f"  {label}: {val_a} vs {val_b}")

    gpu_a = _get_gpu_name(baseline)
    gpu_b = _get_gpu_name(candidate)
    if gpu_a != gpu_b:
        if not warnings_found:
            print("COMPARABILITY WARNINGS:")
        print(f"  GPU: {gpu_a} vs {gpu_b}")

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

    all_timing_keys = list(TIMING_KEYS)
    for key in DETAIL_TIMING_KEYS:
        if key in baseline.get("timing", {}) or key in candidate.get("timing", {}):
            all_timing_keys.append(key)

    for key in all_timing_keys:
        ba = baseline.get("timing", {}).get(key, {})
        ca = candidate.get("timing", {}).get(key, {})
        ba_mean = ba.get("mean", 0)
        ba_std = ba.get("std", 0)
        ca_mean = ca.get("mean", 0)
        ca_std = ca.get("std", 0)
        delta = ca_mean - ba_mean
        pct = (delta / ba_mean * 100) if ba_mean != 0 else 0

        label = f"  {key}" if key in DETAIL_TIMING_KEYS else key
        ba_str = f"{ba_mean:>8.2f} +/- {ba_std:<6.2f}"
        ca_str = f"{ca_mean:>8.2f} +/- {ca_std:<6.2f}"
        stat_str = _fmt_welch(per_step_a, per_step_b, key) if has_stats else ""
        print(
            f"{label:<16} {ba_str:<{col_w}} {ca_str:<{col_w}} "
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
    """Print the memory comparison table."""
    print(f"\n{'Memory':<24} {'Baseline':>12} {'Candidate':>12} {'Delta':>12}")
    print("-" * 62)
    for key in ("peak_allocated_mb", "peak_reserved_mb"):
        ba_val = baseline.get("memory", {}).get(key, 0)
        ca_val = candidate.get("memory", {}).get(key, 0)
        delta = ca_val - ba_val
        print(f"{key:<24} {ba_val:>9.1f} MB {ca_val:>9.1f} MB {delta:>+9.1f} MB")


# ---------------------------------------------------------------------------
# Visualize
# ---------------------------------------------------------------------------

_PHASE_COLORS = {
    "queue_ms": "#636EFA",
    "h2d_ms": "#AB63FA",
    "fwd_ms": "#00CC96",
    "bwd_ms": "#FFA15A",
    "clip_ms": "#FECB52",
    "opt_ms": "#EF553B",
    "fetch_ms": "#636EFA",
}

_PHASE_LABELS = {
    "queue_ms": "DataLoader wait",
    "h2d_ms": "H2D transfer",
    "fwd_ms": "Forward",
    "bwd_ms": "Backward",
    "clip_ms": "Grad clip",
    "opt_ms": "Optimizer",
    "fetch_ms": "Data fetch",
}

_MEMORY_MARK_LABELS = {
    "queue": "After queue",
    "fetch": "After H2D",
    "fwd": "After forward",
    "pre_clip": "After backward",
    "bwd": "After clip",
    "opt": "After optimizer",
}


def _build_timing_traces(per_step):
    """Build stacked area traces for timing breakdown."""
    import plotly.graph_objects as go  # noqa: PLC0415

    steps = list(range(len(per_step)))
    has_detail = "queue_ms" in per_step[0]

    if has_detail:
        phases = [("queue_ms", [s["queue_ms"] for s in per_step])]
        phases.append(("h2d_ms", [s["h2d_ms"] for s in per_step]))
        phases.append(("fwd_ms", [s["fwd_ms"] for s in per_step]))
        if "clip_ms" in per_step[0]:
            bwd_only = [s["bwd_ms"] - s.get("clip_ms", 0) for s in per_step]
            phases.append(("bwd_ms", bwd_only))
            phases.append(("clip_ms", [s["clip_ms"] for s in per_step]))
        else:
            phases.append(("bwd_ms", [s["bwd_ms"] for s in per_step]))
        phases.append(("opt_ms", [s["opt_ms"] for s in per_step]))
    else:
        phases = [
            ("fetch_ms", [s["fetch_ms"] for s in per_step]),
            ("fwd_ms", [s["fwd_ms"] for s in per_step]),
            ("bwd_ms", [s["bwd_ms"] for s in per_step]),
            ("opt_ms", [s["opt_ms"] for s in per_step]),
        ]

    traces = []
    for key, values in phases:
        traces.append(
            go.Scatter(
                x=steps,
                y=values,
                name=_PHASE_LABELS.get(key, key),
                mode="lines",
                stackgroup="timing",
                line={"width": 0.5, "color": _PHASE_COLORS.get(key)},
            )
        )
    return traces


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
    """Compute mean phase durations, splitting bwd/clip when detail is available."""
    has_detail = "queue_ms" in per_step[0]
    if has_detail:
        phases = [
            ("queue_ms", "DataLoader wait"),
            ("h2d_ms", "H2D transfer"),
            ("fwd_ms", "Forward"),
            ("bwd_ms", "Backward"),
            ("clip_ms", "Grad clip"),
            ("opt_ms", "Optimizer"),
        ]
        if "clip_ms" not in per_step[0]:
            phases = [p for p in phases if p[0] != "clip_ms"]
    else:
        phases = [
            ("fetch_ms", "Data fetch"),
            ("fwd_ms", "Forward"),
            ("bwd_ms", "Backward"),
            ("opt_ms", "Optimizer"),
        ]

    result = []
    for key, label in phases:
        values = [s[key] for s in per_step if key in s]
        if not values:
            continue
        mean_val = statistics.mean(values)
        std_val = statistics.stdev(values) if len(values) > 1 else 0.0
        if key == "bwd_ms" and has_detail and "clip_ms" in per_step[0]:
            clip_values = [s["clip_ms"] for s in per_step]
            bwd_only = [b - c for b, c in zip(values, clip_values, strict=True)]
            mean_val = statistics.mean(bwd_only)
            std_val = statistics.stdev(bwd_only) if len(bwd_only) > 1 else 0.0
        result.append((key, label, mean_val, std_val))
    return result


def _build_summary_fig(results, per_step):
    """Build the summary dashboard: aggregate stats as charts."""
    import plotly.graph_objects as go  # noqa: PLC0415
    from plotly.subplots import make_subplots  # noqa: PLC0415

    has_memory = any("memory_mb" in s for s in per_step)

    n_rows = 2 if has_memory else 1
    row_heights = [0.6, 0.4] if has_memory else [1.0]
    titles = [""]
    if has_memory:
        titles.append("Mean GPU Memory Allocated (logical) at Phase Boundaries (MB)")

    fig = make_subplots(
        rows=n_rows,
        cols=1,
        subplot_titles=titles,
        vertical_spacing=0.15,
        row_heights=row_heights,
    )

    phase_stats = _get_phase_means(per_step)
    total_ms = sum(mean for _, _, mean, _ in phase_stats)

    # --- Timing breakdown: horizontal bar with ms and % ---
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
            ),
            row=1,
            col=1,
        )

    fig.update_layout(barmode="stack", yaxis={"categoryorder": "array"})

    step_stats = compute_statistics([s["step_ms"] for s in per_step])
    tps_stats = compute_statistics([s["tokens_per_s"] for s in per_step])
    fig.add_annotation(
        text=(
            f"<b>step: {step_stats['mean']:.1f} ms</b>"
            f" [{step_stats['ci95_lower']:.1f},"
            f" {step_stats['ci95_upper']:.1f}]"
            f"  ·  <b>{tps_stats['mean']:.0f} tok/s</b>"
            f" [{tps_stats['ci95_lower']:.0f},"
            f" {tps_stats['ci95_upper']:.0f}]"
            f"  ·  peak {results['memory']['peak_allocated_mb']:.0f} MB"
            f"<br>"
            f"<span style='color:#888'>95% CI, n={step_stats['count']}</span>"
        ),
        xref="paper",
        yref="paper",
        x=0.5,
        y=1.22,
        showarrow=False,
        font={"size": 13},
    )

    # --- Memory breakdown: mean at each phase mark with std error bars ---
    if has_memory:
        marks = list(_MEMORY_MARK_LABELS.keys())
        labels = list(_MEMORY_MARK_LABELS.values())
        means = []
        stds = []
        for mark in marks:
            vals = [
                s["memory_mb"][mark]
                for s in per_step
                if "memory_mb" in s and mark in s["memory_mb"]
            ]
            if vals:
                means.append(statistics.mean(vals))
                stds.append(statistics.stdev(vals) if len(vals) > 1 else 0.0)
            else:
                means.append(0)
                stds.append(0)

        mem_row = n_rows
        fig.add_trace(
            go.Bar(
                x=labels,
                y=means,
                error_y={"type": "data", "array": stds, "visible": True},
                marker_color="#636EFA",
                text=[f"{m:.0f}" for m in means],
                textposition="outside",
                showlegend=False,
            ),
            row=mem_row,
            col=1,
        )
        fig.update_yaxes(rangemode="tozero", row=mem_row, col=1)

    return fig


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


def visualize_benchmark(result_path: str, output_path: str | None = None) -> None:
    """Generate an interactive HTML report from benchmark results."""
    try:
        import plotly  # noqa: PLC0415, F401
    except ImportError:
        print("plotly is required: pip install plotly", file=sys.stderr)
        sys.exit(1)

    with open(result_path) as f:
        results = json.load(f)

    per_step = results.get("per_step")
    if not per_step:
        print(
            "No per-step data found. Re-run benchmark without --no-per-step.",
            file=sys.stderr,
        )
        sys.exit(1)

    if output_path is None:
        output_path = result_path.replace(".json", "_report.html")

    cfg = results.get("config", {})
    prov = results.get("provenance", {})
    heading = (
        f"{cfg.get('speculator_type', '?')} &middot; "
        f"{cfg.get('verifier_name_or_path', '?')} &middot; "
        f"seq_len={cfg.get('total_seq_len', '?')} &middot; "
        f"{cfg.get('num_gpus_used', 1)} GPU(s)"
    )
    sub = (
        f"{cfg.get('measured_steps', '?')} measured steps &middot; "
        f"{_get_gpu_name(results)} &middot; "
        f"{prov.get('git_sha', 'unknown')[:12]}"
    )

    # --- Summary dashboard ---
    summary_fig = _build_summary_fig(results, per_step)
    has_memory = any("memory_mb" in s for s in per_step)
    summary_height = 500 if has_memory else 350
    summary_fig.update_layout(
        height=summary_height,
        template="plotly_white",
        margin={"t": 110},
    )
    summary_html = summary_fig.to_html(full_html=False, include_plotlyjs="cdn")

    # --- Appendix: per-step raw data ---
    appendix_fig, n_rows = _build_appendix_fig(per_step)
    appendix_fig.update_layout(
        height=300 * n_rows,
        template="plotly_white",
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "x": 0},
    )
    appendix_html = appendix_fig.to_html(full_html=False, include_plotlyjs=False)

    html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8">
<title>Benchmark Report</title>
<style>
  body {{ font-family: system-ui, sans-serif; max-width: 1200px;
         margin: 0 auto; padding: 20px; color: #333; }}
  h1 {{ font-size: 1.3em; margin-bottom: 0; }}
  .sub {{ color: #888; font-size: 0.85em; margin-bottom: 20px; }}
  details {{ margin-top: 30px; }}
  summary {{ cursor: pointer; font-size: 1.1em; font-weight: 600;
             padding: 8px 0; }}
</style>
</head><body>
<h1>{heading}</h1>
<div class="sub">{sub}</div>
{summary_html}
<details>
<summary>Appendix: Per-Step Raw Data</summary>
{appendix_html}
</details>
</body></html>"""

    Path(output_path).write_text(html)
    print(f"Report written to {output_path}")


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
            "Enable torch.profiler to emit a Chrome trace. "
            "Warmup steps are used as profiler warmup; measured steps are "
            "actively profiled. View traces in chrome://tracing or TensorBoard."
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

    # --- visualize ---
    viz_parser = subparsers.add_parser(
        "visualize", help="Generate interactive HTML report from results"
    )
    viz_parser.add_argument("result", help="Path to benchmark result JSON.")
    viz_parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output HTML path. Default: <result>_report.html.",
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

    if bench_args.command == "visualize":
        visualize_benchmark(bench_args.result, bench_args.output)
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
