#!/usr/bin/env python3
"""Training benchmark harness for speculators.

Measures training loop throughput, timing breakdown, and GPU memory usage
with full provenance tracking. Results are stored as JSON for comparison
across code changes.

Uses the real Trainer class so measurements stay in sync with the actual
training code path.

Subcommands:
    run       Run a training benchmark
    compare   Compare two benchmark result files

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

    # Compare two runs
    python scripts/benchmark.py compare baseline.json candidate.json
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import logging
import socket
import statistics
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch
import torch.distributed as dist

# Allow importing from the scripts/ directory (peer imports like train.py).
sys.path.insert(0, str(Path(__file__).resolve().parent))

from train import (
    build_draft_model,
    parse_vocab_mappings,
    set_seed,
)
from train import (
    parse_args as parse_train_args,
)

from hs_connectors import HiddenStatesBackend
from speculators.model import SpeculatorModel
from speculators.models.eagle3.data import shift_batch
from speculators.models.mtp.data import shift_batch_mtp
from speculators.train.dataloader import create_train_val_loaders
from speculators.train.distributed import (
    get_rank,
    maybe_destroy_distributed,
    maybe_setup_distributed,
)
from speculators.train.logger import setup_root_logger
from speculators.train.trainer import Trainer, TrainerConfig

BENCHMARK_VERSION = "1.0"

TIMING_KEYS = (
    "step_ms",
    "fwd_ms",
    "bwd_ms",
    "opt_ms",
    "fetch_ms",
    "tokens_per_s",
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
    }


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------


def compute_statistics(values: list[float]) -> dict[str, float]:
    """Compute summary statistics for a list of measurements."""
    return {
        "mean": statistics.mean(values),
        "std": statistics.stdev(values) if len(values) > 1 else 0.0,
        "min": min(values),
        "max": max(values),
        "median": statistics.median(values),
        "count": len(values),
    }


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
        legacy_data=train_args.legacy_data,
        transfer=transfer,
        vllm_endpoint=train_args.vllm_endpoint,
        on_missing=train_args.on_missing,
        on_generate=train_args.on_generate,
        verifier_name_or_path=train_args.verifier_name_or_path,
        request_timeout=train_args.request_timeout,
        max_retries=train_args.max_retries,
        hidden_size=hidden_size,
        num_target_layers=num_target_layers,
        num_workers=train_args.num_workers,
        prefetch_factor=train_args.prefetch_factor,
        preprocess=preprocess,
        train_data_ratio=train_args.train_data_ratio,
    )
    return train_loader, False


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
    trainer.train_epoch(0)

    # --- Remove capture handler ---
    metric_logger.removeHandler(capture)

    # --- Collect memory ---
    peak_allocated_mb = torch.cuda.max_memory_allocated(local_rank) / (1024**2)
    peak_reserved_mb = torch.cuda.max_memory_reserved(local_rank) / (1024**2)

    # --- Split warmup / measured profiles ---
    all_profiles = capture.profiles
    measured_profiles = all_profiles[bench_args.warmup_steps :]

    if not measured_profiles:
        if rank == 0:
            print(
                f"WARNING: Got {len(all_profiles)} profiles but expected "
                f"{total_steps}. Check warmup_steps={bench_args.warmup_steps}."
            )
        measured_profiles = all_profiles

    # --- Aggregate ---
    timing_agg = {}
    for key in TIMING_KEYS:
        values = [s[key] for s in measured_profiles]
        timing_agg[key] = compute_statistics(values)

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
    del trainer, draft_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    maybe_destroy_distributed()

    return results


def _print_summary(results: dict) -> None:
    """Print a compact summary of benchmark results to stdout."""
    timing = results["timing"]
    memory = results["memory"]

    print(f"\n{'Metric':<16} {'Mean':>10} {'Std':>10} {'Min':>10} {'Max':>10}")
    print("-" * 58)
    for key in TIMING_KEYS:
        stats = timing[key]
        print(
            f"{key:<16} {stats['mean']:>10.2f} "
            f"{stats['std']:>10.2f} "
            f"{stats['min']:>10.2f} {stats['max']:>10.2f}"
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

    # --- Timing comparison ---
    col_w = 26
    print(
        f"\n{'Metric':<16} "
        f"{'Baseline (mean +/- std)':<{col_w}} "
        f"{'Candidate (mean +/- std)':<{col_w}} "
        f"{'Delta':>10} {'Delta %':>10}"
    )
    print("-" * (16 + col_w * 2 + 22))

    for key in TIMING_KEYS:
        ba = baseline.get("timing", {}).get(key, {})
        ca = candidate.get("timing", {}).get(key, {})
        ba_mean = ba.get("mean", 0)
        ba_std = ba.get("std", 0)
        ca_mean = ca.get("mean", 0)
        ca_std = ca.get("std", 0)
        delta = ca_mean - ba_mean
        pct = (delta / ba_mean * 100) if ba_mean != 0 else 0

        ba_str = f"{ba_mean:>8.2f} +/- {ba_std:<6.2f}"
        ca_str = f"{ca_mean:>8.2f} +/- {ca_std:<6.2f}"
        print(
            f"{key:<16} {ba_str:<{col_w}} {ca_str:<{col_w}} "
            f"{delta:>+10.2f} {pct:>+9.1f}%"
        )

    # --- Memory comparison ---
    print(f"\n{'Memory':<24} {'Baseline':>12} {'Candidate':>12} {'Delta':>12}")
    print("-" * 62)
    for key in ("peak_allocated_mb", "peak_reserved_mb"):
        ba_val = baseline.get("memory", {}).get(key, 0)
        ca_val = candidate.get("memory", {}).get(key, 0)
        delta = ca_val - ba_val
        print(f"{key:<24} {ba_val:>9.1f} MB {ca_val:>9.1f} MB {delta:>+9.1f} MB")


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

    # --- compare ---
    cmp_parser = subparsers.add_parser(
        "compare", help="Compare two benchmark result files"
    )
    cmp_parser.add_argument("baseline", help="Path to baseline result JSON.")
    cmp_parser.add_argument("candidate", help="Path to candidate result JSON.")

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

    # --- run command ---
    if bench_args.output is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        bench_args.output = f"benchmark_{ts}.json"

    # Parse train.py args by injecting them into sys.argv.
    original_argv = sys.argv
    sys.argv = ["train.py"] + train_argv
    try:
        train_args = parse_train_args()
    finally:
        sys.argv = original_argv

    run_benchmark(bench_args, train_args)


if __name__ == "__main__":
    main()
