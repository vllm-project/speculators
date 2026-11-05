"""Unit tests for the benchmark harness."""

from __future__ import annotations

import json
import logging
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch

# Add scripts/ to the import path the same way the benchmark script does.
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts"))

from benchmark import (  # type: ignore[import-not-found]
    BENCHMARK_VERSION,
    _aggregate_kernel_rows,
    _aggregate_timing,
    _build_kernel_pie,
    _fmt_welch,
    _gather_per_rank,
    _get_phase_means,
    _MetricCapture,
    _phase_peaks,
    _print_summary,
    _report_paths,
    _run_training,
    _SyntheticLoader,
    _welch_test,
    collect_provenance,
    compare_benchmarks,
    compute_aggregate_throughput,
    compute_statistics,
    create_synthetic_batch,
    measured_step_window,
    report_benchmark,
    select_measured_profiles,
    shutdown_dataloader_workers,
)

# ---------------------------------------------------------------------------
# compute_statistics
# ---------------------------------------------------------------------------


class TestComputeStatistics:
    def test_basic(self):
        values = [10.0, 20.0, 30.0, 40.0, 50.0]
        result = compute_statistics(values)
        assert result["mean"] == pytest.approx(30.0)
        assert result["min"] == 10.0
        assert result["max"] == 50.0
        assert result["median"] == 30.0
        assert result["count"] == 5
        assert result["std"] > 0

    def test_single_value(self):
        result = compute_statistics([42.0])
        assert result["mean"] == 42.0
        assert result["std"] == 0.0
        assert result["min"] == 42.0
        assert result["max"] == 42.0
        assert result["median"] == 42.0
        assert result["count"] == 1

    def test_identical_values(self):
        result = compute_statistics([5.0, 5.0, 5.0])
        assert result["mean"] == 5.0
        assert result["std"] == 0.0


def test_compute_aggregate_throughput_is_time_weighted():
    profiles = [
        {"step_ms": 1000.0, "tokens_per_s": 100.0},
        {"step_ms": 3000.0, "tokens_per_s": 300.0},
    ]

    result = compute_aggregate_throughput(profiles)

    assert result["measured_time_s"] == 4.0
    assert result["rank0_tokens"] == 1000.0
    assert result["effective_rank0_tokens_per_s"] == 250.0


def test_shutdown_dataloader_workers():
    loader = MagicMock()
    iterator = loader._iterator

    shutdown_dataloader_workers(loader)

    iterator._shutdown_workers.assert_called_once_with()
    assert loader._iterator is None


# ---------------------------------------------------------------------------
# collect_provenance
# ---------------------------------------------------------------------------


class TestCollectProvenance:
    @patch("benchmark.torch")
    @patch("benchmark.subprocess.run")
    def test_keys_present(self, mock_run, mock_torch):
        mock_run.return_value = MagicMock(returncode=0, stdout="abc123\n")
        mock_torch.cuda.is_available.return_value = False
        mock_torch.cuda.device_count.return_value = 0
        mock_torch.__version__ = "2.9.0"
        mock_torch.version.cuda = "12.4"

        result = collect_provenance()

        expected_keys = {
            "git_sha",
            "timestamp",
            "hostname",
            "python_version",
            "pytorch_version",
            "cuda_version",
            "speculators_version",
            "transformers_version",
            "gpu_info",
            "num_gpus",
        }
        assert set(result.keys()) == expected_keys
        assert result["git_sha"] == "abc123"
        assert result["num_gpus"] == 0

    @patch("benchmark.torch")
    @patch("benchmark.subprocess.run")
    def test_git_failure(self, mock_run, mock_torch):
        mock_run.return_value = MagicMock(returncode=128, stdout="")
        mock_torch.cuda.is_available.return_value = False
        mock_torch.cuda.device_count.return_value = 0
        mock_torch.__version__ = "2.9.0"
        mock_torch.version.cuda = None

        result = collect_provenance()
        assert result["git_sha"] == "unknown"
        assert result["cuda_version"] == "none"


# ---------------------------------------------------------------------------
# create_synthetic_batch
# ---------------------------------------------------------------------------


class TestCreateSyntheticBatch:
    def test_shapes(self):
        seq_len = 128
        hidden_size = 64
        num_layers = 3
        batch = create_synthetic_batch(
            total_seq_len=seq_len,
            hidden_size=hidden_size,
            num_target_layers=num_layers,
            device="cpu",
        )

        assert batch["hidden_states"].shape == (
            1,
            seq_len,
            num_layers * hidden_size,
        )
        assert batch["input_ids"].shape == (1, seq_len)
        assert batch["verifier_last_hidden_states"].shape == (
            1,
            seq_len,
            hidden_size,
        )
        assert batch["loss_mask"].shape == (1, seq_len)
        assert batch["position_ids"].shape == (1, seq_len)
        assert batch["document_ids"].shape == (1, seq_len)

    def test_dtypes(self):
        batch = create_synthetic_batch(
            total_seq_len=64,
            hidden_size=32,
            num_target_layers=2,
            dtype=torch.bfloat16,
            device="cpu",
        )

        assert batch["hidden_states"].dtype == torch.bfloat16
        assert batch["verifier_last_hidden_states"].dtype == torch.bfloat16
        assert batch["input_ids"].dtype == torch.long
        assert batch["loss_mask"].dtype == torch.bool
        assert batch["position_ids"].dtype == torch.long
        assert batch["document_ids"].dtype == torch.long

    def test_position_ids_start_at_one(self):
        batch = create_synthetic_batch(
            total_seq_len=10,
            hidden_size=16,
            num_target_layers=1,
            device="cpu",
        )
        assert batch["position_ids"][0, 0].item() == 1
        assert batch["position_ids"][0, -1].item() == 10

    def test_document_ids_all_zero(self):
        batch = create_synthetic_batch(
            total_seq_len=10,
            hidden_size=16,
            num_target_layers=1,
            device="cpu",
        )
        assert (batch["document_ids"] == 0).all()

    def test_all_keys_present(self):
        batch = create_synthetic_batch(
            total_seq_len=8,
            hidden_size=16,
            num_target_layers=1,
            device="cpu",
        )
        expected_keys = {
            "hidden_states",
            "input_ids",
            "verifier_last_hidden_states",
            "loss_mask",
            "position_ids",
            "document_ids",
            "error_records",
        }
        assert set(batch.keys()) == expected_keys


# ---------------------------------------------------------------------------
# _aggregate_timing
# ---------------------------------------------------------------------------


class TestAggregateTiming:
    def test_aggregates_numeric_keys_and_skips_memory(self):
        profiles = [
            {
                "step_ms": 40.0 + i,
                "fwd_ms": 20.0,
                "queue_ms": 3.0,
                "memory_mb": {"fetch": 100.0, "opt": 110.0},
            }
            for i in range(3)
        ]
        agg = _aggregate_timing(profiles)
        assert set(agg) == {"step_ms", "fwd_ms", "queue_ms"}
        assert agg["step_ms"]["mean"] == pytest.approx(41.0)
        assert agg["step_ms"]["count"] == 3
        assert "ci95_lower" in agg["fwd_ms"]
        assert "ci95_upper" in agg["fwd_ms"]


# ---------------------------------------------------------------------------
# _welch_test / _fmt_welch
# ---------------------------------------------------------------------------


class TestWelchTest:
    def test_shifted_samples(self):
        a = [{"step_ms": v} for v in (10.0, 11.0, 12.0, 13.0)]
        b = [{"step_ms": v} for v in (20.0, 21.0, 22.0, 23.0)]
        t_stat, p_value, cohens_d = _welch_test(a, b, "step_ms")
        assert t_stat < 0
        assert p_value < 0.05
        assert cohens_d > 0.8

    def test_missing_per_step_returns_none(self):
        assert _welch_test([], [{"step_ms": 1.0}], "step_ms") is None
        assert _welch_test([{"step_ms": 1.0}], [], "step_ms") is None

    def test_single_sample_returns_none(self):
        assert _welch_test([{"step_ms": 1.0}], [{"step_ms": 2.0}], "step_ms") is None


class TestFmtWelch:
    def test_na_without_data(self):
        out = _fmt_welch([], [], "step_ms")
        assert out.count("n/a") == 2

    def test_formats_p_and_d(self):
        a = [{"step_ms": v} for v in (10.0, 11.0, 12.0, 13.0)]
        b = [{"step_ms": v} for v in (20.0, 21.0, 22.0, 23.0)]
        out = _fmt_welch(a, b, "step_ms")
        assert "n/a" not in out


# ---------------------------------------------------------------------------
# _print_summary
# ---------------------------------------------------------------------------


class TestPrintSummary:
    def test_prints_metrics_ci_throughput_memory(self, capsys):
        profiles = [{"step_ms": 40.0 + i, "queue_ms": 3.0} for i in range(3)]
        results = {
            "timing": _aggregate_timing(profiles),
            "memory": {
                "per_rank": [
                    {
                        "rank": 0,
                        "peak_allocated_mb": 2048.0,
                        "peak_reserved_mb": 3072.0,
                    },
                    {
                        "rank": 1,
                        "peak_allocated_mb": 1900.0,
                        "peak_reserved_mb": 2900.0,
                    },
                ],
                "phases": {"fwd": 1800.0, "bwd": 1900.0},
            },
            "aggregate": {
                "effective_rank0_tokens_per_s": 1000.0,
                "measured_time_s": 1.5,
            },
        }
        _print_summary(results)
        out = capsys.readouterr().out
        assert "95% CI" in out
        assert "step_ms" in out
        assert "queue_ms" in out
        assert "[3.00, 3.00]" in out
        assert "Effective rank-0 throughput: 1000.00 tokens/s" in out
        assert "rank 0: 2048.0 MB allocated" in out
        assert "rank 1: 1900.0 MB allocated" in out
        assert "fwd=1800.0" in out


# ---------------------------------------------------------------------------
# _MetricCapture
# ---------------------------------------------------------------------------


class TestMetricCapture:
    def test_captures_profile_dicts(self):
        capture = _MetricCapture()
        profile = {"step_ms": 45.0, "fwd_ms": 20.0}
        record = logging.LogRecord(
            name="speculators.metrics",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg={"train": {}, "profile": profile, "epoch": 0},
            args=None,
            exc_info=None,
        )
        capture.emit(record)
        assert len(capture.profiles) == 1
        assert capture.profiles[0] is profile

    def test_ignores_records_without_profile(self):
        capture = _MetricCapture()
        record = logging.LogRecord(
            name="speculators.metrics",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg={"train": {}, "epoch": 0},
            args=None,
            exc_info=None,
        )
        capture.emit(record)
        assert len(capture.profiles) == 0

    def test_ignores_none_profile(self):
        capture = _MetricCapture()
        record = logging.LogRecord(
            name="speculators.metrics",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg={"train": {}, "profile": None, "epoch": 0},
            args=None,
            exc_info=None,
        )
        capture.emit(record)
        assert len(capture.profiles) == 0

    def test_ignores_non_dict_messages(self):
        capture = _MetricCapture()
        record = logging.LogRecord(
            name="speculators.metrics",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg="some string message",
            args=None,
            exc_info=None,
        )
        capture.emit(record)
        assert len(capture.profiles) == 0

    def test_captures_multiple(self):
        capture = _MetricCapture()
        for i in range(5):
            record = logging.LogRecord(
                name="speculators.metrics",
                level=logging.INFO,
                pathname="",
                lineno=0,
                msg={"profile": {"step_ms": float(i)}, "train": {}},
                args=None,
                exc_info=None,
            )
            capture.emit(record)
        assert len(capture.profiles) == 5
        assert capture.profiles[3]["step_ms"] == 3.0


# ---------------------------------------------------------------------------
# _SyntheticLoader
# ---------------------------------------------------------------------------


class TestSyntheticLoader:
    def test_len(self):
        batch = {"x": torch.zeros(1)}
        loader = _SyntheticLoader(batch, num_steps=7)
        assert len(loader) == 7

    def test_iter_yields_correct_count(self):
        batch = {"x": torch.zeros(1)}
        loader = _SyntheticLoader(batch, num_steps=3)
        batches = list(loader)
        assert len(batches) == 3

    def test_iter_yields_same_batch(self):
        batch = {"x": torch.tensor([1.0, 2.0])}
        loader = _SyntheticLoader(batch, num_steps=3)
        for b in loader:
            assert b is batch

    def test_batch_sampler_has_set_epoch(self):
        batch = {"x": torch.zeros(1)}
        loader = _SyntheticLoader(batch, num_steps=1)
        assert hasattr(loader.batch_sampler, "set_epoch")
        loader.batch_sampler.set_epoch(5)


# ---------------------------------------------------------------------------
# Warmup / measured split
# ---------------------------------------------------------------------------


class TestMeasuredStepWindow:
    def test_window_is_single_source_for_slice_and_normalization(self):
        window = measured_step_window(3, 5)
        assert list(window) == [3, 4, 5, 6, 7]
        assert window.start == 3  # profiler warmup count
        assert len(window) == 5  # kernel normalization divisor


class TestWarmupMeasuredSplit:
    """Tests for the profile slicing logic used in run_benchmark."""

    def test_discard_warmup(self):
        all_profiles = [{"step_ms": float(i)} for i in range(13)]
        measured = select_measured_profiles(all_profiles, measured_step_window(3, 10))
        assert len(measured) == 10
        assert measured[0]["step_ms"] == 3.0

    def test_exact_boundary(self):
        all_profiles = [{"step_ms": float(i)} for i in range(5)]
        with pytest.raises(RuntimeError, match="dataset exhausted"):
            select_measured_profiles(all_profiles, measured_step_window(5, 1))

    def test_zero_warmup(self):
        all_profiles = [{"step_ms": float(i)} for i in range(10)]
        measured = select_measured_profiles(all_profiles, measured_step_window(0, 10))
        assert len(measured) == 10
        assert measured[0]["step_ms"] == 0.0

    def test_insufficient_profiles_raises(self):
        all_profiles = [{"step_ms": float(i)} for i in range(5)]
        with pytest.raises(RuntimeError, match="got 5, requested 15"):
            select_measured_profiles(all_profiles, measured_step_window(10, 5))

    def test_extra_profiles_are_not_measured(self):
        all_profiles = [{"step_ms": float(i)} for i in range(20)]
        measured = select_measured_profiles(all_profiles, measured_step_window(3, 5))
        assert [profile["step_ms"] for profile in measured] == [
            3.0,
            4.0,
            5.0,
            6.0,
            7.0,
        ]


# ---------------------------------------------------------------------------
# Kernel breakdown
# ---------------------------------------------------------------------------


class TestAggregateKernelRows:
    def test_sums_across_ranks_and_normalizes_per_step(self):
        rank0 = [{"name": "kA", "self_device_us": 6000.0, "count": 3}]
        rank1 = [
            {"name": "kA", "self_device_us": 4000.0, "count": 2},
            {"name": "kB", "self_device_us": 1000.0, "count": 1},
        ]
        out = _aggregate_kernel_rows([rank0, rank1], active_steps=2)
        assert out["num_ranks"] == 2
        by_name = {k["name"]: k for k in out["kernels"]}
        assert by_name["kA"]["ms_per_step"] == pytest.approx(5.0)
        assert by_name["kA"]["calls_per_step"] == pytest.approx(2.5)
        assert by_name["kB"]["ms_per_step"] == pytest.approx(0.5)
        assert out["kernels"][0]["name"] == "kA"  # sorted by time
        assert out["total_device_ms_per_step"] == pytest.approx(5.5)

    def test_zero_time_entries_dropped(self):
        rows = [[{"name": "z", "self_device_us": 0.0, "count": 5}]]
        out = _aggregate_kernel_rows(rows, active_steps=1)
        assert out["kernels"] == []
        assert out["total_device_ms_per_step"] == pytest.approx(0.0)


class TestPhasePeaks:
    def test_max_per_phase_across_steps(self):
        profiles = [
            {"memory_mb": {"fwd": 100.0, "bwd": 150.0}},
            {"memory_mb": {"fwd": 120.5, "bwd": 140.0}},
            {"step_ms": 1.0},  # step without memory marks (CPU-only run)
        ]
        assert _phase_peaks(profiles) == {"fwd": 120.5, "bwd": 150.0}


class TestGatherPerRank:
    def test_single_rank_passthrough(self):
        entry = {"rank": 0, "peak_allocated_mb": 1.0}
        assert _gather_per_rank(entry) == [entry]


class TestRunTrainingPlain:
    def test_no_profiler_runs_without_callback(self):
        seen = {}

        class _T:
            @staticmethod
            def train_epoch(epoch, step_callback=None):
                seen["epoch"] = epoch
                seen["step_callback"] = step_callback

        args = types.SimpleNamespace(profile=False)
        result = _run_training(args, _T(), rank=0, window=measured_step_window(1, 2))
        assert result is None
        assert seen == {"epoch": 0, "step_callback": None}


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
class TestRunTrainingProfiled:
    def _args(self, tmp_path):
        return types.SimpleNamespace(
            profile=True,
            profile_dir=str(tmp_path / "traces"),
            profile_stacks=False,
            output=str(tmp_path / "run.json"),
        )

    def _trainer(self, steps):
        class _T:
            @staticmethod
            def train_epoch(epoch, step_callback=None):
                assert step_callback is not None
                x = torch.randn(128, 128, device="cuda")
                for _ in range(steps):
                    x @ x
                    step_callback()

        return _T()

    def test_breakdown_and_traces_over_measured_window(self, tmp_path):
        out = _run_training(
            self._args(tmp_path),
            self._trainer(3),
            rank=0,
            window=measured_step_window(1, 2),
        )
        assert out["num_ranks"] == 1
        assert out["total_device_ms_per_step"] > 0
        traces = list((tmp_path / "traces").iterdir())
        assert traces
        assert all(".pt.trace.json" in f.name for f in traces)
        assert all(f.name.startswith("run-rank0") for f in traces)

    def test_callback_drift_raises(self, tmp_path):
        # 2 steps into a window that expects 3.
        with pytest.raises(RuntimeError, match="step_callback must fire"):
            _run_training(
                self._args(tmp_path),
                self._trainer(2),
                rank=0,
                window=measured_step_window(1, 2),
            )


# ---------------------------------------------------------------------------
# compare_benchmarks
# ---------------------------------------------------------------------------


def _make_result(
    step_ms_mean=45.0,
    step_ms_std=1.0,
    peak_alloc=2048.0,
    git_sha="aaa",
    gpu_name="H100",
    speculator_type="eagle3",
):
    """Create a minimal benchmark result dict for testing."""
    timing = {}
    for key in (
        "step_ms",
        "fwd_ms",
        "bwd_ms",
        "opt_ms",
        "fetch_ms",
        "tokens_per_s",
    ):
        timing[key] = {
            "mean": step_ms_mean,
            "std": step_ms_std,
            "min": step_ms_mean - 2,
            "max": step_ms_mean + 2,
            "median": step_ms_mean,
            "count": 50,
        }
    return {
        "benchmark_version": "1.0",
        "provenance": {
            "git_sha": git_sha,
            "gpu_info": [{"name": gpu_name, "total_memory_gb": 80.0}],
        },
        "config": {
            "speculator_type": speculator_type,
            "hidden_size": 4096,
            "total_seq_len": 8192,
            "num_gpus_used": 1,
            "fsdp_shard": False,
            "optimizer": "muon",
            "hidden_states_dtype": "bfloat16",
        },
        "memory": {
            "per_rank": [
                {
                    "rank": 0,
                    "peak_allocated_mb": peak_alloc,
                    "peak_reserved_mb": peak_alloc + 1024,
                }
            ],
            "phases": {"fwd": peak_alloc * 0.5},
        },
        "timing": timing,
    }


class TestCompareBenchmarks:
    def test_basic_compare(self, tmp_path, capsys):
        baseline = _make_result(step_ms_mean=50.0, git_sha="aaa111")
        candidate = _make_result(step_ms_mean=45.0, git_sha="bbb222")
        baseline["aggregate"] = {"effective_rank0_tokens_per_s": 1000.0}
        candidate["aggregate"] = {"effective_rank0_tokens_per_s": 1200.0}

        baseline_path = tmp_path / "baseline.json"
        candidate_path = tmp_path / "candidate.json"
        baseline_path.write_text(json.dumps(baseline))
        candidate_path.write_text(json.dumps(candidate))

        compare_benchmarks(str(baseline_path), str(candidate_path))

        output = capsys.readouterr().out
        assert "aaa111" in output
        assert "bbb222" in output
        assert "step_ms" in output
        assert "-5.00" in output or "-10.0%" in output
        assert "1000.00 -> 1200.00" in output

    def test_version_mismatch_rejected(self, tmp_path):
        baseline = _make_result()
        candidate = _make_result()
        candidate["benchmark_version"] = "99.9"

        baseline_path = tmp_path / "baseline.json"
        candidate_path = tmp_path / "candidate.json"
        baseline_path.write_text(json.dumps(baseline))
        candidate_path.write_text(json.dumps(candidate))

        with pytest.raises(SystemExit, match="Incompatible benchmark versions"):
            compare_benchmarks(str(baseline_path), str(candidate_path))

    def test_comparability_warning_gpu(self, tmp_path, capsys):
        baseline = _make_result(gpu_name="H100")
        candidate = _make_result(gpu_name="A100")

        baseline_path = tmp_path / "b.json"
        candidate_path = tmp_path / "c.json"
        baseline_path.write_text(json.dumps(baseline))
        candidate_path.write_text(json.dumps(candidate))

        compare_benchmarks(str(baseline_path), str(candidate_path))

        output = capsys.readouterr().out
        assert "GPU" in output
        assert "H100" in output
        assert "A100" in output

    def test_comparability_warning_config(self, tmp_path, capsys):
        baseline = _make_result(speculator_type="eagle3")
        candidate = _make_result(speculator_type="dflash")

        baseline_path = tmp_path / "b.json"
        candidate_path = tmp_path / "c.json"
        baseline_path.write_text(json.dumps(baseline))
        candidate_path.write_text(json.dumps(candidate))

        compare_benchmarks(str(baseline_path), str(candidate_path))

        output = capsys.readouterr().out
        assert "Speculator type" in output

    def test_memory_delta(self, tmp_path, capsys):
        baseline = _make_result(peak_alloc=2000.0)
        candidate = _make_result(peak_alloc=1800.0)

        baseline_path = tmp_path / "b.json"
        candidate_path = tmp_path / "c.json"
        baseline_path.write_text(json.dumps(baseline))
        candidate_path.write_text(json.dumps(candidate))

        compare_benchmarks(str(baseline_path), str(candidate_path))

        output = capsys.readouterr().out
        assert "rank0 peak_allocated" in output
        assert "phase fwd" in output
        assert "-200.0" in output


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------


def _make_per_step(n=4):
    """Per-step profiles shaped like the trainer's, with memory marks."""
    return [
        {
            "step": i,
            "fetch_ms": 7.0,
            "fwd_ms": 30.0,
            "bwd_ms": 62.0,  # includes clip_ms, as the trainer reports it
            "opt_ms": 8.0,
            "step_ms": 107.0,
            "tokens_per_s": 1000.0 + i,
            "fetch_frac": 7.0 / 107.0,
            "queue_ms": 5.0,
            "h2d_ms": 2.0,
            "clip_ms": 2.0,
            "memory_mb": {
                "queue": 100.0,
                "fetch": 110.0,
                "fwd": 200.0,
                "pre_clip": 300.0,
                "bwd": 150.0,
                "opt": 160.0,
            },
        }
        for i in range(n)
    ]


def _make_report_result(per_step=None, kernels=True, verifier="Qwen/Qwen3-8B"):
    """A complete, current-version result dict for the report command."""
    per_step = _make_per_step() if per_step is None else per_step
    measured = [{k: v for k, v in s.items() if k != "step"} for s in per_step]
    result = {
        "benchmark_version": BENCHMARK_VERSION,
        "provenance": {
            "git_sha": "abcdef1234567890",
            "timestamp": "2026-01-01T00:00:00+00:00",
            "gpu_info": [{"name": "H100", "total_memory_gb": 80.0}],
        },
        "config": {
            "speculator_type": "eagle3",
            "verifier_name_or_path": verifier,
            "total_seq_len": 8192,
            "num_gpus_used": 1,
            "optimizer": "muon",
            "hidden_states_dtype": "bfloat16",
            "synthetic_data": True,
            "warmup_steps": 2,
            "measured_steps": len(per_step),
            "profiled": kernels,
        },
        "memory": {
            "per_rank": [
                {"rank": 0, "peak_allocated_mb": 2048.0, "peak_reserved_mb": 3072.0}
            ],
            "phases": {"fwd": 200.0, "pre_clip": 300.0},
        },
        "timing": _aggregate_timing(measured),
        "aggregate": compute_aggregate_throughput(measured),
        "per_step": per_step,
    }
    if kernels:
        rows = [{"name": "bigKernel", "ms_per_step": 95.0, "calls_per_step": 2.0}]
        rows += [
            {"name": f"tiny{i}", "ms_per_step": 1.0, "calls_per_step": 1.0}
            for i in range(5)
        ]
        result["kernels"] = {
            "total_device_ms_per_step": sum(k["ms_per_step"] for k in rows),
            "kernels": rows,
            "num_ranks": 1,
        }
    return result


class TestReportPaths:
    def test_default_names_derive_from_the_result_file(self, tmp_path):
        html, md = _report_paths(str(tmp_path / "run.json"), None)
        assert html == tmp_path / "run_report.html"
        assert md == tmp_path / "run_report.md"

    def test_uppercase_extension_still_yields_distinct_outputs(self, tmp_path):
        html, md = _report_paths(str(tmp_path / "RUN.JSON"), None)
        assert html == tmp_path / "RUN_report.html"
        assert md == tmp_path / "RUN_report.md"

    def test_explicit_output(self, tmp_path):
        html, md = _report_paths(str(tmp_path / "run.json"), str(tmp_path / "r.html"))
        assert html == tmp_path / "r.html"
        assert md == tmp_path / "r.md"

    def test_markdown_output_would_clobber_the_html(self, tmp_path):
        with pytest.raises(SystemExit, match="collide"):
            _report_paths(str(tmp_path / "run.json"), str(tmp_path / "out.md"))

    def test_output_over_the_input_rejected(self, tmp_path):
        with pytest.raises(SystemExit, match="collide"):
            _report_paths(str(tmp_path / "run.json"), str(tmp_path / "run.json"))


class TestPhaseMeans:
    def test_clip_is_subtracted_from_backward(self):
        means = {
            label: mean for _, label, mean, _ in _get_phase_means(_make_per_step())
        }
        assert means["Backward"] == pytest.approx(60.0)  # 62 reported - 2 clip
        assert means["Grad clip"] == pytest.approx(2.0)

    def test_phases_sum_to_the_step(self):
        per_step = _make_per_step()
        total = sum(mean for _, _, mean, _ in _get_phase_means(per_step))
        assert total == pytest.approx(per_step[0]["step_ms"])


class TestKernelPie:
    def test_small_kernels_merge_into_one_other_slice(self):
        result = _make_report_result()
        fig = _build_kernel_pie(result["kernels"])
        labels = list(fig.data[0].labels)
        values = list(fig.data[0].values)
        assert labels[0] == "bigKernel"
        assert labels[-1] == "Other (5 kernels)"  # each tiny kernel is 1% of 100
        assert values[-1] == pytest.approx(5.0)


class TestReportBenchmark:
    def _write(self, tmp_path, result):
        path = tmp_path / "run.json"
        path.write_text(json.dumps(result))
        return path

    def test_writes_html_and_markdown(self, tmp_path, capsys):
        path = self._write(tmp_path, _make_report_result())
        report_benchmark(str(path))

        html = (tmp_path / "run_report.html").read_text()
        md = (tmp_path / "run_report.md").read_text()
        assert json.loads(path.read_text())  # input untouched
        # The plotly bundle is inlined so the report renders without a network.
        assert "cdn.plot.ly/plotly" not in html
        assert "Plotly.newPlot" in html
        assert "GPU Kernel Breakdown" in html
        assert "| 0 | 2048 | 3072 |" in md
        assert "bigKernel" in md
        assert "Grad clip" in md
        out = capsys.readouterr().out
        assert "run_report.html" in out
        assert "run_report.md" in out

    def test_unprofiled_run_omits_the_kernel_sections(self, tmp_path):
        path = self._write(tmp_path, _make_report_result(kernels=False))
        report_benchmark(str(path))

        html = (tmp_path / "run_report.html").read_text()
        md = (tmp_path / "run_report.md").read_text()
        assert "GPU Kernel Breakdown" not in html
        assert "GPU Kernels" not in md
        assert "Step Timing Breakdown" in html

    def test_metadata_is_escaped(self, tmp_path):
        attack = "<script>alert(1)</script>"
        path = self._write(tmp_path, _make_report_result(verifier=attack))
        report_benchmark(str(path))

        html = (tmp_path / "run_report.html").read_text()
        assert attack not in html
        assert "&lt;script&gt;" in html

    def test_old_result_file_rejected(self, tmp_path):
        result = _make_report_result()
        result["benchmark_version"] = "1.0"
        path = self._write(tmp_path, result)
        with pytest.raises(SystemExit, match="Incompatible benchmark version"):
            report_benchmark(str(path))

    def test_missing_per_step_exits(self, tmp_path):
        result = _make_report_result()
        del result["per_step"]
        path = self._write(tmp_path, result)
        with pytest.raises(SystemExit):
            report_benchmark(str(path))
