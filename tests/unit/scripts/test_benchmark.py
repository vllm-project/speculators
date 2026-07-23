"""Unit tests for the benchmark harness."""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch

# Add scripts/ to the import path the same way the benchmark script does.
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts"))

from benchmark import (  # type: ignore[import-not-found]
    _MetricCapture,
    _SyntheticLoader,
    collect_provenance,
    compare_benchmarks,
    compute_statistics,
    create_synthetic_batch,
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
        }
        assert set(batch.keys()) == expected_keys


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


class TestWarmupMeasuredSplit:
    """Tests for the profile slicing logic used in run_benchmark."""

    def test_discard_warmup(self):
        warmup_steps = 3
        all_profiles = [{"step_ms": float(i)} for i in range(13)]
        measured = all_profiles[warmup_steps:]
        assert len(measured) == 10
        assert measured[0]["step_ms"] == 3.0

    def test_exact_boundary(self):
        warmup_steps = 5
        all_profiles = [{"step_ms": float(i)} for i in range(5)]
        measured = all_profiles[warmup_steps:]
        assert len(measured) == 0

    def test_zero_warmup(self):
        warmup_steps = 0
        all_profiles = [{"step_ms": float(i)} for i in range(10)]
        measured = all_profiles[warmup_steps:]
        assert len(measured) == 10
        assert measured[0]["step_ms"] == 0.0

    def test_fallback_on_insufficient_profiles(self):
        """When fewer profiles than expected, fallback uses all of them."""
        warmup_steps = 10
        all_profiles = [{"step_ms": float(i)} for i in range(5)]
        measured = all_profiles[warmup_steps:]
        if not measured:
            measured = all_profiles
        assert len(measured) == 5


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
            "peak_allocated_mb": peak_alloc,
            "peak_reserved_mb": peak_alloc + 1024,
        },
        "timing": timing,
    }


class TestCompareBenchmarks:
    def test_basic_compare(self, tmp_path, capsys):
        baseline = _make_result(step_ms_mean=50.0, git_sha="aaa111")
        candidate = _make_result(step_ms_mean=45.0, git_sha="bbb222")

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
        assert "peak_allocated_mb" in output
        assert "-200.0" in output
