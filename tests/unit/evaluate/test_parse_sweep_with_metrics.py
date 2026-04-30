"""Unit tests for parse_sweep_with_metrics.py script."""

import importlib.util
import json
import sys
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
import requests

# Import the script as a module
SCRIPT_PATH = (
    Path(__file__).parents[3]
    / "examples"
    / "evaluate"
    / "perf-benchmark"
    / "scripts"
    / "parse_sweep_with_metrics.py"
)
spec = importlib.util.spec_from_file_location("parse_sweep_with_metrics", SCRIPT_PATH)
assert spec is not None
assert spec.loader is not None
parse_sweep_with_metrics = importlib.util.module_from_spec(spec)
sys.modules["parse_sweep_with_metrics"] = parse_sweep_with_metrics
spec.loader.exec_module(parse_sweep_with_metrics)

Counter = parse_sweep_with_metrics.Counter
Vector = parse_sweep_with_metrics.Vector


class TestParsePrometheusMetrics:
    """Test prometheus metrics parsing."""

    def test_parse_empty_text(self):
        """Test parsing empty text returns empty list."""
        metrics = parse_sweep_with_metrics.parse_prometheus_metrics("")
        assert metrics == []

    def test_parse_simple_counter(self):
        """Test parsing simple counter metrics."""
        raw_text = "vllm:spec_decode_num_drafts 100.0"
        metrics = parse_sweep_with_metrics.parse_prometheus_metrics(raw_text)

        assert len(metrics) == 1
        assert isinstance(metrics[0], Counter)
        assert metrics[0].value == 100.0

    def test_parse_removes_total_suffix(self):
        """Test that _total suffix is removed from metric names."""
        raw_text = "vllm:spec_decode_num_drafts_total 100.0"
        metrics = parse_sweep_with_metrics.parse_prometheus_metrics(raw_text)

        assert len(metrics) == 1
        assert metrics[0].name == "vllm:spec_decode_num_drafts"

    def test_parse_labeled_metric_removes_total(self):
        """Test that _total suffix is removed from labeled metrics."""
        raw_text = 'vllm:spec_decode_num_drafts_total{model="test"} 100.0'
        metrics = parse_sweep_with_metrics.parse_prometheus_metrics(raw_text)

        assert len(metrics) == 1
        assert metrics[0].name == "vllm:spec_decode_num_drafts"

    def test_parse_per_pos_metric(self):
        """Test parsing per-position vector metrics."""
        raw_text = """
vllm:spec_decode_num_accepted_tokens_per_pos{position="0"} 90.0
vllm:spec_decode_num_accepted_tokens_per_pos{position="2"} 50.0
vllm:spec_decode_num_accepted_tokens_per_pos{position="1"} 70.0
"""
        metrics = parse_sweep_with_metrics.parse_prometheus_metrics(raw_text)

        assert len(metrics) == 1
        assert isinstance(metrics[0], Vector)
        # Should be sorted by position and fill gaps with 0
        assert metrics[0].values == [90.0, 70.0, 50.0]


class TestExtractSpecDecodeMetrics:
    """Test speculative decode metrics extraction."""

    def test_extract_basic_metrics(self):
        """Test extracting basic metrics without baseline."""
        raw_metrics = [
            Counter(name="vllm:spec_decode_num_drafts", value=100.0),
            Counter(name="vllm:spec_decode_num_draft_tokens", value=500.0),
            Counter(name="vllm:spec_decode_num_accepted_tokens", value=400.0),
        ]

        result = parse_sweep_with_metrics.extract_spec_decode_metrics(raw_metrics)

        assert result["num_drafts"] == 100.0
        assert result["num_draft_tokens"] == 500.0
        assert result["num_accepted_tokens"] == 400.0
        # acceptance_length = 1 + (400 / 100) = 5.0
        assert result["acceptance_length"] == 5.0

    def test_extract_with_per_position(self):
        """Test extracting per-position acceptance rates."""
        raw_metrics = [
            Counter(name="vllm:spec_decode_num_drafts", value=100.0),
            Vector(
                name="vllm:spec_decode_num_accepted_tokens_per_pos",
                values=[80.0, 60.0],
            ),
        ]

        result = parse_sweep_with_metrics.extract_spec_decode_metrics(raw_metrics)

        assert result["acceptance_at_pos_0"] == 0.8
        assert result["acceptance_at_pos_1"] == 0.6

    def test_extract_with_baseline_subtraction(self):
        """Test extracting deltas with baseline subtraction."""
        current_metrics = [
            Counter(name="vllm:spec_decode_num_drafts", value=200.0),
            Counter(name="vllm:spec_decode_num_draft_tokens", value=1000.0),
            Counter(name="vllm:spec_decode_num_accepted_tokens", value=800.0),
        ]
        baseline_metrics = [
            Counter(name="vllm:spec_decode_num_drafts", value=100.0),
            Counter(name="vllm:spec_decode_num_draft_tokens", value=500.0),
            Counter(name="vllm:spec_decode_num_accepted_tokens", value=400.0),
        ]

        result = parse_sweep_with_metrics.extract_spec_decode_metrics(
            current_metrics, baseline_metrics
        )

        # Should return deltas
        assert result["num_drafts"] == 100.0  # 200 - 100
        assert result["num_draft_tokens"] == 500.0  # 1000 - 500
        assert result["num_accepted_tokens"] == 400.0  # 800 - 400

    def test_extract_zero_drafts_no_division_error(self):
        """Test that zero drafts doesn't cause division by zero."""
        raw_metrics = [
            Counter(name="vllm:spec_decode_num_drafts", value=0.0),
        ]

        result = parse_sweep_with_metrics.extract_spec_decode_metrics(raw_metrics)

        assert result["acceptance_length"] == 0  # Should handle gracefully

    def test_extract_handles_wrong_types(self):
        """Test that mismatched metric types are silently skipped."""
        raw_metrics = [
            Counter(name="vllm:spec_decode_num_drafts", value=100.0),
            # Wrong type - should be skipped
            Vector(name="vllm:spec_decode_num_draft_tokens", values=[500.0]),
        ]

        result = parse_sweep_with_metrics.extract_spec_decode_metrics(raw_metrics)

        assert result["num_drafts"] == 100.0
        assert result["num_draft_tokens"] == 0.0  # Skipped, defaults to 0


class TestFetchVLLMMetrics:
    """Test vLLM metrics fetching."""

    @patch("parse_sweep_with_metrics.requests.get")
    def test_fetch_success(self, mock_get):
        """Test successful fetch."""
        mock_response = Mock()
        mock_response.text = "vllm:spec_decode_num_drafts 100.0"
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response

        result = parse_sweep_with_metrics.fetch_vllm_metrics("http://localhost:8000")

        assert result == "vllm:spec_decode_num_drafts 100.0"

    @patch("parse_sweep_with_metrics.requests.get")
    def test_fetch_error_returns_empty_string(self, mock_get):
        """Test that fetch errors return empty string instead of raising."""
        mock_get.side_effect = requests.exceptions.RequestException("Connection error")

        result = parse_sweep_with_metrics.fetch_vllm_metrics("http://localhost:8000")

        assert result == ""


class TestExtractSubsetName:
    """Test subset name extraction from filenames."""

    def test_extract_from_sweep_filename(self):
        """Test extracting subset from sweep_*.json filename."""
        path = Path("sweep_HumanEval.json")
        result = parse_sweep_with_metrics.extract_subset_name(path)
        assert result == "HumanEval"

    def test_extract_from_sweeps_filename(self):
        """Test extracting subset from sweeps_*.json filename."""
        path = Path("sweeps_MBPP.json")
        result = parse_sweep_with_metrics.extract_subset_name(path)
        assert result == "MBPP"

    def test_extract_fallback_to_stem(self):
        """Test fallback to stem if pattern doesn't match."""
        path = Path("custom_results.json")
        result = parse_sweep_with_metrics.extract_subset_name(path)
        assert result == "custom_results"


class TestParseSweepFile:
    """Test parsing sweep JSON files."""

    def test_parse_basic_sweep(self, tmp_path):
        """Test parsing a basic sweep file."""
        sweep_data = {
            "benchmarks": [
                {
                    "config": {
                        "strategy": {"type_": "constant_arrival_rate", "rate": 10}
                    },
                    "metrics": {
                        "requests_per_second": {"successful": {"median": 9.5}},
                        "request_latency": {"successful": {"median": 0.5}},
                        "inter_token_latency_ms": {"successful": {"median": 10.0}},
                        "time_to_first_token_ms": {"successful": {"median": 50.0}},
                        "output_tokens_per_second": {"successful": {"median": 100.0}},
                        "output_tokens": {"successful": {"sum": 5000}},
                    },
                }
            ]
        }

        sweep_file = tmp_path / "sweep_test.json"
        sweep_file.write_text(json.dumps(sweep_data))

        rows = parse_sweep_with_metrics.parse_sweep_file(sweep_file)

        assert len(rows) == 1
        row = rows[0]
        assert row["subset"] == "test"
        assert row["strategy"] == "constant_arrival_rate"
        assert row["target_rate"] == 10
        assert row["rps_median"] == 9.5
        assert row["latency_median_s"] == 0.5
        assert row["itl_median_ms"] == 10.0
        assert row["ttft_median_ms"] == 50.0
        assert row["output_tps_median"] == 100.0
        assert row["total_output_tokens"] == 5000

    def test_parse_skips_throughput_strategy(self, tmp_path):
        """Test that throughput strategy is skipped."""
        sweep_data = {
            "benchmarks": [
                {
                    "config": {"strategy": {"type_": "throughput"}},
                    "metrics": {},
                },
                {
                    "config": {"strategy": {"type_": "constant_arrival_rate"}},
                    "metrics": {
                        "requests_per_second": {"successful": {"median": 9.5}},
                        "request_latency": {"successful": {"median": 0.5}},
                        "inter_token_latency_ms": {"successful": {"median": 10.0}},
                        "time_to_first_token_ms": {"successful": {"median": 50.0}},
                        "output_tokens_per_second": {"successful": {"median": 100.0}},
                    },
                },
            ]
        }

        sweep_file = tmp_path / "sweep_test.json"
        sweep_file.write_text(json.dumps(sweep_data))

        rows = parse_sweep_with_metrics.parse_sweep_file(sweep_file)

        # Should only have 1 row (throughput skipped)
        assert len(rows) == 1
        assert rows[0]["strategy"] == "constant_arrival_rate"

    def test_parse_no_benchmarks_raises_error(self, tmp_path):
        """Test that missing benchmarks raises ValueError."""
        sweep_data: dict[str, list] = {"benchmarks": []}

        sweep_file = tmp_path / "sweep_test.json"
        sweep_file.write_text(json.dumps(sweep_data))

        with pytest.raises(ValueError, match="No benchmarks found"):
            parse_sweep_with_metrics.parse_sweep_file(sweep_file)
