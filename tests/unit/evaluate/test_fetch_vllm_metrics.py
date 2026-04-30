"""Unit tests for fetch_vllm_metrics.py script."""

import importlib.util
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
    / "fetch_vllm_metrics.py"
)
spec = importlib.util.spec_from_file_location("fetch_vllm_metrics", SCRIPT_PATH)
assert spec is not None
assert spec.loader is not None
fetch_vllm_metrics = importlib.util.module_from_spec(spec)
sys.modules["fetch_vllm_metrics"] = fetch_vllm_metrics
spec.loader.exec_module(fetch_vllm_metrics)

Counter = fetch_vllm_metrics.Counter
Vector = fetch_vllm_metrics.Vector


class TestParsePrometheusMetrics:
    """Test prometheus metrics parsing."""

    def test_parse_simple_counter(self):
        """Test parsing simple counter metrics."""
        raw_text = """
# HELP vllm:spec_decode_num_drafts Number of draft proposals
# TYPE vllm:spec_decode_num_drafts counter
vllm:spec_decode_num_drafts 100.0
"""
        metrics = fetch_vllm_metrics.parse_prometheus_metrics(raw_text)

        assert len(metrics) == 1
        assert isinstance(metrics[0], Counter)
        assert metrics[0].name == "vllm:spec_decode_num_drafts"
        assert metrics[0].value == 100.0

    def test_parse_labeled_counter(self):
        """Test parsing labeled counter metrics.

        Note: The current implementation only parses labeled metrics for
        per_pos vector metrics. Other labeled counters are not parsed.
        This test documents the current behavior.
        """
        raw_text = """
vllm:spec_decode_num_drafts{model="llama"} 50.0
"""
        metrics = fetch_vllm_metrics.parse_prometheus_metrics(raw_text)

        # Current implementation doesn't parse labeled non-per_pos metrics
        assert len(metrics) == 0

    def test_parse_vector_metric(self):
        """Test parsing vector metrics with position labels."""
        raw_text = """
vllm:spec_decode_num_accepted_tokens_per_pos{position="0"} 100.0
vllm:spec_decode_num_accepted_tokens_per_pos{position="1"} 80.0
vllm:spec_decode_num_accepted_tokens_per_pos{position="2"} 60.0
"""
        metrics = fetch_vllm_metrics.parse_prometheus_metrics(raw_text)

        assert len(metrics) == 1
        assert isinstance(metrics[0], Vector)
        assert metrics[0].name == "vllm:spec_decode_num_accepted_tokens_per_pos"
        assert metrics[0].values == [100.0, 80.0, 60.0]

    def test_parse_multiple_metrics(self):
        """Test parsing multiple different metrics."""
        raw_text = """
vllm:spec_decode_num_drafts 100.0
vllm:spec_decode_num_draft_tokens 500.0
vllm:spec_decode_num_accepted_tokens 400.0
vllm:spec_decode_num_accepted_tokens_per_pos{position="0"} 90.0
vllm:spec_decode_num_accepted_tokens_per_pos{position="1"} 70.0
"""
        metrics = fetch_vllm_metrics.parse_prometheus_metrics(raw_text)

        assert len(metrics) == 4
        counters = [m for m in metrics if isinstance(m, Counter)]
        vectors = [m for m in metrics if isinstance(m, Vector)]

        assert len(counters) == 3
        assert len(vectors) == 1
        assert vectors[0].values == [90.0, 70.0]

    def test_parse_ignores_comments_and_blank_lines(self):
        """Test that comments and blank lines are ignored."""
        raw_text = """
# This is a comment
# TYPE vllm:spec_decode_num_drafts counter

vllm:spec_decode_num_drafts 100.0

# Another comment
"""
        metrics = fetch_vllm_metrics.parse_prometheus_metrics(raw_text)

        assert len(metrics) == 1

    def test_parse_non_spec_decode_metrics_ignored(self):
        """Test that non-spec_decode metrics are ignored."""
        raw_text = """
vllm:other_metric 123.0
vllm:spec_decode_num_drafts 100.0
some_other_metric 456.0
"""
        metrics = fetch_vllm_metrics.parse_prometheus_metrics(raw_text)

        assert len(metrics) == 1
        assert metrics[0].name == "vllm:spec_decode_num_drafts"

    def test_parse_empty_string(self):
        """Test parsing empty input."""
        metrics = fetch_vllm_metrics.parse_prometheus_metrics("")
        assert len(metrics) == 0


class TestExtractMetrics:
    """Test metrics extraction and calculation."""

    def test_extract_basic_metrics(self):
        """Test extracting basic spec decode metrics."""
        raw_metrics = [
            Counter(name="vllm:spec_decode_num_drafts", value=100.0),
            Counter(name="vllm:spec_decode_num_draft_tokens", value=500.0),
            Counter(name="vllm:spec_decode_num_accepted_tokens", value=400.0),
        ]

        result = fetch_vllm_metrics.extract_metrics(
            raw_metrics, total_num_output_tokens=1000
        )

        assert result["total_num_output_tokens"] == 1000
        assert result["num_drafts"] == 100.0
        assert result["num_draft_tokens"] == 500.0
        assert result["num_accepted_tokens"] == 400.0
        # acceptance_length = 1 + (400 / 100) = 5.0
        assert result["acceptance_length"] == 5.0

    def test_extract_with_vector_metrics(self):
        """Test extracting metrics with per-position acceptance rates."""
        raw_metrics = [
            Counter(name="vllm:spec_decode_num_drafts", value=100.0),
            Counter(name="vllm:spec_decode_num_draft_tokens", value=500.0),
            Counter(name="vllm:spec_decode_num_accepted_tokens", value=400.0),
            Vector(
                name="vllm:spec_decode_num_accepted_tokens_per_pos",
                values=[90.0, 70.0, 50.0],
            ),
        ]

        result = fetch_vllm_metrics.extract_metrics(
            raw_metrics, total_num_output_tokens=1000
        )

        assert result["acceptance_at_token_0"] == 0.9  # 90 / 100
        assert result["acceptance_at_token_1"] == 0.7  # 70 / 100
        assert result["acceptance_at_token_2"] == 0.5  # 50 / 100

    def test_extract_zero_drafts(self):
        """Test that zero drafts doesn't cause division by zero."""
        raw_metrics = [
            Counter(name="vllm:spec_decode_num_drafts", value=0.0),
            Counter(name="vllm:spec_decode_num_draft_tokens", value=0.0),
            Counter(name="vllm:spec_decode_num_accepted_tokens", value=0.0),
        ]

        result = fetch_vllm_metrics.extract_metrics(
            raw_metrics, total_num_output_tokens=1000
        )

        assert result["acceptance_length"] == 1  # Default to 1 when no drafts
        assert result["num_drafts"] == 0.0

    def test_extract_handles_wrong_metric_types(self):
        """Test that wrong metric types are silently skipped."""
        raw_metrics = [
            Counter(name="vllm:spec_decode_num_drafts", value=100.0),
            # This should be Counter but is Vector - should be skipped
            Vector(name="vllm:spec_decode_num_draft_tokens", values=[500.0]),
        ]

        result = fetch_vllm_metrics.extract_metrics(
            raw_metrics, total_num_output_tokens=1000
        )

        assert result["num_drafts"] == 100.0
        assert result["num_draft_tokens"] == 0  # Skipped, defaults to 0


class TestFetchMetrics:
    """Test HTTP fetching of metrics."""

    @patch("fetch_vllm_metrics.requests.get")
    def test_fetch_success(self, mock_get):
        """Test successful metrics fetch."""
        mock_response = Mock()
        mock_response.text = "vllm:spec_decode_num_drafts 100.0"
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response

        result = fetch_vllm_metrics.fetch_metrics("http://localhost:8000")

        assert result == "vllm:spec_decode_num_drafts 100.0"
        mock_get.assert_called_once_with("http://localhost:8000/metrics", timeout=10)

    @patch("fetch_vllm_metrics.requests.get")
    def test_fetch_with_custom_timeout(self, mock_get):
        """Test fetch with custom timeout."""
        mock_response = Mock()
        mock_response.text = "test"
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response

        fetch_vllm_metrics.fetch_metrics("http://localhost:8000", timeout=30)

        mock_get.assert_called_once_with("http://localhost:8000/metrics", timeout=30)

    @patch("fetch_vllm_metrics.requests.get")
    def test_fetch_handles_trailing_slash(self, mock_get):
        """Test that trailing slashes are handled correctly."""
        mock_response = Mock()
        mock_response.text = "test"
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response

        fetch_vllm_metrics.fetch_metrics("http://localhost:8000/")

        mock_get.assert_called_once_with("http://localhost:8000/metrics", timeout=10)

    @patch("fetch_vllm_metrics.requests.get")
    @patch("fetch_vllm_metrics.logger")
    def test_fetch_http_error_exits(self, mock_logger, mock_get):
        """Test that HTTP errors cause sys.exit."""
        mock_get.side_effect = requests.exceptions.RequestException("Connection error")

        with pytest.raises(SystemExit):
            fetch_vllm_metrics.fetch_metrics("http://localhost:8000")

        # Verify error was logged
        mock_logger.error.assert_called_once()


class TestFormatOutput:
    """Test output formatting."""

    def test_format_basic_metrics(self):
        """Test formatting basic metrics."""
        metrics = {
            "total_num_output_tokens": 1000,
            "num_drafts": 100,
            "num_draft_tokens": 500,
            "num_accepted_tokens": 400,
            "acceptance_length": 5.0,
        }

        result = fetch_vllm_metrics.format_output(metrics)

        assert "=== Speculative Decoding Metrics ===" in result
        assert "Total output tokens: 1000" in result
        assert "Number of drafts: 100" in result
        assert "Draft tokens proposed: 500" in result
        assert "Draft tokens accepted: 400" in result
        assert "Average acceptance length: 5.00" in result

    def test_format_with_per_position_rates(self):
        """Test formatting with per-position acceptance rates."""
        metrics = {
            "total_num_output_tokens": 1000,
            "num_drafts": 100,
            "num_draft_tokens": 500,
            "num_accepted_tokens": 400,
            "acceptance_length": 5.0,
            "acceptance_at_token_0": 0.9,
            "acceptance_at_token_1": 0.7,
            "acceptance_at_token_2": 0.5,
        }

        result = fetch_vllm_metrics.format_output(metrics)

        assert "Position 0: 0.9000 (90.00%)" in result
        assert "Position 1: 0.7000 (70.00%)" in result
        assert "Position 2: 0.5000 (50.00%)" in result
