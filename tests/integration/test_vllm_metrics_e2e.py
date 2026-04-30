"""Smoke test for vLLM metrics scripts.

To run this test, you need a vLLM server running with speculative decoding enabled.

Example:
    # Start vLLM server with spec decode (in another terminal)
    vllm serve meta-llama/Llama-3.2-1B --speculative-model meta-llama/Llama-3.2-1B \\
        --num-speculative-tokens 5

    # Run the test
    pytest tests/integration/test_vllm_metrics_e2e.py --vllm-url http://localhost:8000
"""

import json
import subprocess
import sys
from pathlib import Path

import pytest
import requests


def pytest_addoption(parser):
    """Add custom pytest options."""
    parser.addoption(
        "--vllm-url",
        action="store",
        default=None,
        help="vLLM server URL for smoke tests (e.g., http://localhost:8000)",
    )


@pytest.fixture
def vllm_url(request):
    """Get vLLM URL from pytest option."""
    url = request.config.getoption("--vllm-url")
    if not url:
        pytest.skip("No --vllm-url provided, skipping vLLM integration tests")
    return url


@pytest.fixture
def vllm_available(vllm_url):
    """Check if vLLM server is available."""
    try:
        response = requests.get(f"{vllm_url}/health", timeout=5)
        response.raise_for_status()
        return True
    except (requests.exceptions.RequestException, Exception):  # noqa: BLE001
        pytest.skip(f"vLLM server not available at {vllm_url}")
        return False


@pytest.mark.integration
class TestVLLMMetricsE2E:
    """End-to-end smoke tests for vLLM metrics scripts."""

    def test_vllm_server_has_metrics_endpoint(self, vllm_url, vllm_available):
        """Verify vLLM server has /metrics endpoint."""
        response = requests.get(f"{vllm_url}/metrics", timeout=10)
        response.raise_for_status()
        assert len(response.text) > 0
        assert "vllm:" in response.text or "# HELP" in response.text

    def test_fetch_vllm_metrics_script(self, vllm_url, vllm_available, tmp_path):
        """Test fetch_vllm_metrics.py script against real vLLM server."""
        script_path = (
            Path(__file__).parents[2]
            / "examples"
            / "evaluate"
            / "perf-benchmark"
            / "scripts"
            / "fetch_vllm_metrics.py"
        )

        output_file = tmp_path / "metrics.json"

        result = subprocess.run(  # noqa: S603
            [
                sys.executable,
                str(script_path),
                "--url",
                vllm_url,
                "--output",
                str(output_file),
                "--total-tokens",
                "1000",
            ],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )

        # Check script succeeded
        assert result.returncode == 0, f"Script failed: {result.stderr}"

        # Check output was created
        assert output_file.exists(), "Output file was not created"

        # Validate output structure
        with output_file.open() as f:
            metrics = json.load(f)

        # Verify expected keys exist
        assert "total_num_output_tokens" in metrics
        assert metrics["total_num_output_tokens"] == 1000

        # Should have spec decode metrics (even if zero)
        assert "num_drafts" in metrics
        assert "num_draft_tokens" in metrics
        assert "num_accepted_tokens" in metrics
        assert "acceptance_length" in metrics

    def test_parse_sweep_with_vllm_metrics(self, vllm_url, vllm_available, tmp_path):
        """Test parse_sweep_with_metrics.py script integration."""
        # Create a minimal sweep file
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

        output_csv = tmp_path / "results.csv"

        script_path = (
            Path(__file__).parents[2]
            / "examples"
            / "evaluate"
            / "perf-benchmark"
            / "scripts"
            / "parse_sweep_with_metrics.py"
        )

        result = subprocess.run(  # noqa: S603
            [
                sys.executable,
                str(script_path),
                str(sweep_file),
                "--output",
                str(output_csv),
                "--vllm-url",
                vllm_url,
            ],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )

        # Check script succeeded
        assert result.returncode == 0, f"Script failed: {result.stderr}"

        # Verify CSV was created
        assert output_csv.exists(), "CSV output was not created"

        # Verify CSV has content
        csv_content = output_csv.read_text()
        lines = csv_content.strip().split("\n")
        assert len(lines) >= 2, "CSV should have header + at least one row"

        # Verify header has expected columns
        header = lines[0]
        assert "subset" in header
        assert "strategy" in header
        assert "rps_median" in header

        # Should include spec decode metrics if vLLM has them
        # (even if they're zero/empty)
        data_row = lines[1]
        assert len(data_row) > 0
