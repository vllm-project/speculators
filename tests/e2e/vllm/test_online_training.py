"""E2E test for the online training workflow.

Exercises the full pipeline documented in examples/ONLINE_TRAINING.md:
  1. Prepare data (scripts/prepare_data.py)
  2. Launch a vLLM server for hidden-state extraction (scripts/launch_vllm.py)
  3. Train a draft model against the live server (scripts/train.py)
  4. Validate the trained checkpoint via vLLM inference (run_vllm_engine)
"""

import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import pytest
from loguru import logger

from tests.e2e.vllm.utils import run_vllm_engine

MODEL = "Qwen/Qwen3-0.6B"
VLLM_PORT = 8321
VLLM_PYTHON = os.environ.get("VLLM_PYTHON", sys.executable)
SCRIPTS_DIR = Path(__file__).resolve().parent.parent.parent.parent / "scripts"


def wait_for_server(port: int, timeout: float = 180.0, poll_interval: float = 2.0):
    """Poll vLLM server health endpoint until ready or timeout."""
    url = f"http://localhost:{port}/health"
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=5) as resp:  # noqa: S310
                if resp.status == 200:
                    return
        except (urllib.error.URLError, ConnectionError, OSError):
            pass
        time.sleep(poll_interval)
    raise TimeoutError(f"vLLM server on port {port} not ready after {timeout}s")


@pytest.fixture
def vllm_server(tmp_path):
    """Launch a vLLM server configured for hidden-state extraction."""
    hidden_states_path = str(tmp_path / "hidden_states")

    cmd = [
        VLLM_PYTHON,
        str(SCRIPTS_DIR / "launch_vllm.py"),
        MODEL,
        "--hidden-states-path",
        hidden_states_path,
        "--",
        "--port",
        str(VLLM_PORT),
        "--max-model-len",
        "513",
        "--gpu-memory-utilization",
        "0.5",
    ]
    logger.info("Starting vLLM server: {}", " ".join(cmd))

    process = subprocess.Popen(cmd)  # noqa: S603

    try:
        wait_for_server(VLLM_PORT)
        logger.info("vLLM server ready on port {}", VLLM_PORT)
    except Exception:
        process.terminate()
        process.wait(timeout=30)
        raise

    yield {"port": VLLM_PORT, "hidden_states_path": hidden_states_path, "process": process}

    # Ensure cleanup if the test didn't stop the server itself
    if process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=10)
        logger.info("vLLM server stopped (fixture teardown)")


@pytest.mark.e2e
@pytest.mark.slow
def test_online_training(
    tmp_path: Path, prompts: list[list[dict[str, str]]], vllm_server
):
    data_path = tmp_path / "data"
    save_path = tmp_path / "checkpoints"
    port = vllm_server["port"]

    # Step 1: Prepare data
    prepare_cmd = [
        sys.executable,
        str(SCRIPTS_DIR / "prepare_data.py"),
        "--model",
        MODEL,
        "--data",
        "sharegpt",
        "--output",
        str(data_path),
        "--max-samples",
        "50",
        "--seq-length",
        "512",
    ]
    logger.info("Preparing data: {}", " ".join(prepare_cmd))
    result = subprocess.run(  # noqa: S603
        prepare_cmd, stderr=subprocess.PIPE, text=True, check=False
    )
    assert result.returncode == 0, f"prepare_data.py failed:\n{result.stderr}"

    # Step 2: Train against live vLLM server
    train_cmd = [
        sys.executable,
        str(SCRIPTS_DIR / "train.py"),
        "--verifier-name-or-path",
        MODEL,
        "--data-path",
        str(data_path),
        "--vllm-endpoint",
        f"http://localhost:{port}/v1",
        "--save-path",
        str(save_path),
        "--draft-vocab-size",
        "8192",
        "--epochs",
        "1",
        "--lr",
        "3e-4",
        "--total-seq-len",
        "512",
        "--on-missing",
        "generate",
        "--on-generate",
        "delete",
    ]
    logger.info("Running training: {}", " ".join(train_cmd))
    result = subprocess.run(  # noqa: S603
        train_cmd, stderr=subprocess.PIPE, text=True, check=False
    )
    assert result.returncode == 0, f"train.py failed:\n{result.stderr}"

    # Stop the vLLM server to free GPU memory before running inference
    server_process = vllm_server["process"]
    server_process.terminate()
    try:
        server_process.wait(timeout=30)
    except subprocess.TimeoutExpired:
        server_process.kill()
        server_process.wait(timeout=10)
    logger.info("vLLM server stopped before inference validation")

    # Step 3: Validate trained checkpoint with vLLM inference
    checkpoint_path = str(save_path / "0")
    run_vllm_engine(model_path=checkpoint_path, tmp_path=tmp_path, prompts=prompts)
