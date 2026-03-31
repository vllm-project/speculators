"""E2E test for the online training workflow.

Exercises the full pipeline documented in examples/ONLINE_TRAINING.md:
  1. Prepare data (scripts/prepare_data.py)
  2. Launch a vLLM server for hidden-state extraction (scripts/launch_vllm.py)
  3. Train a draft model against the live server (scripts/train.py)
  4. Validate the trained checkpoint via vLLM inference (run_vllm_engine)
"""

import subprocess
import sys
from pathlib import Path

import pytest
from loguru import logger

from tests.e2e.vllm.utils import (
    SCRIPTS_DIR,
    launch_vllm_server,
    prepare_data,
    run_vllm_engine,
    stop_vllm_server,
)

MODEL = "Qwen/Qwen3-0.6B"
VLLM_PORT = 8321


@pytest.fixture
def vllm_server(tmp_path):
    """Launch a vLLM server configured for hidden-state extraction."""
    hidden_states_path = str(tmp_path / "hidden_states")
    process = launch_vllm_server(MODEL, VLLM_PORT, hidden_states_path)

    yield {"port": VLLM_PORT, "hidden_states_path": hidden_states_path, "process": process}

    stop_vllm_server(process)


@pytest.mark.e2e
@pytest.mark.slow
def test_online_training(
    tmp_path: Path, prompts: list[list[dict[str, str]]], vllm_server
):
    data_path = tmp_path / "data"
    save_path = tmp_path / "checkpoints"
    port = vllm_server["port"]

    # Step 1: Prepare data
    prepare_data(MODEL, data_path)

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
    stop_vllm_server(vllm_server["process"])

    # Step 3: Validate trained checkpoint with vLLM inference
    checkpoint_path = str(save_path / "0")
    run_vllm_engine(model_path=checkpoint_path, tmp_path=tmp_path, prompts=prompts)
