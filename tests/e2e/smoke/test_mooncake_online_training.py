"""E2E smoke test for online training with the Mooncake hidden-states backend.

Same pipeline as test_online_training but uses Mooncake for hidden-state
transfer instead of a shared filesystem:
  1. Prepare data (scripts/prepare_data.py)
  2. Start a Mooncake master process
  3. Launch a vLLM server with --hidden-states-backend mooncake
  4. Train a draft model against the live server (scripts/train.py)
  5. Validate the trained checkpoint via vLLM inference (run_vllm_engine)
"""

import shutil
import subprocess
import time
from contextlib import contextmanager
from pathlib import Path

import pytest
from loguru import logger

from tests.e2e.utils import (
    launch_vllm_server_context,
    run_prepare_data,
    run_training,
    run_vllm_engine,
)

MODEL = "Qwen/Qwen3-0.6B"
MOONCAKE_MASTER_PORT = 50051
MOONCAKE_MASTER_ADDR = f"127.0.0.1:{MOONCAKE_MASTER_PORT}"


@contextmanager
def mooncake_master_context(port: int = MOONCAKE_MASTER_PORT):
    """Start and stop a mooncake_master process."""
    exe = shutil.which("mooncake_master")
    if exe is None:
        pytest.skip("mooncake_master not found on PATH")

    cmd = [exe, "--port", str(port)]
    logger.info("Starting mooncake_master: {}", " ".join(cmd))
    proc = subprocess.Popen(cmd)  # noqa: S603
    time.sleep(2)

    if proc.poll() is not None:
        raise RuntimeError(
            f"mooncake_master exited immediately with code {proc.returncode}"
        )

    try:
        yield
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
        logger.info("mooncake_master stopped (exit code {})", proc.returncode)


@pytest.mark.e2e
@pytest.mark.slow
def test_mooncake_online_smoke(
    tmp_path: Path,
    prompts: list[list[dict[str, str]]],
):
    data_path = tmp_path / "data"
    save_path = tmp_path / "checkpoints"
    seq_length = 512
    port = 8322

    run_prepare_data(
        MODEL, "sharegpt", data_path, max_samples=50, seq_length=seq_length
    )

    mooncake_kwargs = {
        "hidden_states_backend": "mooncake",
        "mooncake_master": MOONCAKE_MASTER_ADDR,
        "mooncake_metadata_server": "P2PHANDSHAKE",
        "mooncake_protocol": "tcp",
    }

    with (
        mooncake_master_context(),
        launch_vllm_server_context(
            MODEL,
            port,
            hidden_states_path=str(tmp_path / "hidden_states"),
            max_model_len=seq_length + 1,
            enforce_eager=True,
            **mooncake_kwargs,
        ),
    ):
        run_training(
            MODEL,
            data_path,
            save_path,
            seq_length,
            port,
            draft_vocab_size=8192,
            epochs=1,
            lr=3e-4,
            log_freq=1,
            timeout=30 * 60,
            **mooncake_kwargs,  # type: ignore[arg-type]
        )

    checkpoint_path = str(save_path / "checkpoint_best")
    run_vllm_engine(
        model_path=checkpoint_path,
        tmp_path=tmp_path,
        prompts=prompts,
        max_tokens=50,
        ignore_eos=True,
        enforce_eager=True,
    )
