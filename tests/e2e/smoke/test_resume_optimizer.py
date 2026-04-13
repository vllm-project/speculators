"""E2E regression test: resume from checkpoint with checkpoint_best symlink.

Verifies that distributed training can resume from a checkpoint when
checkpoint_best exists.

Steps:
  1. Prepare data
  2. Generate hidden states offline (via vLLM server)
  3. Train for 1 epoch with --save-best (creates checkpoint_best symlink)
  4. Resume training for epoch 2
"""

import subprocess
import sys
from pathlib import Path

import pytest
from loguru import logger

from tests.e2e.utils import (
    SCRIPTS_DIR,
    launch_vllm_server_context,
    run_data_generation_offline2,
    run_prepare_data,
)

MODEL = "Qwen/Qwen3-0.6B"
VLLM_PORT = 8323


def _run_distributed_training(
    data_path: Path,
    hidden_states_path: Path,
    save_path: Path,
    *,
    epochs: int,
    nproc: int = 2,
) -> subprocess.CompletedProcess:
    """Run distributed training via torchrun with --save-best."""
    cmd = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--nproc_per_node",
        str(nproc),
        str(SCRIPTS_DIR / "train.py"),
        "--verifier-name-or-path",
        MODEL,
        "--speculator-type",
        "eagle3",
        "--data-path",
        str(data_path),
        "--hidden-states-path",
        str(hidden_states_path),
        "--save-path",
        str(save_path),
        "--draft-vocab-size",
        "8192",
        "--lr",
        "3e-4",
        "--total-seq-len",
        "512",
        "--on-missing",
        "raise",
        "--save-best",
        "--epochs",
        str(epochs),
    ]
    logger.info("Running distributed training: {}", " ".join(cmd))
    return subprocess.run(  # noqa: S603
        cmd, stderr=subprocess.PIPE, text=True, check=False
    )


@pytest.mark.e2e
@pytest.mark.slow
def test_resume_after_checkpoint_best(tmp_path: Path):
    data_path = tmp_path / "data"
    hidden_states_path = tmp_path / "offline_hidden_states"
    save_path = tmp_path / "checkpoints"

    # Step 1: Prepare data
    run_prepare_data(MODEL, data_path)

    # Step 2: Generate hidden states offline
    with launch_vllm_server_context(MODEL, VLLM_PORT, str(tmp_path / "hidden_states")):
        run_data_generation_offline2(data_path, hidden_states_path, port=VLLM_PORT)

    # Step 3: Train 1 epoch with --save-best
    result = _run_distributed_training(
        data_path, hidden_states_path, save_path, epochs=1
    )
    assert result.returncode == 0, f"Training epoch 0 failed:\n{result.stderr}"

    checkpoint_best = save_path / "checkpoint_best"
    assert checkpoint_best.is_symlink(), "checkpoint_best symlink was not created"
    logger.info("checkpoint_best -> {}", checkpoint_best.resolve())

    # Step 4: Resume training for epoch 2
    result = _run_distributed_training(
        data_path, hidden_states_path, save_path, epochs=2
    )
    assert result.returncode == 0, (
        f"Resume from checkpoint failed (optimizer loading bug?):\n{result.stderr}"
    )
