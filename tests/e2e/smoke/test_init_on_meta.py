"""E2E: ``--init-on-meta`` is a pure memory optimization (result unchanged).

Launches the trainer's real build + broadcast-materialize path under torchrun and
asserts that non-rank0 ranks build the draft on the meta device, then materialize --
via ``set_model_state_dict(broadcast_from_rank0=True)`` -- real, finite weights that
are bit-equal to rank0. See ``init_on_meta_worker.py`` for the assertions.

Runs on 2 GPUs; skipped when fewer than 2 are available.
"""

import subprocess
import sys
from pathlib import Path

import pytest
from loguru import logger

from tests.conftest import requires_multi_gpu

WORKER = Path(__file__).resolve().parent / "init_on_meta_worker.py"


@pytest.mark.e2e
@pytest.mark.slow
@requires_multi_gpu
def test_init_on_meta_matches_rank0(tmp_path: Path):
    cmd = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--nproc_per_node",
        "2",
        str(WORKER),
        str(tmp_path),  # shared dir for the tiny verifier (ranks share the fs)
    ]
    logger.info("Running --init-on-meta distributed check: {}", " ".join(cmd))
    result = subprocess.run(  # noqa: S603
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, check=False
    )
    logger.info("worker output:\n{}", result.stdout)
    assert result.returncode == 0, f"--init-on-meta check failed:\n{result.stdout}"
    assert "PASS" in result.stdout, f"worker did not report PASS:\n{result.stdout}"
