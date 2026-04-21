"""E2E test for the online training workflow.

Exercises the full pipeline documented in examples/ONLINE_TRAINING.md:
  1. Prepare data (scripts/prepare_data.py)
  2. Launch a vLLM server for hidden-state extraction (scripts/launch_vllm.py)
  3. Train a draft model against the live server (scripts/train.py)
  4. Validate the trained checkpoint via vLLM inference (run_vllm_engine)
"""

from pathlib import Path

from tests.e2e.smoke.test_online_training import run_online_e2e
from tests.utils import requires_cadence


@requires_cadence("nightly")
def test_online_regression(tmp_path: Path, prompts):
    run_online_e2e(
        tmp_path,
        "Qwen/Qwen3-8B",
        max_samples=5000,
        seq_length=8192,
        vllm_gpu_util=0.75,
        epochs=3,
        prompts=prompts,
        acceptance_thresholds=[0.4, 0.1, 0.01],
    )
