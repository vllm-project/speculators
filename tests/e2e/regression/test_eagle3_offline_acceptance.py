"""E2E test for the offline training workflow.

Exercises the full offline pipeline:
  1. Prepare data (scripts/prepare_data.py)
  2. Launch a vLLM server for hidden-state extraction (scripts/launch_vllm.py)
  3. Generate hidden states offline (scripts/data_generation_offline.py)
  4. Stop the vLLM server
  5. Train a draft model using pre-generated hidden states (scripts/train.py)
  6. Validate the trained checkpoint via vLLM inference (run_vllm_engine)
"""

from pathlib import Path

from tests.e2e.smoke.test_offline_training import run_offline_e2e
from tests.utils import requires_cadence


@requires_cadence("nightly")
def test_offline_regression(tmp_path: Path, prompts):
    run_offline_e2e(
        tmp_path,
        "Qwen/Qwen3-8B",
        max_samples=5000,
        seq_length=8192,
        vllm_gpu_util=0.9,
        epochs=3,
        prompts=prompts,
        acceptance_thresholds=[0.4, 0.07, 0.007],
        log_freq=50,
    )
