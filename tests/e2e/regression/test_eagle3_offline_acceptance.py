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

import pytest

from tests.e2e.smoke.test_offline_training import run_offline_e2e
from tests.utils import requires_cadence


@requires_cadence("nightly")
@pytest.mark.parametrize(
    ("model", "dataset", "acceptance_thresholds"),
    [
        ("Qwen/Qwen3-8B", "sharegpt", [0.4, 0.07, 0.007]),
        ("Qwen/Qwen3-VL-2B-Instruct", "sharegpt4v_coco", [0.4, 0.07, 0.007]),
    ],
)
def test_offline_regression(
    tmp_path: Path,
    model: str,
    dataset: str,
    acceptance_thresholds: list[float],
    prompts: list[list[dict[str, str]]],
):
    run_offline_e2e(
        tmp_path,
        model,
        dataset,
        max_samples=5000,
        seq_length=8192,
        vllm_gpu_util=0.9,
        vllm_enforce_eager=dataset == "sharegpt4v_coco",
        epochs=3,
        prompts=prompts,
        acceptance_thresholds=acceptance_thresholds,
        log_freq=50,
    )
