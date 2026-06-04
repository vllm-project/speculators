"""Weekly regression test for MTP online finetuning acceptance rates.

Imports and calls the shared ``run_mtp_finetuning_e2e()`` pipeline from the
smoke module with Qwen3.5-9B parameters and placeholder acceptance thresholds.
"""

from pathlib import Path

import pytest

from tests.conftest import requires_cuda, requires_transformers_version, requires_vllm_version
from tests.e2e.smoke.test_mtp_finetuning import run_mtp_finetuning_e2e
from tests.utils import requires_cadence


@requires_cadence("weekly")
@pytest.mark.regression
@requires_cuda
@requires_transformers_version("5.2.0")
@requires_vllm_version("0.22.0")
def test_mtp_online_regression(
    tmp_path: Path,
    prompts: list[list[dict[str, str]]],
):
    run_mtp_finetuning_e2e(
        tmp_path=tmp_path,
        verifier="Qwen/Qwen3.5-9B",
        training_data_repo="inference-optimization/Qwen3.5-9B-responses",
        num_speculative_tokens=3,
        target_layer_ids=[32],
        max_samples=5000,
        seq_length=8192,
        epochs=3,
        lr=3e-4,
        prompts=prompts,
        # TODO: placeholder thresholds — calibrate from first successful run
        acceptance_thresholds=[0.3, 0.1, 0.01],
        train_timeout=60 * 60,
        gpu_memory_utilization=0.75,
    )
