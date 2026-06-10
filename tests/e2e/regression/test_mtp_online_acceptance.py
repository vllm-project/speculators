"""Weekly regression test for MTP online finetuning acceptance rates.

Imports and calls the shared ``run_mtp_finetuning_e2e()`` pipeline from the
smoke module with Qwen3.5-4B parameters. Evaluates on generic (non-GSM8k)
prompts to verify finetuning doesn't cause catastrophic forgetting of the
base model's speculative decoding quality.

Base model acceptance rates on generic prompts (4B): ~88%, ~75%, ~59%.
Thresholds are set conservatively to catch catastrophic regressions.
"""

from pathlib import Path

import pytest

from tests.conftest import (
    requires_cuda,
    requires_transformers_version,
    requires_vllm_version,
)
from tests.e2e.smoke.test_mtp_finetuning import run_mtp_finetuning_e2e
from tests.utils import requires_cadence

TRAINING_DATA_REPO = "inference-optimization/Qwen3.5-4B-responses"


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
        verifier="Qwen/Qwen3.5-4B",
        training_data_repo=TRAINING_DATA_REPO,
        num_speculative_tokens=3,
        target_layer_ids=[32],
        max_samples=5000,
        seq_length=8192,
        epochs=1,
        lr=1e-5,
        prompts=prompts,
        acceptance_thresholds=[0.85, 0.70, 0.56],
        max_tokens=512,
        train_timeout=60 * 60,
        gpu_memory_utilization=0.3,
    )
