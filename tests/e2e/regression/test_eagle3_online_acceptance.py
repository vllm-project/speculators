"""E2E test for the online training workflow.

Exercises the full pipeline documented in
docs/user_guide/tutorials/train_eagle3_online.md:
  1. Prepare data (scripts/prepare_data.py)
  2. Launch a vLLM server for hidden-state extraction (scripts/launch_vllm.py)
  3. Train a draft model against the live server (scripts/train.py)
  4. Validate the trained checkpoint via vLLM inference (run_vllm_engine)
"""

from pathlib import Path

import pytest

from speculators.data_generation.configs import get_coco_dir
from speculators.data_generation.preprocessing import (
    _adapt_conv_for_vllm,
    _normalize_conversation,
    load_raw_dataset,
)
from tests.e2e.smoke.test_online_training import run_online_e2e
from tests.utils import requires_cadence


@requires_cadence("nightly")
@pytest.mark.parametrize(
    ("model", "dataset", "acceptance_thresholds"),
    [
        ("Qwen/Qwen3-8B", "sharegpt", [0.4, 0.1, 0.01]),
        ("Qwen/Qwen3-VL-2B-Instruct", "sharegpt4v_coco", [0.4, 0.2, 0.04]),
    ],
)
def test_online_regression(
    tmp_path: Path,
    model: str,
    dataset: str,
    acceptance_thresholds: list[float],
    prompts: list[list[dict[str, str]]],
):
    if dataset == "sharegpt4v_coco":
        coco_dir = get_coco_dir()

        if not Path(coco_dir).exists():
            pytest.skip(f"Cannot find COCO dataset at {coco_dir}")

        vllm_media_path = coco_dir

        raw_dataset, normalize_fn = load_raw_dataset(dataset)
        raw_dataset = raw_dataset.skip(len(raw_dataset) - len(prompts))
        if normalize_fn is not None:
            raw_dataset = raw_dataset.map(normalize_fn, keep_in_memory=True)

        raw_convs = raw_dataset["conversations"]
        normalized_convs = [_normalize_conversation(conv) for conv in raw_convs]
        prompts = [_adapt_conv_for_vllm(conv) for conv in normalized_convs]
    else:
        vllm_media_path = None

    run_online_e2e(
        tmp_path,
        model,
        dataset,
        max_samples=5000,
        seq_length=8192,
        vllm_kwargs={
            "gpu_memory_utilization": 0.75,
            "allowed_local_media_path": vllm_media_path,
        },
        epochs=3,
        prompts=prompts,
        acceptance_thresholds=acceptance_thresholds,
        log_freq=50,
        train_timeout=45 * 60,  # 45 mins
    )
