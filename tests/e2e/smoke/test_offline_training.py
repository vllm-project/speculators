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
from typing import Any

import pytest

from tests.e2e.utils import (
    launch_vllm_server_context,
    run_data_generation_offline,
    run_prepare_data,
    run_training,
    run_vllm_engine,
    setup_dummy_sharegpt4v_coco,
)

TEXT_MODEL = "Qwen/Qwen3-0.6B"
MM_MODEL = "Qwen/Qwen3-VL-2B-Instruct"


@pytest.mark.e2e
@pytest.mark.slow
@pytest.mark.parametrize(
    ("model", "dataset", "speculator_type", "extra_train_args", "target_layer_ids"),
    [
        (TEXT_MODEL, "sharegpt", "eagle3", [], None),  # Use default EAGLE layers
        (
            TEXT_MODEL,
            "sharegpt",
            "dflash",
            ["--block-size", "8", "--max-anchors", "256", "--num-layers", "3"],
            [1, 13, 25],
        ),  # DFlash with 3 layers + verifier last layer
        (MM_MODEL, "sharegpt4v_coco", "eagle3", [], None),  # Multimodal
    ],
)
def test_offline_smoke(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    model: str,
    dataset: str,
    prompts: list[list[dict[str, str]]],
    speculator_type: str,
    extra_train_args: list[str],
    target_layer_ids: list[int] | None,
):
    if dataset == "sharegpt4v_coco":
        monkeypatch.setenv("COCO_DIR", str(tmp_path / "coco"))
        setup_dummy_sharegpt4v_coco(tmp_path / "coco")

        vllm_media_path = str(tmp_path / "coco")
    else:
        vllm_media_path = None

    run_offline_e2e(
        tmp_path,
        model,
        dataset=dataset,
        prompts=prompts,
        vllm_kwargs={
            "gpu_memory_utilization": 0.9,
            "allowed_local_media_path": vllm_media_path,
        },
        speculator_type=speculator_type,
        extra_train_args=extra_train_args,
        target_layer_ids=target_layer_ids,
    )


def run_offline_e2e(
    tmp_path: Path,
    model: str,
    dataset: str,
    max_samples: int = 50,
    seq_length: int = 512,
    vllm_kwargs: dict[str, Any] | None = None,
    port: int = 8321,
    draft_vocab_size: int = 8192,
    epochs: int = 1,
    lr: float = 3e-4,
    prompts: list[list[dict[str, str]]] | None = None,
    disable_compile_cache: bool = False,
    max_tokens: int = 50,
    ignore_eos: bool = True,
    acceptance_thresholds: list[float] | None = None,
    speculator_type: str = "eagle3",
    extra_train_args: list[str] | None = None,
    target_layer_ids: list[int] | None = None,
    log_freq: int = 1,
    train_timeout: int = 30 * 60,  # 30 mins
    datagen_timeout: int = 25 * 60,  # 25 mins
):
    data_path = tmp_path / "data"
    offline_hidden_states = tmp_path / "offline_hidden_states"
    save_path = tmp_path / "checkpoints"

    # Step 1: Prepare data
    run_prepare_data(model, dataset, data_path, max_samples, seq_length)

    with launch_vllm_server_context(
        model,
        port,
        str(tmp_path / "vllm_hidden_states"),
        max_model_len=seq_length + 1,
        target_layer_ids=target_layer_ids,
        **(vllm_kwargs or {}),
    ):
        # Step 2: Generate hidden states offline
        run_data_generation_offline(
            data_path,
            offline_hidden_states,
            port,
            max_samples,
            timeout=datagen_timeout,
        )

    # Step 3: Train using pre-generated hidden states (no live server needed)
    run_training(
        model,
        data_path,
        save_path,
        seq_length,
        port,
        draft_vocab_size,
        epochs,
        lr,
        online=False,
        hidden_states_path=offline_hidden_states,
        speculator_type=speculator_type,
        extra_train_args=extra_train_args,
        target_layer_ids=target_layer_ids,
        log_freq=log_freq,
        timeout=train_timeout,
    )

    # Step 4: Validate trained checkpoint with vLLM inference
    if prompts is not None:
        checkpoint_path = str(save_path / "checkpoint_best")
        run_vllm_engine(
            model_path=checkpoint_path,
            tmp_path=tmp_path,
            prompts=prompts,
            disable_compile_cache=disable_compile_cache,
            max_tokens=max_tokens,
            ignore_eos=ignore_eos,
            acceptance_thresholds=acceptance_thresholds,
        )
