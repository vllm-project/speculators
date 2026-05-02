"""E2E test for the online training workflow.

Exercises the full pipeline documented in
docs/user_guide/tutorials/train_eagle3_online.md:
  1. Prepare data (scripts/prepare_data.py)
  2. Launch a vLLM server for hidden-state extraction (scripts/launch_vllm.py)
  3. Train a draft model against the live server (scripts/train.py)
  4. Validate the trained checkpoint via vLLM inference (run_vllm_engine)
"""

from pathlib import Path
from typing import Any

import pytest

from tests.e2e.utils import (
    launch_vllm_server_context,
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
    ("model", "dataset"),
    [
        (TEXT_MODEL, "sharegpt"),
        (MM_MODEL, "sharegpt4v_coco"),
    ],
)
def test_online_smoke(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    model: str,
    dataset: str,
    prompts: list[list[dict[str, str]]],
):
    if dataset == "sharegpt4v_coco":
        monkeypatch.setenv("COCO_DIR", str(tmp_path / "coco"))
        setup_dummy_sharegpt4v_coco(tmp_path / "coco")

        vllm_enforce_eager = True
        vllm_media_path = str(tmp_path / "coco")
    else:
        vllm_enforce_eager = False
        vllm_media_path = None

    run_online_e2e(
        tmp_path,
        model,
        dataset=dataset,
        prompts=prompts,
        vllm_kwargs={
            "enforce_eager": vllm_enforce_eager,
            "allowed_local_media_path": vllm_media_path,
        },
    )


def run_online_e2e(
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
    log_freq: int = 1,
    train_timeout: int = 30 * 60,  # 30 mins
):
    """
    Run online training e2e testing pipeline.

    If prompts is None, skip testing the final model in vllm
    """
    data_path = tmp_path / "data"
    save_path = tmp_path / "checkpoints"

    # Step 1: Prepare data
    run_prepare_data(model, dataset, data_path, max_samples, seq_length)

    hidden_states_path = str(tmp_path / "hidden_states")
    with launch_vllm_server_context(
        model,
        port,
        hidden_states_path,
        max_model_len=seq_length + 1,
        **(vllm_kwargs or {}),
    ):
        # Step 2: Train against live vLLM server
        run_training(
            model,
            data_path,
            save_path,
            seq_length,
            port,
            draft_vocab_size,
            epochs,
            lr,
            log_freq=log_freq,
            timeout=train_timeout,
        )

    # Step 3: Validate trained checkpoint with vLLM inference
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
