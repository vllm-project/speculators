"""E2E test for the online training workflow.

Exercises the full pipeline documented in examples/ONLINE_TRAINING.md:
  1. Prepare data (scripts/prepare_data.py)
  2. Launch a vLLM server for hidden-state extraction (scripts/launch_vllm.py)
  3. Train a draft model against the live server (scripts/train.py)
  4. Validate the trained checkpoint via vLLM inference (run_vllm_engine)
"""

from pathlib import Path

import pytest

from tests.e2e.utils import (
    launch_vllm_server_context,
    run_prepare_data,
    run_training,
    run_vllm_engine,
)

MODEL = "Qwen/Qwen3-0.6B"


@pytest.mark.e2e
@pytest.mark.slow
def test_online_smoke(tmp_path: Path, prompts: list[list[dict[str, str]]]):
    run_online_e2e(tmp_path, MODEL, prompts=prompts)


def run_online_e2e(
    tmp_path: Path,
    model: str,
    max_samples: int = 50,
    seq_length: int = 512,
    vllm_gpu_util: float = 0.5,
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
    run_prepare_data(model, data_path, max_samples, seq_length)

    hidden_states_path = str(tmp_path / "hidden_states")
    with launch_vllm_server_context(
        model,
        port,
        hidden_states_path,
        max_model_len=seq_length + 1,
        gpu_memory_utilization=vllm_gpu_util,
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
