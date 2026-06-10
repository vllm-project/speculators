"""E2E test for the MTP finetuning pipeline.

Exercises the full pipeline:
  1. Download training data from HF
  2. Prepare tokenized data
  3. Launch vLLM server for hidden-state extraction
  4. Train online with --speculator-type mtp (auto-extracts native MTP weights)
  5. Stitch finetuned weights back into verifier
  6. Validate via vLLM speculative decoding with MTP method
"""

import logging
import sys
from pathlib import Path

import pytest
from huggingface_hub import hf_hub_download

from tests.conftest import (
    requires_cuda,
    requires_transformers_version,
    requires_vllm_version,
)
from tests.e2e.utils import (
    launch_vllm_server_context,
    run_prepare_data,
    run_stitch_mtp,
    run_training,
    run_vllm_engine,
)

logger = logging.getLogger(__name__)


@pytest.mark.e2e
@pytest.mark.slow
@requires_cuda
@requires_transformers_version("5.2.0")
@requires_vllm_version("0.22.0")
def test_mtp_finetuning_smoke(
    tmp_path: Path,
    prompts: list[list[dict[str, str]]],
):
    run_mtp_finetuning_e2e(
        tmp_path=tmp_path,
        verifier="Qwen/Qwen3.5-0.8B",
        training_data_repo="inference-optimization/Qwen3.5-0.8B-responses",
        num_speculative_tokens=3,
        target_layer_ids=[24],
        max_samples=50,
        seq_length=512,
        epochs=1,
        lr=3e-4,
        prompts=prompts,
        enforce_eager=True,
    )


def run_mtp_finetuning_e2e(
    tmp_path: Path,
    verifier: str,
    training_data_repo: str,
    num_speculative_tokens: int = 3,
    target_layer_ids: list[int] | None = None,
    max_samples: int = 50,
    seq_length: int = 512,
    epochs: int = 1,
    lr: float = 3e-4,
    prompts: list[list[dict[str, str]]] | None = None,
    acceptance_thresholds: list[float] | None = None,
    max_tokens: int = 50,
    train_timeout: int = 30 * 60,
    gpu_memory_utilization: float = 0.5,
    enforce_eager: bool = False,
):
    """Run MTP online finetuning E2E pipeline.

    If prompts is None, skip stitch and vLLM inference steps.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(name)s %(message)s",
        stream=sys.stderr,
        force=True,
    )

    data_path = tmp_path / "data"
    save_path = tmp_path / "checkpoints"
    stitched_path = tmp_path / "stitched"
    hidden_states_path = tmp_path / "hidden_states"

    port = 8321

    # Step 1: Download training data
    logger.info("Downloading training data from %s", training_data_repo)
    training_data_path = hf_hub_download(
        repo_id=training_data_repo,
        filename="gsm8k.jsonl",
        repo_type="dataset",
    )
    logger.info("Training data: %s", training_data_path)

    # Step 2: Prepare data
    logger.info("Preparing tokenized data")
    run_prepare_data(
        model=verifier,
        data=training_data_path,
        data_path=data_path,
        max_samples=max_samples,
        seq_length=seq_length,
    )
    logger.info("Data prepared: %s", data_path)

    # Step 3-4: Launch vLLM, train online
    logger.info("Starting online training")
    with launch_vllm_server_context(
        verifier,
        port,
        str(hidden_states_path),
        max_model_len=seq_length + 1,
        gpu_memory_utilization=gpu_memory_utilization,
        target_layer_ids=target_layer_ids,
        enforce_eager=enforce_eager,
    ):
        run_training(
            model=verifier,
            data_path=data_path,
            save_path=save_path,
            seq_length=seq_length,
            port=port,
            speculator_type="mtp",
            epochs=epochs,
            lr=lr,
            hidden_states_path=hidden_states_path,
            target_layer_ids=target_layer_ids,
            extra_train_args=[
                "--num-speculative-steps",
                str(num_speculative_tokens),
            ],
            timeout=train_timeout,
        )
    logger.info("Training complete")

    # Step 5-6: Stitch and validate via vLLM
    if prompts is not None:
        checkpoint_path = save_path / "checkpoint_best"
        assert checkpoint_path.exists(), f"No checkpoint at {checkpoint_path}"

        logger.info("Stitching finetuned weights into verifier")
        run_stitch_mtp(
            finetuned_checkpoint=checkpoint_path,
            verifier_path=verifier,
            output_path=stitched_path,
            timeout=10 * 60,
        )
        assert stitched_path.exists()
        logger.info("Stitch complete: %s", stitched_path)

        logger.info("Running vLLM MTP inference on stitched checkpoint")
        run_vllm_engine(
            model_path=str(stitched_path),
            tmp_path=tmp_path,
            prompts=prompts,
            speculative_config={
                "method": "mtp",
                "num_speculative_tokens": num_speculative_tokens,
            },
            enforce_eager=enforce_eager,
            acceptance_thresholds=acceptance_thresholds,
            max_tokens=max_tokens,
            timeout=15 * 60,
        )
    logger.info("MTP finetuning E2E test passed")
