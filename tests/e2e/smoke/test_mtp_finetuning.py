"""E2E test for the MTP finetuning pipeline.

Exercises the full pipeline:
  1. Download training data from HF
  2. Convert Qwen3.5-0.8B MTP head to speculators format
  3. Prepare tokenized data
  4. Launch vLLM server for hidden-state extraction
  5. Train online with --speculator-type mtp
  6. Stitch finetuned weights back into verifier
  7. Evaluate via vLLM speculative decoding with acceptance thresholds
"""

import json
import logging
import sys
from pathlib import Path

import pytest
from huggingface_hub import hf_hub_download

from speculators.convert.mtp import MTPConverter
from tests.conftest import requires_cuda, requires_transformers_version
from tests.e2e.utils import (
    launch_vllm_server_context,
    run_prepare_data,
    run_stitch_mtp,
    run_training,
    run_vllm_engine,
)

logger = logging.getLogger(__name__)

VERIFIER = "Qwen/Qwen3.5-0.8B"
TRAINING_DATA_REPO = "inference-optimization/Qwen3.5-0.8B-responses"
TRAINING_DATA_FILE = "gsm8k-test-50.json"
EVAL_DATA_REPO = "RedHatAI/speculator_benchmarks"
EVAL_DATA_FILE = "math_reasoning.jsonl"
NUM_SPECULATIVE_TOKENS = 3
TARGET_LAYER_IDS = [24]

# TODO: Placeholders replace with measured baselines
ACCEPTANCE_THRESHOLDS = [0.3, 0.1, 0.01]


def _load_eval_prompts(path: Path) -> list[list[dict[str, str]]]:
    prompts = []
    with path.open() as f:
        for line in f:
            sample = json.loads(line)
            text = sample.get("text") or sample.get("prompt", "")
            prompts.append([{"role": "user", "content": text}])
    return prompts


@pytest.mark.e2e
@pytest.mark.slow
@requires_cuda
@requires_transformers_version("5.2.0")
def test_mtp_finetuning(tmp_path: Path):
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(name)s %(message)s",
        stream=sys.stderr,
        force=True,
    )

    converted_path = tmp_path / "converted"
    data_path = tmp_path / "data"
    save_path = tmp_path / "checkpoints"
    stitched_path = tmp_path / "stitched"
    hidden_states_path = tmp_path / "hidden_states"

    port = 8321
    seq_length = 512

    # -- Step 1: Download training data --
    logger.info("Downloading training data from %s", TRAINING_DATA_REPO)
    training_data_path = hf_hub_download(
        repo_id=TRAINING_DATA_REPO,
        filename=TRAINING_DATA_FILE,
        repo_type="dataset",
    )
    logger.info("Training data: %s", training_data_path)

    # -- Step 2: Convert MTP head --
    logger.info("Converting %s MTP head to speculators format", VERIFIER)
    converter = MTPConverter()
    converter.convert(
        input_path=VERIFIER,
        output_path=str(converted_path),
        base_model=VERIFIER,
        num_speculative_steps=NUM_SPECULATIVE_TOKENS,
        validate=False,
    )
    assert converted_path.exists()
    assert any(converted_path.glob("*.safetensors"))
    logger.info("Conversion complete: %s", converted_path)

    # -- Step 3: Prepare data --
    logger.info("Preparing tokenized data")
    run_prepare_data(
        model=VERIFIER,
        data=training_data_path,
        data_path=data_path,
        max_samples=50,
        seq_length=seq_length,
    )
    logger.info("Data prepared: %s", data_path)

    # -- Steps 4-6: Launch vLLM, train online, stop vLLM --
    logger.info("Starting online training")
    with launch_vllm_server_context(
        VERIFIER,
        port,
        str(hidden_states_path),
        max_model_len=seq_length + 1,
        target_layer_ids=TARGET_LAYER_IDS,
        enforce_eager=True,
    ):
        run_training(
            model=VERIFIER,
            data_path=data_path,
            save_path=save_path,
            seq_length=seq_length,
            port=port,
            speculator_type="mtp",
            epochs=1,
            lr=3e-4,
            hidden_states_path=hidden_states_path,
            target_layer_ids=TARGET_LAYER_IDS,
            extra_train_args=[
                "--from-pretrained",
                str(converted_path),
            ],
            timeout=30 * 60,
        )
    logger.info("Training complete")

    # -- Step 7: Stitch finetuned weights --
    checkpoint_path = save_path / "checkpoint_best"
    assert checkpoint_path.exists(), f"No checkpoint at {checkpoint_path}"

    logger.info("Stitching finetuned weights into verifier")
    run_stitch_mtp(
        finetuned_checkpoint=checkpoint_path,
        verifier_path=VERIFIER,
        output_path=stitched_path,
        timeout=10 * 60,
    )
    assert stitched_path.exists()
    logger.info("Stitch complete: %s", stitched_path)

    # -- Steps 8-10: Load eval data, run vLLM engine, assert thresholds --
    logger.info("Downloading evaluation data")
    eval_data_path = hf_hub_download(
        repo_id=EVAL_DATA_REPO,
        filename=EVAL_DATA_FILE,
        repo_type="dataset",
    )
    prompts = _load_eval_prompts(Path(eval_data_path))
    logger.info("Loaded %d evaluation prompts", len(prompts))

    logger.info("Running vLLM MTP inference on stitched checkpoint")
    run_vllm_engine(
        model_path=str(stitched_path),
        tmp_path=tmp_path,
        prompts=prompts,
        speculative_config={
            "method": "mtp",
            "num_speculative_tokens": NUM_SPECULATIVE_TOKENS,
        },
        enforce_eager=True,
        acceptance_thresholds=ACCEPTANCE_THRESHOLDS,
        timeout=15 * 60,
    )
    logger.info("MTP finetuning E2E test passed")
