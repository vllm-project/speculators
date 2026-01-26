"""Tests for vLLM hidden states generator accuracy against HuggingFace baseline."""

import gc
import logging
import time

import pytest
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from speculators.data_generation import VllmHiddenStatesGenerator

logger = logging.getLogger(__name__)


@pytest.fixture(autouse=True)
def cleanup_memory():
    """Fixture to clean up GPU memory before and after each test."""
    # Cleanup before test
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    yield  # Run the test

    # Cleanup after test
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    time.sleep(1)  # Give time for cleanup


@pytest.mark.regression
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize(
    ("model_path", "tensor_parallel_size"),
    [
        ("Qwen/Qwen2-0.5B", 1),
    ],
)
def test_vllm_vs_huggingface_accuracy(model_path, tensor_parallel_size):
    """Test vLLM hidden states match HuggingFace baseline within tolerance."""

    test_prompts = [
        (
            "The future of artificial intelligence is bright and full "
            "of possibilities that will transform humanity."
        ),
        (
            "In a world where technology advances rapidly, we must "
            "carefully consider the ethical implications."
        ),
    ]

    logger.info("=" * 80)
    logger.info(f"Testing {model_path}")
    logger.info(f"Prompts: {len(test_prompts)}")
    logger.info("=" * 80)

    # HuggingFace baseline Implementation, adapted from research/eagle3/ge_data
    logger.info("[1/2] Running HuggingFace baseline...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    hf_model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    ).to("cuda")  # type: ignore[arg-type]
    num_layers = len(hf_model.model.layers)
    logger.info(f"Model has {num_layers} layers")

    inputs = tokenizer(
        test_prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=2048,
    ).to(hf_model.device)
    logger.info(f"Input shape: {inputs['input_ids'].shape}")

    with torch.no_grad():
        hf_output = hf_model(**inputs, output_hidden_states=True)

    # Extract layers using EAGLE3 pattern
    # Feature fusion: layers 2, num_layers//2, num_layers-3 (before norm)
    # Excluding the last layer (after norm) which has different behavior
    expected_layer_ids = [2, num_layers // 2, num_layers - 3]
    hf_layers = [
        hf_output.hidden_states[3],  # layer 2 (before norm)
        hf_output.hidden_states[
            num_layers // 2 + 1
        ],  # layer num_layers//2 (before norm)
        hf_output.hidden_states[num_layers - 2],  # layer num_layers-3 (before norm)
    ]

    hf_concat = torch.cat(hf_layers, dim=-1).cpu()
    logger.info(f"HuggingFace layers {expected_layer_ids}: {hf_concat.shape}")

    # Cleanup HuggingFace model - aggressive cleanup
    del hf_model, hf_output, hf_layers, inputs, tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    gc.collect()
    time.sleep(3)

    logger.info(
        f"GPU memory freed, available: {torch.cuda.mem_get_info()[0] / 1024**3:.2f} GiB"
    )

    # 2. vLLM implementation
    logger.info("[2/2] Running vLLM implementation...")
    # Only test feature fusion layers (before norm), exclude the last layer (after norm)
    test_layer_ids = [2, num_layers // 2, num_layers - 3]
    generator = VllmHiddenStatesGenerator(
        model_path=model_path,
        layer_ids=test_layer_ids,
        max_model_len=2048,
        gpu_memory_utilization=0.3,  # Conservative to avoid OOM after HF cleanup
        tensor_parallel_size=tensor_parallel_size,
    )

    try:
        # Tokenize prompts for vLLM (current implementation expects token_ids)
        # IMPORTANT: Use the SAME tokenizer that was used for HuggingFace
        # to ensure identical tokenization
        vllm_tokenizer = AutoTokenizer.from_pretrained(model_path)
        if vllm_tokenizer.pad_token is None:
            vllm_tokenizer.pad_token = vllm_tokenizer.eos_token

        # Tokenize with padding to match HuggingFace behavior
        vllm_inputs = vllm_tokenizer(
            test_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048,
        )
        token_ids_batch = vllm_inputs["input_ids"].tolist()

        vllm_results = generator.generate(token_ids=token_ids_batch)
        if not isinstance(vllm_results, list):
            vllm_results = [vllm_results]

        vllm_concat_per_seq = []
        for r in vllm_results:
            seq_concat = torch.cat(r["hidden_states"], dim=-1)
            vllm_concat_per_seq.append(seq_concat)
        vllm_concat = torch.stack(vllm_concat_per_seq).cpu()
        logger.info(f"vLLM layers {expected_layer_ids}: {vllm_concat.shape}")

        # Check layer IDs before cleanup
        actual_layer_ids = generator.layer_ids
    finally:
        del generator
        gc.collect()
        torch.cuda.empty_cache()
        time.sleep(1)

    # Verify layer IDs
    assert actual_layer_ids == expected_layer_ids, (
        f"Layer IDs mismatch! Got {actual_layer_ids}, expected {expected_layer_ids}"
    )

    # Verify shapes
    assert hf_concat.shape == vllm_concat.shape, (
        f"Shape mismatch! HF: {hf_concat.shape}, vLLM: {vllm_concat.shape}"
    )

    # Verify EAGLE3 output format
    for result in vllm_results:
        assert "input_ids" in result
        assert "hidden_states" in result
        assert "loss_mask" in result
        assert isinstance(result["hidden_states"], list)
        for layer_state in result["hidden_states"]:
            assert layer_state.shape[0] == result["input_ids"].shape[0], (
                "Sequence length mismatch"
            )

    # Numerical comparison
    max_diff = torch.abs(hf_concat - vllm_concat).max().item()
    mean_diff = torch.abs(hf_concat - vllm_concat).mean().item()
    logger.info(f"Max diff: {max_diff:.6f}, Mean diff: {mean_diff:.6f}")

    assert mean_diff < 0.02, (
        f"Mean difference {mean_diff} too large. "
        f"Expected layer_ids={expected_layer_ids}"
    )
