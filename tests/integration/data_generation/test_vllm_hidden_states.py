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


@pytest.mark.parametrize(
    ("model_path", "tensor_parallel_size"),
    [
        ("Qwen/Qwen2-0.5B", 1),  # Small model for quick testing
        pytest.param("meta-llama/Llama-3.1-8B", 1, marks=pytest.mark.slow),
        pytest.param("meta-llama/Llama-3.1-70B", 4, marks=pytest.mark.slow),
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
        device_map="auto",
        trust_remote_code=True,
    )
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
    hf_layers = [
        hf_output.hidden_states[3],
        hf_output.hidden_states[num_layers // 2 + 1],
        hf_output.hidden_states[-3],
    ]
    expected_layer_ids = [2, num_layers // 2, num_layers - 3]

    logger.info(f"HuggingFace layers: {expected_layer_ids}")
    hf_concat = torch.cat(hf_layers, dim=-1).cpu()
    logger.info(f"HuggingFace concat: {hf_concat.shape}")

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
    generator = VllmHiddenStatesGenerator(
        model_path=model_path,
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

        logger.info(f"Tokenized {len(token_ids_batch)} prompts for vLLM")
        logger.info(f"Token IDs shape: {vllm_inputs['input_ids'].shape}")

        vllm_results = generator.generate(token_ids=token_ids_batch)
        if not isinstance(vllm_results, list):
            vllm_results = [vllm_results]

        logger.info(f"vLLM layers: {expected_layer_ids}")
        logger.info(f"vLLM returned {len(vllm_results)} results")

        vllm_hidden_states = [r["hidden_state"] for r in vllm_results]
        vllm_concat = torch.stack(vllm_hidden_states).cpu()
        logger.info(f"vLLM concat: {vllm_concat.shape}")

        # Check layer IDs before cleanup
        actual_layer_ids = generator.layer_ids
    finally:
        del generator
        gc.collect()
        torch.cuda.empty_cache()
        time.sleep(1)

    logger.info("=" * 80)
    logger.info("COMPARISON:")
    logger.info("=" * 80)

    # Check layer IDs
    assert actual_layer_ids == expected_layer_ids, (
        f"Layer IDs mismatch! Got {actual_layer_ids}, expected {expected_layer_ids}"
    )
    logger.info(f"✓ Layer IDs match: {expected_layer_ids}")

    # Check shapes
    assert hf_concat.shape == vllm_concat.shape, (
        f"Shape mismatch! HF: {hf_concat.shape}, vLLM: {vllm_concat.shape}"
    )
    logger.info(f"✓ Shapes match: {hf_concat.shape}")

    # Check EAGLE3 format
    for _i, result in enumerate(vllm_results):
        assert "input_ids" in result
        assert "hidden_state" in result
        assert "loss_mask" in result
        assert result["hidden_state"].shape[0] == result["input_ids"].shape[0], (
            "Sequence length mismatch"
        )
        assert result["hidden_state"].shape[1] == hf_concat.shape[2], (
            "Hidden dimension mismatch"
        )

    logger.info("saving format validated")

    # Numerical comparison
    max_diff = torch.abs(hf_concat - vllm_concat).max().item()
    mean_diff = torch.abs(hf_concat - vllm_concat).mean().item()

    logger.info("Numerical differences:")
    logger.info(f"  Max absolute diff:  {max_diff:.6f}")
    logger.info(f"  Mean absolute diff: {mean_diff:.6f}")

    # Debug: check sample values
    logger.info(f"HF sample [0,0,:5]: {hf_concat[0, 0, :5]}")
    logger.info(f"vLLM sample [0,0,:5]: {vllm_concat[0, 0, :5]}")

    # Mean diff should be very small (< 0.01)
    assert mean_diff < 0.01, (
        f"Mean difference {mean_diff} too large. "
        f"Expected layer_ids={expected_layer_ids}"
    )
