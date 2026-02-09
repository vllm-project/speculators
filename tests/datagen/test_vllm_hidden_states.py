"""Tests for vLLM hidden states generator accuracy against HuggingFace baseline."""

import gc
import logging
import os
import time

import pytest
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from speculators.data_generation import VllmHiddenStatesGenerator

logger = logging.getLogger(__name__)

# Set vLLM multiprocessing method to spawn for CUDA compatibility
# Must be set before vLLM imports to avoid CUDA re-initialization errors
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")


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


@pytest.mark.regression
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize(
    ("model_path", "tensor_parallel_size"),
    [
        ("Qwen/Qwen2-0.5B", 1),
        ("meta-llama/Llama-3.1-8B", 1),
    ],
)
def test_batch_vs_individual_consistency(  # noqa: C901
    model_path, tensor_parallel_size
):
    """Test that batch processing matches individual processing.

    Regression test for GitHub issue #279: VllmHiddenStatesGenerator returns
    silently wrong hidden states with batch_size > 1 or repeated calls.

    This test verifies:
    1. No KV cache state leakage between calls (Bug 1)
    2. Correct token ordering in chunked prefill (Bug 2)
    """
    # 8 distinct prompts of varying length to trigger chunked prefill
    test_prompts = [
        "What is 2+2?",
        "Explain the theory of relativity in simple terms.",
        "Write a haiku about the ocean.",
        "What are the main differences between Python and JavaScript?",
        "Hello!",
        "Translate 'good morning' to French, Spanish, and German.",
        "What is the capital of Brazil?",
        "Describe the process of photosynthesis step by step.",
    ]

    logger.info(f"Testing batch vs individual consistency: {model_path}")

    # Initialize generator with aggressive chunking to properly test the fix
    # This forces multi-iteration chunked prefill which exposes token ordering bugs
    generator = VllmHiddenStatesGenerator(
        model_path=model_path,
        layer_ids=[10],  # Single layer for faster testing
        max_model_len=2048,
        gpu_memory_utilization=0.3,
        tensor_parallel_size=tensor_parallel_size,
        max_num_batched_tokens=100,  # Force chunking: ~212 tokens / 100 = 3 iterations
    )

    try:
        # Tokenize prompts
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Use chat template if available, otherwise just tokenize
        all_ids = []
        for text in test_prompts:
            try:
                # Try chat template first (for instruct models)
                msgs = [{"role": "user", "content": text}]
                ids = tokenizer.apply_chat_template(
                    msgs,
                    tokenize=True,
                    add_generation_prompt=True,
                    return_tensors="pt",
                    padding=False,
                )
                if isinstance(ids, dict):
                    ids = ids["input_ids"]
                all_ids.append(ids.squeeze(0).tolist())
            except (ValueError, AttributeError):
                # Fallback for base models without chat template
                ids = tokenizer(text, return_tensors="pt")["input_ids"]
                all_ids.append(ids.squeeze(0).tolist())

        seq_lens = [len(ids) for ids in all_ids]
        logger.info(f"Sequence lengths: {seq_lens}")
        logger.info(f"Total tokens: {sum(seq_lens)}")

        # --- Ground truth: process each sequence individually ---
        logger.info("Processing sequences individually...")
        individual_results = []
        for i, ids in enumerate(all_ids):
            results = generator.generate([ids])
            individual_results.append(results[0])
            hs = results[0]["hidden_states"][0]
            logger.info(
                f"  Seq {i}: input_len={seq_lens[i]:3d}, hs_shape={list(hs.shape)}"
            )

        # --- Batch processing ---
        logger.info("Processing all sequences as batch...")
        batch_results = generator.generate(all_ids)

        # --- Verify results match ---
        misaligned = 0
        empty = 0
        for i in range(len(all_ids)):
            individual_hs = individual_results[i]["hidden_states"][0]
            batch_hs = batch_results[i]["hidden_states"][0]

            expected_shape = list(individual_hs.shape)
            got_shape = list(batch_hs.shape)

            # Check for empty results
            if batch_hs.numel() == 0:
                empty += 1
                logger.error(f"  Seq {i}: EMPTY (bug reproduced)")
                continue

            # Check for shape mismatch
            if got_shape != expected_shape:
                misaligned += 1
                logger.error(
                    f"  Seq {i}: WRONG SHAPE "
                    f"(got {got_shape}, expected {expected_shape})"
                )
                continue

            # Check for value mismatch
            if individual_hs.shape[0] > 0 and batch_hs.shape[0] > 0:
                mean_diff = torch.abs(individual_hs - batch_hs).mean().item()

                if mean_diff > 0.01:  # Tolerance for numerical differences
                    misaligned += 1
                    logger.error(f"  Seq {i}: WRONG VALUES (mean_diff={mean_diff:.6f})")
                    continue

        # Assert no errors
        total_errors = empty + misaligned
        assert total_errors == 0, (
            f"Batch processing returned wrong hidden states: "
            f"{empty} empty, {misaligned} misaligned out of {len(all_ids)} sequences. "
            f"This indicates bug #279 regression."
        )

        logger.info(
            f"SUCCESS: All {len(all_ids)} sequences matched between "
            f"individual and batch processing"
        )

    finally:
        del generator
        gc.collect()
        torch.cuda.empty_cache()
        time.sleep(1)
