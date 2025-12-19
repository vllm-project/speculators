"""
Tests for dynamic regex assistant pattern detection across different model families.
"""

import sys
from pathlib import Path
from re import Pattern

import pytest
from transformers import AutoTokenizer

sys.path.append(str(Path(__file__).resolve().parents[2] / "src"))

from loguru import logger as log

from speculators.data_generation.preprocessing import (
    _detect_assistant_pattern,
    _preprocess_batch,
)

# Test models covering major template families
MODELS = [
    # Qwen/ChatML style
    "Qwen/Qwen2-0.5B-Instruct",
    # Llama-3 style (<|begin_of_text|>...) - used unsloth to avoid gated repo
    "unsloth/llama-3-8b-Instruct",
    # Mistral family ([INST] ... [/INST])
    "mistralai/Mistral-7B-Instruct-v0.2",
    # Gemma style
    "google/gemma-2b-it",
    # Phi-3 style
    "microsoft/Phi-3-mini-4k-instruct",
    # GPT-OSS
    "openai/gpt-oss-20b",
]


@pytest.fixture(scope="module", params=MODELS)
def tokenizer(request):
    model_id = request.param
    try:
        # Using trust_remote_code=True for variety of templates
        return AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    except (TypeError, ValueError, KeyError, AttributeError, RuntimeError) as e:
        pytest.skip(f"Failed to load tokenizer for {model_id}: {e}")


def test_regex_detection_across_models(tokenizer):
    """
    Verify that _detect_assistant_pattern and _preprocess_batch (regex path)
    work correctly for a variety of model families.
    """
    model_name = tokenizer.name_or_path
    log.info(f"Testing family: {model_name}")

    # 1. Detect pattern
    try:
        pattern = _detect_assistant_pattern(tokenizer)
    except (ValueError, RuntimeError) as e:
        pytest.fail(f"Failed to detect assistant pattern for {model_name}: {e}")

    log.info(f"Detected pattern: {pattern}")
    assert isinstance(pattern, (str, Pattern)), "Pattern must be str or regex object"

    # 2. Preprocess a simple multi-turn conversation using REGEX path
    examples = {
        "conversations": [
            [
                {"role": "user", "content": "Hello, how are you?"},
                {"role": "assistant", "content": "I am a helpful assistant."},
                {"role": "user", "content": "What is the capital of France?"},
                {"role": "assistant", "content": "The capital of France is Paris."},
            ]
        ]
    }

    # Regex path by passing the explicit pattern
    results = _preprocess_batch(
        examples, tokenizer, max_length=512, assistant_pattern=pattern
    )

    assert len(results["input_ids"]) == 1
    assert len(results["loss_mask"]) == 1

    input_ids = results["input_ids"][0]
    loss_mask = results["loss_mask"][0]

    # Verify basic properties
    assert len(input_ids) == len(loss_mask)
    assert loss_mask.sum() > 0, "Loss mask should not be all zeros"

    # 3. Qualitative check: Assistant content should be masked as 1
    trainable_tokens = input_ids[loss_mask == 1]
    decoded_assistant = tokenizer.decode(trainable_tokens)

    log.info(f"Decoded trainable regions: {decoded_assistant}")

    # It should at least contain parts of our assistant messages
    assert "helpful assistant" in decoded_assistant
    assert "Paris" in decoded_assistant

    # It should NOT contain user message content
    assert "Hello" not in decoded_assistant
    assert "France?" not in decoded_assistant


if __name__ == "__main__":
    pass
