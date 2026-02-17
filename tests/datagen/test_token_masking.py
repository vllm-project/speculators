"""
Tests for token-ID-based assistant masking.

Tests detect_assistant_token_markers and create_loss_mask_from_token_ids,
which compute assistant loss masks purely from token IDs without needing
access to raw text or regex patterns at inference time.
"""

import pytest
import torch
from transformers import AutoTokenizer

from speculators.data_generation.preprocessing import (
    create_loss_mask_from_token_ids,
    detect_assistant_token_markers,
)

# ──────────────────────────────────────────────────────────────────────
# Unit tests for create_loss_mask_from_token_ids (synthetic sequences)
# ──────────────────────────────────────────────────────────────────────

START = [10, 11]
END = [20, 21]


@pytest.mark.sanity
def test_mask_single_turn():
    """Single assistant turn between markers."""
    token_ids = [1, 2, 3, 10, 11, 100, 101, 102, 20, 21, 4, 5]
    mask = create_loss_mask_from_token_ids(token_ids, START, END)
    assert mask.tolist() == [0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0]


@pytest.mark.sanity
def test_mask_multi_turn():
    """Multiple assistant turns."""
    token_ids = [1, 10, 11, 50, 51, 20, 21, 2, 3, 10, 11, 60, 61, 20, 21]
    mask = create_loss_mask_from_token_ids(token_ids, START, END)
    assert mask.tolist() == [0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0]


@pytest.mark.sanity
def test_mask_truncated_no_end_marker():
    """Sequence truncated mid-assistant-turn (no end marker) marks to end."""
    token_ids = [1, 10, 11, 50, 51, 52]
    mask = create_loss_mask_from_token_ids(token_ids, START, END)
    assert mask.tolist() == [0, 0, 0, 1, 1, 1]


@pytest.mark.sanity
def test_mask_no_assistant():
    """Sequence with no assistant markers produces all-zeros mask."""
    token_ids = [1, 2, 3, 4, 5]
    mask = create_loss_mask_from_token_ids(token_ids, START, END)
    assert mask.tolist() == [0, 0, 0, 0, 0]


@pytest.mark.sanity
def test_mask_empty_sequence():
    """Empty sequence produces empty mask."""
    mask = create_loss_mask_from_token_ids([], START, END)
    assert mask.tolist() == []


@pytest.mark.sanity
def test_mask_empty_assistant_content():
    """Start marker immediately followed by end marker produces no 1s."""
    token_ids = [1, 10, 11, 20, 21, 2]
    mask = create_loss_mask_from_token_ids(token_ids, START, END)
    assert mask.tolist() == [0, 0, 0, 0, 0, 0]


@pytest.mark.sanity
def test_mask_tensor_input():
    """Accepts torch.Tensor input."""
    token_ids = torch.tensor([1, 10, 11, 99, 20, 21])
    mask = create_loss_mask_from_token_ids(token_ids, START, END)
    assert mask.tolist() == [0, 0, 0, 1, 0, 0]
    assert mask.dtype == torch.long


@pytest.mark.sanity
def test_mask_markers_not_included():
    """Start and end marker tokens themselves are not marked as 1."""
    # Only content between markers should be 1
    token_ids = [10, 11, 42, 20, 21]
    mask = create_loss_mask_from_token_ids(token_ids, START, END)
    assert mask.tolist() == [0, 0, 1, 0, 0]


# ──────────────────────────────────────────────────────────────────────
# Integration tests with real tokenizers
# ──────────────────────────────────────────────────────────────────────

# Models covering major chat template families
MODELS = [
    "Qwen/Qwen2-0.5B-Instruct",  # ChatML style
    "unsloth/llama-3-8b-Instruct",  # Llama-3 style
    "mistralai/Mistral-7B-Instruct-v0.2",  # Mistral style
    "unsloth/gemma-2b-it",  # Gemma style
    "microsoft/Phi-3-mini-4k-instruct",  # Phi-3 style
    "openai/gpt-oss-20b",  # GPT-OSS style
]


@pytest.fixture(scope="module", params=MODELS)
def tokenizer(request):
    model_id = request.param
    try:
        return AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    except (
        TypeError,
        ValueError,
        KeyError,
        AttributeError,
        RuntimeError,
        OSError,
    ) as e:
        pytest.skip(f"Failed to load tokenizer for {model_id}: {e}")


def test_detect_markers(tokenizer):
    """Marker detection produces non-empty start and end sequences."""
    start_ids, end_ids = detect_assistant_token_markers(tokenizer)
    assert len(start_ids) > 0, "Start marker should be non-empty"
    assert len(end_ids) > 0, "End marker should be non-empty"
    assert all(isinstance(t, int) for t in start_ids)
    assert all(isinstance(t, int) for t in end_ids)


def test_token_mask_matches_content(tokenizer):
    """Token-ID mask correctly identifies assistant vs user content."""
    start_ids, end_ids = detect_assistant_token_markers(tokenizer)

    conv = [
        {"role": "user", "content": "Hello, how are you?"},
        {"role": "assistant", "content": "I am a helpful assistant."},
        {"role": "user", "content": "What is the capital of France?"},
        {"role": "assistant", "content": "The capital of France is Paris."},
    ]

    token_ids = tokenizer.apply_chat_template(
        conv, tokenize=True, add_generation_prompt=False
    )
    mask = create_loss_mask_from_token_ids(token_ids, start_ids, end_ids)

    assert len(mask) == len(token_ids)
    assert mask.sum() > 0, "Mask should not be all zeros"

    # Decode masked (assistant) tokens
    masked_ids = torch.tensor(token_ids)[mask.bool()]
    decoded_assistant = tokenizer.decode(masked_ids)

    # Assistant content should be present
    assert "I am a helpful assistant" in decoded_assistant
    assert "The capital of France is Paris." in decoded_assistant

    # User content should NOT be present
    assert "Hello" not in decoded_assistant
    assert "you?" not in decoded_assistant
    assert "France?" not in decoded_assistant
    assert "What is" not in decoded_assistant


def test_token_mask_with_system_message(tokenizer):
    """System messages are not marked as assistant content."""
    start_ids, end_ids = detect_assistant_token_markers(tokenizer)

    conv = [
        {"role": "system", "content": "You are a helpful bot."},
        {"role": "user", "content": "Hi"},
        {"role": "assistant", "content": "Hello!"},
    ]

    try:
        token_ids = tokenizer.apply_chat_template(
            conv, tokenize=True, add_generation_prompt=False
        )
    except Exception:
        pytest.skip("Tokenizer does not support system messages")

    mask = create_loss_mask_from_token_ids(token_ids, start_ids, end_ids)

    masked_ids = torch.tensor(token_ids)[mask.bool()]
    decoded = tokenizer.decode(masked_ids)

    # System message should NOT be in masked content
    assert "helpful bot" not in decoded
    # Assistant content should be present
    assert "Hello!" in decoded


def test_token_mask_single_turn(tokenizer):
    """Single user-assistant turn works correctly."""
    start_ids, end_ids = detect_assistant_token_markers(tokenizer)

    conv = [
        {"role": "user", "content": "Say hello"},
        {"role": "assistant", "content": "Hello world!"},
    ]

    token_ids = tokenizer.apply_chat_template(
        conv, tokenize=True, add_generation_prompt=False
    )
    mask = create_loss_mask_from_token_ids(token_ids, start_ids, end_ids)

    assert mask.sum() > 0

    masked_ids = torch.tensor(token_ids)[mask.bool()]
    decoded = tokenizer.decode(masked_ids)

    assert "Hello world" in decoded
    assert "Say hello" not in decoded
