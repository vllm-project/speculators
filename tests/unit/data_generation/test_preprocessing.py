"""
Unit tests for the preprocessing module in the Speculators data generation.
"""

import re

import pytest
import torch
from datasets import Dataset as HFDataset
from transformers import AutoTokenizer

from speculators.data_generation.preprocessing import (
    _create_loss_mask_from_offsets,
    _detect_assistant_pattern,
    _normalize_conversation,
    _preprocess_batch,
    build_eagle3_dataset,
)

# Test model from HuggingFace with chat template
# Using Qwen2-0.5B-Instruct: small (0.5B params), fast model with proper chat template support
TEST_MODEL_REPO = "Qwen/Qwen2-0.5B-Instruct"


# Tests for _normalize_conversation
@pytest.mark.sanity
def test_normalize_conversation_sharegpt_format():
    """Test normalizing conversation from ShareGPT format (from/value keys)."""
    conv = [
        {"from": "human", "value": "What is the capital of France?"},
        {"from": "gpt", "value": "Paris is the capital of France."},
    ]
    result = _normalize_conversation(conv)

    assert len(result) == 2
    assert result[0] == {"role": "user", "content": "What is the capital of France?"}
    assert result[1] == {"role": "assistant", "content": "Paris is the capital of France."}


@pytest.mark.sanity
def test_normalize_conversation_with_system():
    """Test normalizing conversation with system messages."""
    conv = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi!"},
    ]
    result = _normalize_conversation(conv)

    assert len(result) == 3
    assert result[0]["role"] == "system"
    assert result[1]["role"] == "user"
    assert result[2]["role"] == "assistant"


@pytest.mark.sanity
def test_normalize_conversation_unknown_role():
    """Test that unknown roles are skipped with warning."""
    conv = [
        {"role": "user", "content": "Hello"},
        {"role": "unknown", "content": "Should be skipped"},
        {"role": "assistant", "content": "Hi!"},
    ]
    result = _normalize_conversation(conv)

    # Unknown role should be skipped
    assert len(result) == 2
    assert result[0]["role"] == "user"
    assert result[1]["role"] == "assistant"



# Tests for _detect_assistant_pattern
@pytest.mark.sanity
def test_detect_assistant_pattern_structure():
    """Test that the detected pattern has the correct regex structure."""
    tokenizer = AutoTokenizer.from_pretrained(TEST_MODEL_REPO, trust_remote_code=True)

    if not hasattr(tokenizer, "apply_chat_template") or tokenizer.chat_template is None:
        pytest.skip("Tokenizer does not support chat templates")

    pattern = _detect_assistant_pattern(tokenizer)

    # Pattern should be a valid regex string
    assert isinstance(pattern, str)
    assert len(pattern) > 0

    # Pattern should compile without errors
    compiled = re.compile(pattern, re.DOTALL)
    assert compiled is not None

    # Pattern should contain exactly one capture group for content
    assert pattern.count("(") == pattern.count(")")
    assert "(.*?)" in pattern, "Pattern should have a non-greedy capture group for content"


@pytest.mark.sanity
def test_detect_assistant_pattern_correctly_identifies_assistant_vs_user():
    """Test that pattern correctly distinguishes assistant from user content."""
    tokenizer = AutoTokenizer.from_pretrained(TEST_MODEL_REPO, trust_remote_code=True)

    if not hasattr(tokenizer, "apply_chat_template") or tokenizer.chat_template is None:
        pytest.skip("Tokenizer does not support chat templates")

    # Get the pattern
    pattern = _detect_assistant_pattern(tokenizer)

    # Format a conversation manually to test the pattern
    test_conv = [
        {"role": "user", "content": "USER_MSG"},
        {"role": "assistant", "content": "ASSISTANT_MSG"},
    ]
    formatted = tokenizer.apply_chat_template(
        test_conv, tokenize=False, add_generation_prompt=False
    )

    # Apply the pattern
    matches = list(re.finditer(pattern, formatted, re.DOTALL))

    # Should find exactly 1 match (the assistant message)
    assert len(matches) == 1, f"Expected 1 match, got {len(matches)}"

    # The match should capture only ASSISTANT_MSG, not USER_MSG
    captured_content = matches[0].group(1) if matches[0].lastindex else matches[0].group(0)
    assert "ASSISTANT_MSG" in captured_content, "Pattern should capture assistant content"
    assert "USER_MSG" not in captured_content, "Pattern should NOT capture user content"


@pytest.mark.sanity
def test_detect_assistant_pattern_extracts_correct_content():
    """Test that the pattern's capture group extracts only assistant message content."""
    tokenizer = AutoTokenizer.from_pretrained(TEST_MODEL_REPO, trust_remote_code=True)

    if not hasattr(tokenizer, "apply_chat_template") or tokenizer.chat_template is None:
        pytest.skip("Tokenizer does not support chat templates")

    pattern = _detect_assistant_pattern(tokenizer)

    # Test with a multi-turn conversation
    test_conv = [
        {"role": "user", "content": "First question"},
        {"role": "assistant", "content": "First answer"},
        {"role": "user", "content": "Second question"},
        {"role": "assistant", "content": "Second answer"},
    ]

    formatted = tokenizer.apply_chat_template(
        test_conv, tokenize=False, add_generation_prompt=False
    )

    matches = list(re.finditer(pattern, formatted, re.DOTALL))

    # Should match exactly 2 assistant messages
    assert len(matches) == 2, f"Expected 2 assistant matches, got {len(matches)}"

    # First match should contain "First answer" but not questions
    first_match = matches[0].group(0)
    assert "First answer" in first_match
    assert "First question" not in first_match
    assert "Second question" not in first_match

    # Second match should contain "Second answer" but not questions
    second_match = matches[1].group(0)
    assert "Second answer" in second_match
    assert "First question" not in second_match
    assert "Second question" not in second_match


# Tests for _create_loss_mask_from_offsets


@pytest.mark.sanity
def test_create_loss_mask_simple():
    """Test creating loss mask for a simple case."""
    text = "User: Hello\nAssistant: Hi there!\nUser: How are you?\nAssistant: Good!"
    pattern = r"Assistant: (.*?)(?=\n|$)"

    # Simulate token offsets (character positions)
    offsets = [
        (0, 4),  # "User"
        (4, 5),  # ":"
        (6, 11),  # "Hello"
        (11, 12),  # "\n"
        (12, 21),  # "Assistant"
        (21, 22),  # ":"
        (23, 25),  # "Hi"
        (26, 31),  # "there"
        (31, 32),  # "!"
        (32, 33),  # "\n"
        (33, 37),  # "User"
        (37, 38),  # ":"
        (39, 42),  # "How"
        (43, 46),  # "are"
        (47, 51),  # "you?"
        (51, 52),  # "\n"
        (52, 61),  # "Assistant"
        (61, 62),  # ":"
        (63, 67),  # "Good"
        (67, 68),  # "!"
    ]

    mask = _create_loss_mask_from_offsets(text, offsets, pattern)

    assert len(mask) == len(offsets)
    assert mask.dtype == torch.long

    # Tokens in assistant responses should have mask = 1
    # "Hi there!" is at positions 6-8 (indices in offsets)
    # "Good!" is at positions 18-19
    assert mask[6].item() == 1  # "Hi"
    assert mask[7].item() == 1  # "there"
    assert mask[8].item() == 1  # "!"
    assert mask[18].item() == 1  # "Good"
    assert mask[19].item() == 1  # "!"

    # User messages should have mask = 0
    assert mask[0].item() == 0  # "User"
    assert mask[2].item() == 0  # "Hello"


@pytest.mark.sanity
def test_create_loss_mask_no_matches():
    """Test creating loss mask when no assistant patterns match."""
    text = "User: Hello\nUser: How are you?"
    pattern = r"Assistant: (.*?)(?=\n|$)"

    offsets = [(0, 4), (4, 5), (6, 11)]

    mask = _create_loss_mask_from_offsets(text, offsets, pattern)

    # All zeros when no matches
    assert torch.all(mask == 0)


@pytest.mark.sanity
def test_create_loss_mask_empty_offsets():
    """Test creating loss mask with empty offsets."""
    text = "User: Hello\nAssistant: Hi!"
    pattern = r"Assistant: (.*?)(?=\n|$)"

    mask = _create_loss_mask_from_offsets(text, [], pattern)

    assert len(mask) == 0


# Tests for _preprocess_batch


@pytest.mark.sanity
def test_preprocess_batch_basic():
    """Test preprocessing a basic batch of conversations."""
    tokenizer = AutoTokenizer.from_pretrained(TEST_MODEL_REPO, trust_remote_code=True)

    if not hasattr(tokenizer, "apply_chat_template") or tokenizer.chat_template is None:
        pytest.skip("Tokenizer does not support chat templates")

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    examples = {
        "conversations": [
            [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi there!"},
            ],
            [
                {"role": "user", "content": "How are you?"},
                {"role": "assistant", "content": "I'm doing well!"},
            ],
        ]
    }

    assistant_pattern = _detect_assistant_pattern(tokenizer)
    results = _preprocess_batch(examples, tokenizer, max_length=512, assistant_pattern=assistant_pattern)

    assert "input_ids" in results
    assert "loss_mask" in results
    assert len(results["input_ids"]) == 2
    assert len(results["loss_mask"]) == 2

    # Check that input_ids and loss_mask have same length for each example
    for i in range(2):
        assert len(results["input_ids"][i]) == len(results["loss_mask"][i])
        assert isinstance(results["input_ids"][i], torch.Tensor)
        assert isinstance(results["loss_mask"][i], torch.Tensor)


@pytest.mark.sanity
def test_preprocess_batch_empty_conversations():
    """Test preprocessing batch with no conversations."""
    tokenizer = AutoTokenizer.from_pretrained(TEST_MODEL_REPO, trust_remote_code=True)

    if not hasattr(tokenizer, "apply_chat_template") or tokenizer.chat_template is None:
        pytest.skip("Tokenizer does not support chat templates")

    examples = {"conversations": []}
    assistant_pattern = _detect_assistant_pattern(tokenizer)
    results = _preprocess_batch(examples, tokenizer, max_length=512, assistant_pattern=assistant_pattern)

    assert results["input_ids"] == []
    assert results["loss_mask"] == []


@pytest.mark.sanity
def test_preprocess_batch_invalid_conversation():
    """Test preprocessing batch with invalid conversations."""
    tokenizer = AutoTokenizer.from_pretrained(TEST_MODEL_REPO, trust_remote_code=True)

    if not hasattr(tokenizer, "apply_chat_template") or tokenizer.chat_template is None:
        pytest.skip("Tokenizer does not support chat templates")

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    examples = {
        "conversations": [
            None,  # Invalid
            [],  # Empty
            [{"role": "user", "content": "Valid"}],  # Valid
        ]
    }

    assistant_pattern = _detect_assistant_pattern(tokenizer)
    results = _preprocess_batch(examples, tokenizer, max_length=512, assistant_pattern=assistant_pattern)

    # Should only process the valid conversation
    assert len(results["input_ids"]) <= 1
    assert len(results["loss_mask"]) <= 1


@pytest.mark.sanity
def test_preprocess_batch_truncation():
    """Test that long sequences are truncated to max_length."""
    tokenizer = AutoTokenizer.from_pretrained(TEST_MODEL_REPO, trust_remote_code=True)

    if not hasattr(tokenizer, "apply_chat_template") or tokenizer.chat_template is None:
        pytest.skip("Tokenizer does not support chat templates")

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Create a very long message
    long_content = "word " * 1000

    examples = {
        "conversations": [
            [
                {"role": "user", "content": long_content},
                {"role": "assistant", "content": "Short reply"},
            ]
        ]
    }

    max_length = 100
    assistant_pattern = _detect_assistant_pattern(tokenizer)
    results = _preprocess_batch(examples, tokenizer, max_length=max_length, assistant_pattern=assistant_pattern)

    if len(results["input_ids"]) > 0:
        # Should be truncated to max_length
        assert len(results["input_ids"][0]) <= max_length
        assert len(results["loss_mask"][0]) <= max_length


# Tests for build_eagle3_dataset


@pytest.mark.sanity
def test_build_eagle3_dataset_basic():
    """Test building EAGLE3 dataset from a simple HuggingFace dataset."""
    tokenizer = AutoTokenizer.from_pretrained(TEST_MODEL_REPO, trust_remote_code=True)

    if not hasattr(tokenizer, "apply_chat_template") or tokenizer.chat_template is None:
        pytest.skip("Tokenizer does not support chat templates")

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Create a simple dataset
    data = {
        "conversations": [
            [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi!"},
            ],
            [
                {"role": "user", "content": "Goodbye"},
                {"role": "assistant", "content": "Bye!"},
            ],
        ]
    }

    dataset = HFDataset.from_dict(data)
    result = build_eagle3_dataset(dataset, tokenizer, max_length=512, num_proc=1)

    assert isinstance(result, HFDataset)
    assert len(result) <= len(dataset)

    # Check that the dataset has the expected columns
    if len(result) > 0:
        assert "input_ids" in result.column_names
        assert "loss_mask" in result.column_names


@pytest.mark.sanity
def test_build_eagle3_dataset_preserves_format():
    """Test that build_eagle3_dataset sets the correct format."""
    tokenizer = AutoTokenizer.from_pretrained(TEST_MODEL_REPO, trust_remote_code=True)

    if not hasattr(tokenizer, "apply_chat_template") or tokenizer.chat_template is None:
        pytest.skip("Tokenizer does not support chat templates")

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    data = {
        "conversations": [
            [
                {"role": "user", "content": "Test"},
                {"role": "assistant", "content": "Response"},
            ]
        ]
    }

    dataset = HFDataset.from_dict(data)
    result = build_eagle3_dataset(dataset, tokenizer, max_length=512, num_proc=1)

    # Dataset should be in torch format
    assert result.format["type"] == "torch"


@pytest.mark.sanity
def test_build_eagle3_dataset_removes_original_columns():
    """Test that original columns are removed after processing."""
    tokenizer = AutoTokenizer.from_pretrained(TEST_MODEL_REPO, trust_remote_code=True)

    if not hasattr(tokenizer, "apply_chat_template") or tokenizer.chat_template is None:
        pytest.skip("Tokenizer does not support chat templates")

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    data = {
        "conversations": [
            [
                {"role": "user", "content": "Test"},
                {"role": "assistant", "content": "Response"},
            ]
        ],
        "extra_column": ["extra_data"],
    }

    dataset = HFDataset.from_dict(data)
    result = build_eagle3_dataset(dataset, tokenizer, max_length=512, num_proc=1)

    # Original columns should be removed
    if len(result) > 0:
        assert "conversations" not in result.column_names
        assert "extra_column" not in result.column_names
