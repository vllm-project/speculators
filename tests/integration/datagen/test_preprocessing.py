"""
Unit tests for the preprocessing module in the Speculators data generation.
"""

import re

import pytest
import torch
from datasets import Dataset as HFDataset
from PIL import Image

from speculators.data_generation.preprocessing import (
    _create_loss_mask_from_offsets,
    _detect_assistant_pattern,
    _hf_to_vllm_conv,
    _load_processor,
    _normalize_conversation,
    _preprocess_batch,
    _supports_assistant_mask,
    build_eagle3_dataset,
)

# Test model from HuggingFace with chat template
# Using Qwen2-0.5B-Instruct: small (0.5B params), fast model with proper
# chat template support
TEXT_MODEL_REPO = "Qwen/Qwen2-0.5B-Instruct"
# For testing multi-modal support
MM_MODEL_REPO = "Qwen/Qwen3-VL-2B-Instruct"


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
    assert result[0] == {
        "role": "user",
        "content": "What is the capital of France?",
    }
    assert result[1] == {
        "role": "assistant",
        "content": "Paris is the capital of France.",
    }


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


# Tests for _hf_to_vllm_conv
@pytest.mark.sanity
def test_hf_to_vllm_all_content_formats():
    """
    Test converting from HF-format to vLLM-format messages with
    each supported content format.
    """
    conv: list[dict] = [
        {
            "role": "system",
            "content": "You are a helpful assistant.",  # Content as string
        },
        {
            "role": "assistant",
            "content": [  # Content as list
                "Hello,",  # Text as string
                {"type": "text", "text": "I am"},  # Text as dictionary
                {"type": "image", "path": "/path/to/img"},  # Image file path
                {"type": "image", "url": "http://path/to/img"},  # Image URL
            ],
        },
    ]
    result = _hf_to_vllm_conv(conv)

    assert len(result) == 2

    assert result[0]["role"] == "system"
    assert result[0]["content"] == "You are a helpful assistant."

    assert result[1]["role"] == "assistant"
    assert result[1]["content"][0] == {"type": "text", "text": "Hello,"}
    assert result[1]["content"][1] == {"type": "text", "text": "I am"}
    assert result[1]["content"][2] == {
        "type": "image_url",
        "image_url": {"url": "file:///path/to/img"},
    }
    assert result[1]["content"][3] == {
        "type": "image_url",
        "image_url": {"url": "http://path/to/img"},
    }


@pytest.mark.sanity
def test_hf_to_vllm_invalid_content_formats():
    """
    Test converting from HF-format to vLLM-format messages with
    unsupported content formats.
    """
    # Image object is not supported to discourage copying images
    # when saving the preprocessed dataset
    with pytest.raises(
        NotImplementedError, match=r"No handler .* for fields: \{'part\.image'\}"
    ):
        _hf_to_vllm_conv(
            [
                {
                    "role": "assistant",
                    "content": [
                        {"type": "image", "image": Image.new("RGB", (256, 256))},
                    ],
                },
            ]
        )

    # Image base64 is not supported to discourage copying images
    # when saving the preprocessed dataset
    with pytest.raises(
        NotImplementedError, match=r"No handler .* for fields: \{'part\.base64'\}"
    ):
        _hf_to_vllm_conv(
            [
                {
                    "role": "assistant",
                    "content": [
                        {"type": "image", "base64": "abcdef"},
                    ],
                },
            ]
        )


# Tests for _detect_assistant_pattern
@pytest.mark.sanity
def test_detect_assistant_pattern_structure():
    """Test that the detected pattern has the correct regex structure."""
    processor = _load_processor(TEXT_MODEL_REPO, trust_remote_code=True)

    if not hasattr(processor, "apply_chat_template") or processor.chat_template is None:
        pytest.skip("Processor does not support chat templates")

    pattern = _detect_assistant_pattern(processor)

    # Pattern should be a valid regex string
    assert isinstance(pattern, str)
    assert len(pattern) > 0

    # Pattern should compile without errors
    compiled = re.compile(pattern, re.DOTALL)
    assert compiled is not None

    # Pattern should contain balanced parentheses
    assert pattern.count("(") == pattern.count(")")
    # Pattern should have at least one capture group (may use negative lookahead)
    assert "(" in pattern, "Pattern should have a capture group for content"


@pytest.mark.sanity
def test_detect_assistant_pattern_correctly_identifies_assistant_vs_user():
    """Test that pattern correctly distinguishes assistant from user content."""
    processor = _load_processor(TEXT_MODEL_REPO, trust_remote_code=True)

    if not hasattr(processor, "apply_chat_template") or processor.chat_template is None:
        pytest.skip("Processor does not support chat templates")

    # Get the pattern
    pattern = _detect_assistant_pattern(processor)

    # Format a conversation manually to test the pattern
    test_conv = [
        {"role": "user", "content": "USER_MSG"},
        {"role": "assistant", "content": "ASSISTANT_MSG"},
    ]
    formatted: str = processor.apply_chat_template(  # type: ignore[assignment]
        test_conv, tokenize=False, add_generation_prompt=False
    )

    # Apply the pattern
    matches = list(re.finditer(pattern, formatted, re.DOTALL))

    # Should find exactly 1 match (the assistant message)
    assert len(matches) == 1, f"Expected 1 match, got {len(matches)}"

    # The match should capture only ASSISTANT_MSG, not USER_MSG
    captured_content = (
        matches[0].group(1) if matches[0].lastindex else matches[0].group(0)
    )
    assert "ASSISTANT_MSG" in captured_content, (
        "Pattern should capture assistant content"
    )
    assert "USER_MSG" not in captured_content, "Pattern should NOT capture user content"


@pytest.mark.sanity
def test_detect_assistant_pattern_extracts_correct_content():
    """Test that the pattern's capture group extracts only assistant message content."""
    processor = _load_processor(TEXT_MODEL_REPO, trust_remote_code=True)

    if not hasattr(processor, "apply_chat_template") or processor.chat_template is None:
        pytest.skip("Processor does not support chat templates")

    pattern = _detect_assistant_pattern(processor)

    # Test with a multi-turn conversation
    test_conv = [
        {"role": "user", "content": "First question"},
        {"role": "assistant", "content": "First answer"},
        {"role": "user", "content": "Second question"},
        {"role": "assistant", "content": "Second answer"},
    ]

    formatted: str = processor.apply_chat_template(  # type: ignore [assignment]
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
    assert mask.dtype == torch.bool

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
    processor = _load_processor(TEXT_MODEL_REPO, trust_remote_code=True)

    if not hasattr(processor, "apply_chat_template") or processor.chat_template is None:
        pytest.skip("Processor does not support chat templates")

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

    assistant_pattern = _detect_assistant_pattern(processor)
    results = _preprocess_batch(
        examples, processor, max_length=512, assistant_pattern=assistant_pattern
    )

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
def test_preprocess_batch_multimodal(tmp_path):
    """Test preprocessing a batch of multimodal conversations."""
    processor = _load_processor(MM_MODEL_REPO, trust_remote_code=True)

    if not hasattr(processor, "apply_chat_template") or processor.chat_template is None:
        pytest.skip("Processor does not support chat templates")

    img_path = str(tmp_path / "blank.png")
    Image.new("RGB", (256, 256)).save(img_path)

    examples = {
        "conversations": [
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Hello, how are you?"},
                    ],
                },
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": "I am a helpful assistant."},
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "What is the capital of France?"},
                    ],
                },
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "text",
                            "text": "The capital of France is Paris.",
                        },
                    ],
                },
            ],
            [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "What is the difference between these two images?",
                        },
                        {"type": "image", "path": img_path},
                        {"type": "image", "path": img_path},
                    ],
                },
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": "They are the exact same image."},
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Why?"},
                    ],
                },
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "text",
                            "text": "They are both blank.",
                        },
                    ],
                },
            ],
        ]
    }

    assistant_pattern = _detect_assistant_pattern(processor)
    results = _preprocess_batch(
        examples, processor, max_length=2048, assistant_pattern=assistant_pattern
    )

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
    processor = _load_processor(TEXT_MODEL_REPO, trust_remote_code=True)

    if not hasattr(processor, "apply_chat_template") or processor.chat_template is None:
        pytest.skip("Processor does not support chat templates")

    examples: dict[str, list] = {"conversations": []}
    assistant_pattern = _detect_assistant_pattern(processor)
    results = _preprocess_batch(
        examples, processor, max_length=512, assistant_pattern=assistant_pattern
    )

    assert results["input_ids"] == []
    assert results["loss_mask"] == []


@pytest.mark.sanity
def test_preprocess_batch_invalid_conversation():
    """Test preprocessing batch with invalid conversations."""
    processor = _load_processor(TEXT_MODEL_REPO, trust_remote_code=True)

    if not hasattr(processor, "apply_chat_template") or processor.chat_template is None:
        pytest.skip("Processor does not support chat templates")

    examples = {
        "conversations": [
            None,  # Invalid
            [],  # Empty
            [{"role": "user", "content": "Valid"}],  # Valid
        ]
    }

    assistant_pattern = _detect_assistant_pattern(processor)
    results = _preprocess_batch(
        examples, processor, max_length=512, assistant_pattern=assistant_pattern
    )

    # Should only process the valid conversation
    assert len(results["input_ids"]) <= 1
    assert len(results["loss_mask"]) <= 1


@pytest.mark.sanity
def test_preprocess_batch_truncation():
    """Test that long sequences are truncated to max_length."""
    processor = _load_processor(TEXT_MODEL_REPO, trust_remote_code=True)

    if not hasattr(processor, "apply_chat_template") or processor.chat_template is None:
        pytest.skip("Processor does not support chat templates")

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
    assistant_pattern = _detect_assistant_pattern(processor)
    results = _preprocess_batch(
        examples, processor, max_length=max_length, assistant_pattern=assistant_pattern
    )

    if len(results["input_ids"]) > 0:
        # Should be truncated to max_length
        assert len(results["input_ids"][0]) <= max_length
        assert len(results["loss_mask"][0]) <= max_length


@pytest.mark.sanity
def test_preprocess_batch_uses_hf_assistant_mask():
    """Test that HF assistant token mask is used when supported."""
    processor = _load_processor(TEXT_MODEL_REPO, trust_remote_code=True)

    if not hasattr(processor, "apply_chat_template") or processor.chat_template is None:
        pytest.skip("Processor does not support chat templates")

    # Skip test if assistant mask is not supported/functional for this processor
    if not _supports_assistant_mask(processor):
        pytest.skip("Processor does not support assistant token mask")

    examples = {
        "conversations": [
            [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi there!"},
            ]
        ]
    }

    # Pass None to trigger masking path
    results = _preprocess_batch(
        examples,
        processor,
        max_length=128,
        assistant_pattern=None,
    )

    assert "input_ids" in results
    assert "loss_mask" in results
    assert len(results["input_ids"]) == 1
    assert len(results["loss_mask"]) == 1

    # Ensure at least some assistant tokens are trainable
    assert torch.any(results["loss_mask"][0] == 1)


@pytest.mark.sanity
def test_preprocess_batch_falls_back_to_regex():
    """Test that preprocessing falls back to regex-based detection
    when HF mask is unavailable.
    """
    processor = _load_processor(TEXT_MODEL_REPO, trust_remote_code=True)

    if not hasattr(processor, "apply_chat_template") or processor.chat_template is None:
        pytest.skip("Processor does not support chat templates")

    # Monkeypatch apply_chat_template to force HF mask failure
    original_apply_chat_template = processor.apply_chat_template

    def patched_apply_chat_template(*args, **kwargs):
        if kwargs.get("return_assistant_tokens_mask", False):
            raise ValueError("Forcing fallback to regex path")
        return original_apply_chat_template(*args, **kwargs)

    processor.apply_chat_template = patched_apply_chat_template  # type: ignore [method-assign]

    examples = {
        "conversations": [
            [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi!"},
            ]
        ]
    }

    assistant_pattern = _detect_assistant_pattern(processor)

    results = _preprocess_batch(
        examples,
        processor,
        max_length=128,
        assistant_pattern=assistant_pattern,
    )

    assert "input_ids" in results
    assert "loss_mask" in results
    assert len(results["input_ids"]) == 1
    assert len(results["loss_mask"]) == 1

    # Regex path should still mark assistant tokens
    assert torch.any(results["loss_mask"][0] == 1)


@pytest.mark.sanity
def test_preprocess_batch_minimum_valid_tokens_filters_regex_path():
    """Test that minimum_valid_tokens drops short samples on regex path."""
    processor = _load_processor(TEXT_MODEL_REPO, trust_remote_code=True)

    if not hasattr(processor, "apply_chat_template") or processor.chat_template is None:
        pytest.skip("Processor does not support chat templates")

    examples = {
        "conversations": [
            [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "OK"},
            ]
        ]
    }

    assistant_pattern = _detect_assistant_pattern(processor)

    baseline = _preprocess_batch(
        examples,
        processor,
        max_length=128,
        assistant_pattern=assistant_pattern,
    )

    assert len(baseline["loss_mask"]) == 1
    valid_count = int(baseline["loss_mask"][0].sum().item())
    assert valid_count > 0

    filtered = _preprocess_batch(
        examples,
        processor,
        max_length=128,
        assistant_pattern=assistant_pattern,
        minimum_valid_tokens=valid_count + 1,
    )

    assert filtered["input_ids"] == []
    assert filtered["loss_mask"] == []
    assert filtered["seq_len"] == []


@pytest.mark.sanity
def test_preprocess_batch_minimum_valid_tokens_keeps_boundary_case():
    """Test that a sample is kept when valid tokens equal the threshold."""
    processor = _load_processor(TEXT_MODEL_REPO, trust_remote_code=True)

    if not hasattr(processor, "apply_chat_template") or processor.chat_template is None:
        pytest.skip("Processor does not support chat templates")

    examples = {
        "conversations": [
            [
                {"role": "user", "content": "Explain speculative decoding."},
                {
                    "role": "assistant",
                    "content": (
                        "Speculative decoding uses a draft model to propose tokens "
                        "that a verifier model then checks."
                    ),
                },
            ]
        ]
    }

    assistant_pattern = _detect_assistant_pattern(processor)

    baseline = _preprocess_batch(
        examples,
        processor,
        max_length=256,
        assistant_pattern=assistant_pattern,
    )

    assert len(baseline["loss_mask"]) == 1
    valid_count = int(baseline["loss_mask"][0].sum().item())
    assert valid_count > 0

    kept = _preprocess_batch(
        examples,
        processor,
        max_length=256,
        assistant_pattern=assistant_pattern,
        minimum_valid_tokens=valid_count,
    )

    assert len(kept["input_ids"]) == 1
    assert len(kept["loss_mask"]) == 1
    assert int(kept["loss_mask"][0].sum().item()) == valid_count


# Tests for build_eagle3_dataset


@pytest.mark.sanity
def test_build_eagle3_dataset_basic():
    """Test building EAGLE3 dataset from a simple HuggingFace dataset."""
    processor = _load_processor(TEXT_MODEL_REPO, trust_remote_code=True)

    if not hasattr(processor, "apply_chat_template") or processor.chat_template is None:
        pytest.skip("Processor does not support chat templates")

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
    result = build_eagle3_dataset(dataset, processor, max_length=512, num_proc=1)

    assert isinstance(result, HFDataset)
    assert len(result) <= len(dataset)

    # Check that the dataset has the expected columns
    if len(result) > 0:
        assert "input_ids" in result.column_names
        assert "loss_mask" in result.column_names


@pytest.mark.sanity
def test_build_eagle3_dataset_preserves_format():
    """Test that build_eagle3_dataset sets the correct format."""
    processor = _load_processor(TEXT_MODEL_REPO, trust_remote_code=True)

    if not hasattr(processor, "apply_chat_template") or processor.chat_template is None:
        pytest.skip("Processor does not support chat templates")

    data = {
        "conversations": [
            [
                {"role": "user", "content": "Test"},
                {"role": "assistant", "content": "Response"},
            ]
        ]
    }

    dataset = HFDataset.from_dict(data)
    result = build_eagle3_dataset(dataset, processor, max_length=512, num_proc=1)

    # Dataset should be in torch format
    assert result.format["type"] == "torch"


@pytest.mark.sanity
def test_build_eagle3_dataset_removes_original_columns():
    """Test that original columns are removed after processing."""
    processor = _load_processor(TEXT_MODEL_REPO, trust_remote_code=True)

    if not hasattr(processor, "apply_chat_template") or processor.chat_template is None:
        pytest.skip("Processor does not support chat templates")

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
    result = build_eagle3_dataset(dataset, processor, max_length=512, num_proc=1)

    # Original columns should be removed
    if len(result) > 0:
        assert "conversations" not in result.column_names
        assert "extra_column" not in result.column_names


@pytest.mark.sanity
def test_build_eagle3_dataset_minimum_valid_tokens_filters_short_samples():
    """Test that build_eagle3_dataset removes samples below the token threshold."""
    processor = _load_processor(TEXT_MODEL_REPO, trust_remote_code=True)

    if not hasattr(processor, "apply_chat_template") or processor.chat_template is None:
        pytest.skip("Processor does not support chat templates")

    short_conv = [
        {"role": "user", "content": "Hi"},
        {"role": "assistant", "content": "OK"},
    ]
    long_conv = [
        {"role": "user", "content": "Explain speculative decoding."},
        {
            "role": "assistant",
            "content": (
                "Speculative decoding uses a draft model to propose multiple "
                "candidate tokens that a stronger verifier then checks."
            ),
        },
    ]

    assistant_pattern = _detect_assistant_pattern(processor)

    short_baseline = _preprocess_batch(
        {"conversations": [short_conv]},
        processor,
        max_length=256,
        assistant_pattern=assistant_pattern,
    )
    long_baseline = _preprocess_batch(
        {"conversations": [long_conv]},
        processor,
        max_length=256,
        assistant_pattern=assistant_pattern,
    )

    assert len(short_baseline["loss_mask"]) == 1
    assert len(long_baseline["loss_mask"]) == 1

    short_count = int(short_baseline["loss_mask"][0].sum().item())
    long_count = int(long_baseline["loss_mask"][0].sum().item())

    assert short_count > 0
    assert long_count > short_count

    threshold = short_count + 1

    dataset = HFDataset.from_dict({"conversations": [short_conv, long_conv]})
    result = build_eagle3_dataset(
        dataset,
        processor,
        max_length=256,
        num_proc=1,
        assistant_pattern=assistant_pattern,
        minimum_valid_tokens=threshold,
    )

    assert isinstance(result, HFDataset)
    assert len(result) == 1

    remaining_valid_count = int(result[0]["loss_mask"].sum().item())
    assert remaining_valid_count >= threshold


# Tests for turn dropout feature


@pytest.mark.sanity
def test_preprocess_batch_with_turn_dropout():
    """Test preprocessing batch with turn dropout enabled."""
    processor = _load_processor(TEXT_MODEL_REPO, trust_remote_code=True)

    if not hasattr(processor, "apply_chat_template") or processor.chat_template is None:
        pytest.skip("Processor does not support chat templates")

    examples = {
        "conversations": [
            [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi there!"},
                {"role": "user", "content": "How are you?"},
                {"role": "assistant", "content": "I'm good!"},
            ]
        ]
    }

    assistant_pattern = _detect_assistant_pattern(processor)
    results = _preprocess_batch(
        examples,
        processor,
        max_length=512,
        assistant_pattern=assistant_pattern,
        turn_dropout=True,
    )

    # Should still produce valid results
    assert "input_ids" in results
    assert "loss_mask" in results
    assert len(results["input_ids"]) > 0


# Tests for custom assistant pattern feature


@pytest.mark.sanity
def test_detect_assistant_pattern_thinking_model():
    """Test pattern detection with a real thinking model (Qwen3).

    Thinking templates wrap assistant content in <think>...</think> tags.
    The detection uses simple test messages that produce empty think blocks,
    but the pattern must still match real conversations where the think block
    contains substantial content.
    """
    processor = _load_processor("Qwen/Qwen3-8B", trust_remote_code=True)
    pattern = _detect_assistant_pattern(processor)

    # Format a multi-turn conversation with thinking content injected
    # directly into the formatted string (as it would appear in real data)
    test_conv = [
        {"role": "user", "content": "What is 2+2?"},
        {
            "role": "assistant",
            "content": "The answer is 4.",
            "reasoning_content": "We are adding 2 and 2.",
        },
        {"role": "user", "content": "What is 3+3?"},
        {
            "role": "assistant",
            "content": "The answer is 6.",
            "reasoning_content": "We are adding 3 and 3.",
        },
    ]
    formatted: str = processor.apply_chat_template(  # type: ignore[assignment]
        test_conv, tokenize=False, add_generation_prompt=False, enable_thinking=True
    )

    matches = list(re.finditer(pattern, formatted, re.DOTALL))
    assert len(matches) == 2, (
        f"Expected 2 matches, got {len(matches)}.\n"
        f"Pattern: {pattern}\nText: {formatted}"
    )

    # Each match should capture its own assistant content, not the other's
    assert "answer is 4" in matches[0].group(1)
    assert "answer is 6" not in matches[0].group(1)
    assert "answer is 6" in matches[1].group(1)

    # Neither match should contain user content
    for m in matches:
        assert "What is" not in m.group(1)

    # Reasoning content should be stripped from context turns
    assert "2 and 2" not in matches[0].group(1)

    # Reasoning content should be present in the final turn
    assert "3 and 3" in matches[1].group(1)


@pytest.mark.sanity
@pytest.mark.parametrize(
    "thinking_content",
    [
        "",
        "Let me think step by step.\nThe user asked about France.",
    ],
    ids=["no_thinking", "with_thinking"],
)
def test_create_loss_mask_thinking_model(thinking_content):
    """Test _create_loss_mask_from_offsets with Qwen3's thinking template.

    Verifies correct masking both with and without thinking content in the
    <think> block.
    """
    processor = _load_processor("Qwen/Qwen3-8B", trust_remote_code=True)
    pattern = _detect_assistant_pattern(processor)

    # Build formatted text using the real chat template
    conv = [
        {"role": "user", "content": "What is the capital of France?"},
        {"role": "assistant", "content": "Paris is the capital."},
    ]
    if thinking_content:
        conv[-1]["reasoning_content"] = thinking_content
    formatted: str = processor.apply_chat_template(  # type: ignore[assignment]
        conv,
        tokenize=False,
        add_generation_prompt=False,
        enable_thinking=bool(thinking_content),
    )

    # Tokenize with offsets
    encoding = processor(
        formatted,
        return_offsets_mapping=True,
        add_special_tokens=False,
    )
    offsets = encoding["offset_mapping"]

    mask = _create_loss_mask_from_offsets(formatted, offsets, pattern)

    assert len(mask) == len(offsets)
    assert mask.sum() > 0, "Loss mask should not be all zeros"

    # Decode masked vs unmasked regions
    input_ids = torch.tensor(encoding["input_ids"])
    trainable_text = processor.decode(input_ids[mask == 1])
    masked_text = processor.decode(input_ids[mask == 0])

    # Assistant response must be in the trainable region
    assert "Paris is the capital" in trainable_text

    # User message must NOT be in the trainable region
    assert "What is the capital of France" not in trainable_text
    assert "What is the capital of France" in masked_text

    # Thinking content should be in the trainable region (part of assistant turn)
    if thinking_content:
        assert "step by step" in trainable_text


@pytest.mark.sanity
def test_build_eagle3_dataset_with_custom_pattern():
    """Test building dataset with custom assistant pattern."""
    processor = _load_processor(TEXT_MODEL_REPO, trust_remote_code=True)

    if not hasattr(processor, "apply_chat_template") or processor.chat_template is None:
        pytest.skip("Processor does not support chat templates")

    data = {
        "conversations": [
            [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi!"},
            ]
        ]
    }

    # Use a simple custom pattern
    custom_pattern = r"<\|im_start\|>assistant\s*(.*?)<\|im_end\|>"

    dataset = HFDataset.from_dict(data)
    result = build_eagle3_dataset(
        dataset, processor, max_length=512, num_proc=1, assistant_pattern=custom_pattern
    )

    # Should successfully build dataset with custom pattern
    assert isinstance(result, HFDataset)
    assert len(result) > 0
