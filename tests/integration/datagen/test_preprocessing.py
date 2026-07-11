"""
Unit tests for the preprocessing module in the Speculators data generation.
"""

import json
from typing import Any
from unittest.mock import patch

import pytest
import torch
from datasets import Dataset as HFDataset
from PIL import Image
from transformers import AutoTokenizer

from speculators.data_generation.configs import (
    DATASET_CONFIGS,
    _normalize_nemotron,
)
from speculators.data_generation.preprocessing import (
    _adapt_conv_for_hf,
    _adapt_conv_for_vllm,
    _load_hf_dataset,
    _normalize_conversation,
    _preprocess_batch,
    build_eagle3_dataset,
    get_tokenizer,
    load_and_preprocess_dataset,
    load_processor,
    load_raw_dataset,
)

# Test model from HuggingFace with chat template
# Using Qwen2-0.5B-Instruct: small (0.5B params), fast model with proper
# chat template support
TEXT_MODEL_REPO = "Qwen/Qwen2-0.5B-Instruct"
# For testing multi-modal support
MM_MODEL_REPO = "Qwen/Qwen3.5-0.8B"


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


@pytest.mark.sanity
def test_normalize_conversation_tool_calls():
    """Test that assistant tool_calls are preserved through normalization."""
    tool_calls: list[dict] = [
        {
            "id": "call_123",
            "type": "function",
            "function": {"name": "get_weather", "arguments": '{"city": "Paris"}'},
        }
    ]
    conv: list[dict] = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is the weather in Paris?"},
        {"role": "assistant", "content": None, "tool_calls": tool_calls},
        {"role": "tool", "content": '{"temperature": 20}', "tool_call_id": "call_123"},
        {"role": "assistant", "content": "The weather in Paris is 20°C."},
    ]
    result = _normalize_conversation(conv)

    assert len(result) == 5
    assert result[0]["role"] == "system"
    assert result[1]["role"] == "user"
    assert result[2]["role"] == "assistant"
    assert result[2]["tool_calls"] == tool_calls
    assert result[2]["content"] == ""
    assert result[3]["role"] == "tool"
    assert result[3]["tool_call_id"] == "call_123"
    assert result[3]["content"] == '{"temperature": 20}'
    assert result[4]["role"] == "assistant"
    assert result[4]["content"] == "The weather in Paris is 20°C."
    assert "tool_calls" not in result[4]
    assert "tool_call_id" not in result[4]


@pytest.mark.sanity
def test_normalize_conversation_tool_role_preserved():
    """Test that tool role messages are preserved and not skipped."""
    conv = [
        {"role": "user", "content": "Run a search"},
        {"role": "tool", "content": "search results", "tool_call_id": "call_456"},
        {"role": "assistant", "content": "Here are the results."},
    ]
    result = _normalize_conversation(conv)

    assert len(result) == 3
    assert result[1]["role"] == "tool"
    assert result[1]["tool_call_id"] == "call_456"


@pytest.mark.sanity
def test_normalize_conversation_tool_calls_not_leaked():
    """Test that tool_calls/tool_call_id are not added to turns that don't have them."""
    conv = [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi!"},
    ]
    result = _normalize_conversation(conv)

    assert len(result) == 2
    assert "tool_calls" not in result[0]
    assert "tool_call_id" not in result[0]
    assert "tool_calls" not in result[1]
    assert "tool_call_id" not in result[1]


# Tests for _adapt_conv_for_hf
@pytest.mark.sanity
def test_adapt_conv_for_hf_text_only_processor():
    """
    Test converting from normalized conversation to HF format
    with a text-only processor (i.e. tokenizer).
    """
    processor = load_processor(TEXT_MODEL_REPO, trust_remote_code=True)

    conv: list[dict] = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": ["Hello"]},
        {"role": "assistant", "content": "Hi!"},
    ]
    result = _adapt_conv_for_hf(conv, processor)

    assert result == conv


@pytest.mark.sanity
def test_adapt_conv_for_hf_multimodal_processor():
    """
    Test converting from normalized conversation to HF format
    with a multi-modal processor.
    """
    processor = load_processor(MM_MODEL_REPO, trust_remote_code=True)

    conv: list[dict] = [
        {
            "role": "system",
            "content": "You are a helpful assistant.",
        },
        {
            "role": "user",
            "content": ["Hello", {"type": "image", "path": "/path/to/img"}],
        },
        {
            "role": "assistant",
            "content": "Hi!",
        },
    ]
    result = _adapt_conv_for_hf(conv, processor)

    assert result == [
        {
            "role": "system",
            "content": [{"type": "text", "text": "You are a helpful assistant."}],
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Hello"},
                {"type": "image", "path": "/path/to/img"},
            ],
        },
        {
            "role": "assistant",
            "content": [{"type": "text", "text": "Hi!"}],
        },
    ]


# Tests for _adapt_conv_for_vllm
@pytest.mark.sanity
def test_adapt_conv_for_vllm_all_content_formats():
    """
    Test converting from normalized conversation to vLLM format
    with each supported content format.
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
    result = _adapt_conv_for_vllm(conv)

    assert len(result) == 2

    assert result[0]["role"] == "system"
    assert result[0]["content"] == "You are a helpful assistant."

    assert result[1]["role"] == "assistant"
    assert result[1]["content"] == [
        {"type": "text", "text": "Hello,"},
        {"type": "text", "text": "I am"},
        {
            "type": "image_url",
            "image_url": {"url": "file:///path/to/img"},
        },
        {
            "type": "image_url",
            "image_url": {"url": "http://path/to/img"},
        },
    ]


@pytest.mark.sanity
def test_adapt_conv_for_vllm_invalid_content_formats():
    """
    Test converting from normalized conversation to vLLM format
    with unsupported content formats.
    """
    with pytest.raises(ValueError, match=r"'image':.* is not supported"):
        _adapt_conv_for_vllm(
            [
                {
                    "role": "assistant",
                    "content": [
                        {"type": "image", "image": Image.new("RGB", (256, 256))},
                    ],
                },
            ]
        )

    with pytest.raises(ValueError, match=r"'base64':.* is not supported"):
        _adapt_conv_for_vllm(
            [
                {
                    "role": "assistant",
                    "content": [
                        {"type": "image", "base64": "abcdef"},
                    ],
                },
            ]
        )


@pytest.mark.sanity
def test_preprocess_batch_basic():
    """Test preprocessing a basic batch of conversations."""
    processor = load_processor(TEXT_MODEL_REPO, trust_remote_code=True)

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

    results = _preprocess_batch(examples, processor, max_length=512)

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
    processor = load_processor(MM_MODEL_REPO, trust_remote_code=True)

    if not hasattr(processor, "apply_chat_template") or processor.chat_template is None:
        pytest.skip("Processor does not support chat templates")

    img_path = str(tmp_path / "blank.png")
    Image.new("RGB", (256, 256)).save(img_path)

    examples = {
        "conversations": [
            [
                {
                    "role": "user",
                    "content": "Hello, how are you?",
                },
                {
                    "role": "assistant",
                    "content": "I am a helpful assistant.",
                },
                {
                    "role": "user",
                    "content": "What is the capital of France?",
                },
                {
                    "role": "assistant",
                    "content": "The capital of France is Paris.",
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
                        {"type": "text", "text": "They are both blank."},
                    ],
                },
            ],
        ]
    }

    results = _preprocess_batch(examples, processor, max_length=2048)

    assert "input_ids" in results
    assert "loss_mask" in results
    # One row per conversation when packed; history-rewriting templates
    # (e.g. thinking scaffolds) fan out to one row per assistant turn.
    n_rows = len(results["input_ids"])
    assert 2 <= n_rows <= 4
    assert len(results["loss_mask"]) == n_rows
    # MM rows carry a messages twin for hidden-state generation
    assert len(results["messages"]) == n_rows

    # Check that input_ids and loss_mask have same length for each row
    for i in range(n_rows):
        assert len(results["input_ids"][i]) == len(results["loss_mask"][i])
        assert isinstance(results["input_ids"][i], torch.Tensor)
        assert isinstance(results["loss_mask"][i], torch.Tensor)
        assert results["loss_mask"][i].sum() > 0


@pytest.mark.sanity
def test_preprocess_batch_empty_conversations():
    """Test preprocessing batch with no conversations."""
    processor = load_processor(TEXT_MODEL_REPO, trust_remote_code=True)

    if not hasattr(processor, "apply_chat_template") or processor.chat_template is None:
        pytest.skip("Processor does not support chat templates")

    examples: dict[str, list] = {"conversations": []}
    results = _preprocess_batch(examples, processor, max_length=512)

    assert results["input_ids"] == []
    assert results["loss_mask"] == []


@pytest.mark.sanity
def test_preprocess_batch_invalid_conversation():
    """Test preprocessing batch with invalid conversations."""
    processor = load_processor(TEXT_MODEL_REPO, trust_remote_code=True)

    if not hasattr(processor, "apply_chat_template") or processor.chat_template is None:
        pytest.skip("Processor does not support chat templates")

    examples = {
        "conversations": [
            None,  # Invalid
            [],  # Empty
            [{"role": "user", "content": "Valid"}],  # Valid
        ]
    }

    results = _preprocess_batch(examples, processor, max_length=512)

    # Should only process the valid conversation
    assert len(results["input_ids"]) <= 1
    assert len(results["loss_mask"]) <= 1


@pytest.mark.sanity
def test_preprocess_batch_truncation():
    """Test that long sequences are truncated to max_length."""
    processor = load_processor(TEXT_MODEL_REPO, trust_remote_code=True)

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
    results = _preprocess_batch(examples, processor, max_length=max_length)

    # The assistant response starts past max_length, so the truncated row has
    # no supervised tokens and is dropped (zero gradient) with a warning.
    assert results["input_ids"] == []

    # With a window that reaches the response, the row is kept and bounded.
    results = _preprocess_batch(examples, processor, max_length=1300)
    assert len(results["input_ids"]) == 1
    assert len(results["input_ids"][0]) <= 1300
    assert len(results["input_ids"][0]) == len(results["loss_mask"][0])
    assert results["loss_mask"][0].sum() > 0


@pytest.mark.sanity
def test_preprocess_batch_minimum_valid_tokens_filters_short_samples():
    """Test that minimum_valid_tokens drops short samples."""
    processor = load_processor(TEXT_MODEL_REPO, trust_remote_code=True)

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

    baseline = _preprocess_batch(
        examples,
        processor,
        max_length=128,
    )

    assert len(baseline["loss_mask"]) == 1
    valid_count = int(baseline["loss_mask"][0].sum().item())
    assert valid_count > 0

    filtered = _preprocess_batch(
        examples,
        processor,
        max_length=128,
        minimum_valid_tokens=valid_count + 1,
    )

    assert filtered["input_ids"] == []
    assert filtered["loss_mask"] == []
    assert filtered["seq_len"] == []


@pytest.mark.sanity
def test_preprocess_batch_minimum_valid_tokens_keeps_boundary_case():
    """Test that a sample is kept when valid tokens equal the threshold."""
    processor = load_processor(TEXT_MODEL_REPO, trust_remote_code=True)

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

    baseline = _preprocess_batch(
        examples,
        processor,
        max_length=256,
    )

    assert len(baseline["loss_mask"]) == 1
    valid_count = int(baseline["loss_mask"][0].sum().item())
    assert valid_count > 0

    kept = _preprocess_batch(
        examples,
        processor,
        max_length=256,
        minimum_valid_tokens=valid_count,
    )

    assert len(kept["input_ids"]) == 1
    assert len(kept["loss_mask"]) == 1
    assert int(kept["loss_mask"][0].sum().item()) == valid_count


# Tests for build_eagle3_dataset


@pytest.mark.sanity
def test_build_eagle3_dataset_basic():
    """Test building EAGLE3 dataset from a simple HuggingFace dataset."""
    processor = load_processor(TEXT_MODEL_REPO, trust_remote_code=True)

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
    processor = load_processor(TEXT_MODEL_REPO, trust_remote_code=True)

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
    processor = load_processor(TEXT_MODEL_REPO, trust_remote_code=True)

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
    processor = load_processor(TEXT_MODEL_REPO, trust_remote_code=True)

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

    short_baseline = _preprocess_batch(
        {"conversations": [short_conv]},
        processor,
        max_length=256,
    )
    long_baseline = _preprocess_batch(
        {"conversations": [long_conv]},
        processor,
        max_length=256,
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
    processor = load_processor(TEXT_MODEL_REPO, trust_remote_code=True)

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

    results = _preprocess_batch(
        examples,
        processor,
        max_length=512,
        turn_dropout=True,
    )

    # Should still produce valid results
    assert "input_ids" in results
    assert "loss_mask" in results
    assert len(results["input_ids"]) > 0


# Tests for tool role and tool_calls / thinking field preservation


@pytest.mark.sanity
def test_normalize_conversation_with_tool_role():
    """Test that 'tool' role is normalized correctly and not skipped."""
    conv: list[dict[str, Any]] = [
        {"role": "user", "content": "Call the weather API"},
        {"role": "assistant", "content": "Sure, calling now."},
        {"role": "tool", "content": '{"temperature": 20}'},
        {"role": "assistant", "content": "It is 20 degrees."},
    ]
    result = _normalize_conversation(conv)

    roles = [t["role"] for t in result]
    assert "tool" in roles, "tool role should be preserved"
    tool_turn = next(t for t in result if t["role"] == "tool")
    assert tool_turn["content"] == '{"temperature": 20}'


@pytest.mark.sanity
def test_normalize_conversation_preserves_tool_calls_field():
    """Test that 'tool_calls' field is preserved when present."""
    tool_calls = [
        {"id": "call_1", "type": "function", "function": {"name": "get_weather"}}
    ]
    conv: list[dict[str, Any]] = [
        {"role": "user", "content": "What's the weather?"},
        {"role": "assistant", "content": "", "tool_calls": tool_calls},
    ]
    result = _normalize_conversation(conv)

    assert len(result) == 2
    assistant_turn = result[1]
    assert "tool_calls" in assistant_turn
    assert assistant_turn["tool_calls"] == tool_calls


@pytest.mark.sanity
def test_preprocess_batch_with_tools():
    """Test that tools from the dataset are forwarded to apply_chat_template.

    tools must be a list of JSON strings (one per conversation in the batch),
    matching the HuggingFace datasets batched-column convention.
    """
    tokenizer = AutoTokenizer.from_pretrained(TEXT_MODEL_REPO, trust_remote_code=True)

    if not hasattr(tokenizer, "apply_chat_template") or tokenizer.chat_template is None:
        pytest.skip("Tokenizer does not support chat templates")

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    example_tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get the current weather",
                "parameters": {
                    "type": "object",
                    "properties": {"location": {"type": "string"}},
                    "required": ["location"],
                },
            },
        }
    ]

    conv = [
        {"role": "user", "content": "What's the weather in Prague?"},
        {"role": "assistant", "content": "Let me check."},
    ]
    # tools must be a list — one JSON string per conversation
    examples_with_tools = {
        "conversations": [conv],
        "tools": [json.dumps(example_tools)],
    }
    examples_without_tools = {
        "conversations": [conv],
    }

    results_with = _preprocess_batch(
        examples_with_tools,
        tokenizer,
        max_length=512,
    )
    results_without = _preprocess_batch(
        examples_without_tools,
        tokenizer,
        max_length=512,
    )

    assert "input_ids" in results_with
    assert "loss_mask" in results_with
    assert len(results_with["input_ids"]) == 1

    # When the template renders tool definitions the sequence must be strictly
    # longer than without tools.  Skip the length check if the template silently
    # ignores the tools kwarg (tool name absent from decoded output).
    decoded_with = tokenizer.decode(results_with["input_ids"][0])
    if "get_weather" in decoded_with:
        assert len(results_with["input_ids"][0]) > len(
            results_without["input_ids"][0]
        ), "Token sequence should be longer when tool definitions are included"


@pytest.mark.sanity
def test_preprocess_batch_with_invalid_tools_json():
    """Test that invalid JSON in the tools column is handled gracefully.

    The pipeline should warn and continue without tools rather than raising.
    """
    tokenizer = AutoTokenizer.from_pretrained(TEXT_MODEL_REPO, trust_remote_code=True)

    if not hasattr(tokenizer, "apply_chat_template") or tokenizer.chat_template is None:
        pytest.skip("Tokenizer does not support chat templates")

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    examples = {
        "conversations": [
            [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi!"},
            ]
        ],
        "tools": ["this is not valid json"],
    }

    # Must not raise; the bad JSON entry is skipped with a warning
    results = _preprocess_batch(examples, tokenizer, max_length=512)

    assert "input_ids" in results
    assert len(results["input_ids"]) == 1


@pytest.mark.sanity
def test_normalize_conversation_tool_calls_with_empty_content():
    """Test that an assistant turn with tool_calls and no text content is normalized."""
    tool_calls = [
        {"id": "call_1", "type": "function", "function": {"name": "get_weather"}}
    ]
    conv: list[dict[str, Any]] = [
        {"role": "user", "content": "What's the weather?"},
        {"role": "assistant", "content": "", "tool_calls": tool_calls},
        {"role": "tool", "content": '{"temperature": 22}'},
        {"role": "assistant", "content": "It is 22 degrees outside."},
    ]
    result = _normalize_conversation(conv)

    assert len(result) == 4
    assistant_tool_call_turn = result[1]
    assert assistant_tool_call_turn["role"] == "assistant"
    assert assistant_tool_call_turn["content"] == ""
    assert assistant_tool_call_turn["tool_calls"] == tool_calls


# Tests for load_raw_dataset resolution chain (issue #661)

PREFIX = "speculators.data_generation.preprocessing"


def _write_jsonl(path, rows):
    """Write rows as newline-delimited JSON to path."""
    path.write_text("\n".join(json.dumps(r) for r in rows))


def _conv_row(prompt: str) -> dict:
    """Return one conversations-format row for the given prompt."""
    return {
        "conversations": [
            {"from": "human", "value": prompt},
            {"from": "gpt", "value": f"answer to {prompt}"},
        ]
    }


@pytest.mark.sanity
def test_load_raw_dataset_local_file(tmp_path):
    """A local .jsonl file loads with no normalize_fn."""
    data_file = tmp_path / "data.jsonl"
    _write_jsonl(data_file, [_conv_row("a"), _conv_row("b")])

    dataset, normalize_fn = load_raw_dataset(str(data_file))

    assert len(dataset) == 2
    assert normalize_fn is None


@pytest.mark.sanity
def test_load_raw_dataset_local_directory(tmp_path):
    """A directory of .json/.jsonl shards loads as one combined dataset."""
    _write_jsonl(tmp_path / "shard1.jsonl", [_conv_row("a"), _conv_row("b")])
    _write_jsonl(tmp_path / "shard2.jsonl", [_conv_row("c")])
    # Nested file is discovered recursively; .json extension also matched.
    nested = tmp_path / "nested"
    nested.mkdir()
    _write_jsonl(nested / "shard3.json", [_conv_row("d")])

    dataset, normalize_fn = load_raw_dataset(str(tmp_path))

    assert len(dataset) == 4
    assert normalize_fn is None


@pytest.mark.sanity
def test_load_raw_dataset_empty_directory_raises(tmp_path):
    """A directory with no .json/.jsonl files raises an actionable error."""
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()

    with pytest.raises(ValueError, match="No .json/.jsonl files found"):
        load_raw_dataset(str(empty_dir))


@pytest.mark.sanity
def test_load_raw_dataset_named_preset():
    """A named preset resolves through DATASET_CONFIGS to load_dataset."""
    sentinel = HFDataset.from_list([_conv_row("x")])
    with patch(f"{PREFIX}.load_dataset", return_value=sentinel) as mock_load:
        dataset, normalize_fn = load_raw_dataset("sharegpt")

    config = DATASET_CONFIGS["sharegpt"]
    mock_load.assert_called_once_with(
        config.hf_path, name=config.subset, split=config.split
    )
    assert dataset is sentinel
    assert normalize_fn is config.normalize_fn


@pytest.mark.sanity
def test_load_raw_dataset_unsupported_source_raises():
    """An unknown source that is not a file/dir/preset/hf: spec raises."""
    with pytest.raises(ValueError, match="Unsupported dataset"):
        load_raw_dataset("not_a_real_preset")


@pytest.mark.sanity
@pytest.mark.parametrize(
    ("spec", "expected_id", "expected_name", "expected_split"),
    [
        ("hf:org/name", "org/name", None, "train"),
        ("hf:org/name:custom_split", "org/name", None, "custom_split"),
        ("hf:org/name:subset:custom_split", "org/name", "subset", "custom_split"),
    ],
)
def test_load_hf_dataset_spec_parsing(spec, expected_id, expected_name, expected_split):
    """hf: specs parse into (id, subset, split) and call load_dataset."""
    sentinel = HFDataset.from_list([_conv_row("x")])
    with patch(f"{PREFIX}.load_dataset", return_value=sentinel) as mock_load:
        dataset, normalize_fn = load_raw_dataset(spec)

    mock_load.assert_called_once_with(
        expected_id, name=expected_name, split=expected_split
    )
    assert dataset is sentinel
    assert normalize_fn is None


# A small public dataset already in conversations format, used to exercise the
# real hf: download path end to end (the RFC asks for a small public dataset).
HF_CONV_DATASET = "philschmid/guanaco-sharegpt-style"


@pytest.mark.sanity
def test_load_raw_dataset_hf_real_download():
    """End-to-end hf: load of a small public conversations dataset.

    Skipped when the dataset cannot be fetched, e.g. network-restricted CI
    runners that only have a pre-baked model cache. The hf: parsing and the
    conversations guard are covered deterministically by the mocked tests above.
    """
    try:
        dataset, normalize_fn = load_raw_dataset(f"hf:{HF_CONV_DATASET}")
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"Could not fetch {HF_CONV_DATASET}: {exc}")

    assert normalize_fn is None
    assert "conversations" in dataset.column_names
    assert len(dataset) > 0


@pytest.mark.sanity
def test_load_hf_dataset_non_conversations_raises():
    """An hf: dataset without a conversations column fails loudly."""
    non_conv = HFDataset.from_list([{"prompt": "hi", "response": "there"}])
    with (
        patch(f"{PREFIX}.load_dataset", return_value=non_conv),
        pytest.raises(ValueError, match="conversations format"),
    ):
        _load_hf_dataset("hf:org/name")


@pytest.mark.sanity
@pytest.mark.parametrize(
    "spec",
    [
        "hf:org/name:a:b:c",  # too many segments
        "hf:",  # missing id
        "hf:org/name:",  # trailing colon -> empty split
        "hf:org/name:subset:",  # empty split with subset
        "hf:org/name::split",  # empty subset
    ],
)
def test_load_hf_dataset_malformed_spec_raises(spec):
    """Malformed hf: specs raise locally without touching the network."""
    with (
        patch(f"{PREFIX}.load_dataset") as mock_load,
        pytest.raises(ValueError, match="Invalid hf: spec"),
    ):
        _load_hf_dataset(spec)
    mock_load.assert_not_called()


@pytest.mark.sanity
def test_dataset_configs_has_magpie_and_nemotron():
    """magpie and nemotron presets are registered."""
    assert "magpie" in DATASET_CONFIGS
    assert "nemotron" in DATASET_CONFIGS
    # magpie ships conversations already, so needs no normalize_fn.
    assert DATASET_CONFIGS["magpie"].normalize_fn is None


@pytest.mark.sanity
def test_normalize_nemotron_builds_conversations():
    """nemotron normalize_fn appends the output as an assistant turn."""
    example = {
        "input": [{"role": "user", "content": "hi"}],
        "output": "hello",
    }
    result = _normalize_nemotron(example)

    assert result["conversations"] == [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello"},
    ]


# Tests for load_and_preprocess_dataset


@pytest.mark.sanity
def test_load_and_preprocess_dataset_shuffles_combined_datasets(tmp_path):
    """Combined datasets must be shuffled before truncating to max_samples,
    otherwise only samples from the first dataset are kept."""
    paths = []
    for marker in ("ALPHA", "BETA"):
        path = tmp_path / f"{marker.lower()}.jsonl"
        path.write_text(
            "\n".join(
                json.dumps(
                    {
                        "conversations": [
                            {"role": "user", "content": f"Question {i}?"},
                            {"role": "assistant", "content": f"{marker} answer {i}."},
                        ]
                    }
                )
                for i in range(12)
            )
        )
        paths.append(str(path))

    dataset, processor = load_and_preprocess_dataset(
        TEXT_MODEL_REPO,
        paths,
        seq_length=128,
        build_dataset_num_proc=1,
        seed=0,
        max_samples=12,
        token_freq_path=tmp_path / "token_freq.pt",
        trust_remote_code=True,
    )

    assert len(dataset) == 12
    tokenizer = get_tokenizer(processor)
    decoded = [tokenizer.decode(row["input_ids"]) for row in dataset]
    assert any("BETA" in text for text in decoded)
