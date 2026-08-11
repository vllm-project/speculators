"""Unit tests for data processing in speculators.train.data."""

import json
from pathlib import Path

import torch
from datasets import Dataset
from safetensors.torch import save_file

from speculators.models.eagle3.data import shift_batch
from speculators.train.data import (
    ArrowDataset,
    CollateFn,
    build_client_item,
)


def test_shift_batch():
    """Test shift_batch function."""
    batch = {
        "input_ids": torch.tensor([0, 1, 2, 3, 4], dtype=torch.long),
        "hidden_states": torch.tensor(
            [
                [0.0, 0.1, 0.2],
                [1.0, 1.1, 1.2],
                [2.0, 2.1, 2.2],
                [3.0, 3.1, 3.2],
                [4.0, 4.1, 4.2],
            ]
        ),
        "verifier_last_hidden_states": torch.tensor(
            [[10.0], [11.0], [12.0], [13.0], [14.0]]
        ),
        "loss_mask": torch.tensor([0, 0, 1, 1, 1], dtype=torch.long),
        "lengths": torch.tensor([5], dtype=torch.long),
        "position_ids": torch.tensor([0, 1, 2, 3, 4], dtype=torch.long),
    }

    expected_output = {
        "input_ids": torch.tensor([1, 2, 3, 4], dtype=torch.long),
        "hidden_states": torch.tensor(
            [[0.0, 0.1, 0.2], [1.0, 1.1, 1.2], [2.0, 2.1, 2.2], [3.0, 3.1, 3.2]]
        ),
        "verifier_last_hidden_states": torch.tensor([[11.0], [12.0], [13.0], [14.0]]),
        "loss_mask": torch.tensor([0, 1, 1, 1], dtype=torch.long),
        "lengths": torch.tensor([4], dtype=torch.long),
        "position_ids": torch.tensor([1, 2, 3, 4], dtype=torch.long),
    }

    shifted = shift_batch(batch)

    for key, value in shifted.items():
        assert torch.allclose(value, expected_output[key])


def test_collate_fn_basic():
    """Test basic collation functionality."""
    max_len = 10
    hidden_size = 1
    num_target_layers = 3
    collate_fn = CollateFn(
        max_len, hidden_size, num_target_layers=num_target_layers, dtype=torch.float32
    )

    batch = [
        {
            "input_ids": torch.tensor([0, 1], dtype=torch.long),
            "hidden_states": torch.tensor([[0.0, 0.1, 0.2], [1.0, 1.1, 1.2]]),
            "verifier_last_hidden_states": torch.tensor([[2.0], [3.0]]),
            "loss_mask": torch.tensor([0, 1], dtype=torch.long),
            "lengths": torch.tensor([2], dtype=torch.long),
            "position_ids": torch.tensor([0, 1], dtype=torch.long),
        },
        {
            "input_ids": torch.tensor([2, 3, 4, 5, 6, 7], dtype=torch.long),
            "hidden_states": torch.tensor(
                [
                    [4.0, 4.1, 4.2],
                    [5.0, 5.1, 5.2],
                    [6.0, 6.1, 6.2],
                    [7.0, 7.1, 7.2],
                    [8.0, 8.1, 8.2],
                    [9.0, 9.1, 9.2],
                ]
            ),
            "verifier_last_hidden_states": torch.tensor(
                [[10.0], [11.0], [12.0], [13.0], [14.0], [15.0]]
            ),
            "loss_mask": torch.tensor([0, 0, 1, 0, 1, 1], dtype=torch.long),
            "lengths": torch.tensor([6], dtype=torch.long),
            "position_ids": torch.tensor([0, 1, 2, 3, 4, 5], dtype=torch.long),
        },
    ]

    expected_output = {
        "input_ids": torch.tensor([[0, 1, 2, 3, 4, 5, 6, 7, -1, -1]], dtype=torch.long),
        "hidden_states": torch.tensor(
            [
                [
                    [0.0, 0.1, 0.2],
                    [1.0, 1.1, 1.2],
                    [4.0, 4.1, 4.2],
                    [5.0, 5.1, 5.2],
                    [6.0, 6.1, 6.2],
                    [7.0, 7.1, 7.2],
                    [8.0, 8.1, 8.2],
                    [9.0, 9.1, 9.2],
                    [-1, -1, -1],
                    [-1, -1, -1],
                ]
            ]
        ),
        "verifier_last_hidden_states": torch.tensor(
            [[[2.0], [3.0], [10.0], [11.0], [12.0], [13.0], [14.0], [15.0], [-1], [-1]]]
        ),
        "loss_mask": torch.tensor([[0, 1, 0, 0, 1, 0, 1, 1, -1, -1]], dtype=torch.long),
        "document_ids": torch.tensor(
            [[0, 0, 1, 1, 1, 1, 1, 1, -1, -1]], dtype=torch.long
        ),
        "position_ids": torch.tensor(
            [[0, 1, 0, 1, 2, 3, 4, 5, -1, -1]], dtype=torch.long
        ),
    }

    collated = collate_fn(batch)

    for key, value in collated.items():
        assert value.shape == expected_output[key].shape

        is_masking = expected_output[key] == -1
        assert torch.all(
            torch.isclose(value[~is_masking], expected_output[key][~is_masking])
        )


def test_collate_fn_casts_hidden_states_dtype():
    """Test that hidden-states keys are cast to the target dtype during collation."""
    collate_fn = CollateFn(4, 1, dtype=torch.bfloat16)
    batch = [
        {
            "input_ids": torch.tensor([0], dtype=torch.long),
            "hidden_states": torch.ones(1, 3, dtype=torch.float32),
            "verifier_last_hidden_states": torch.ones(1, 1, dtype=torch.float32),
            "loss_mask": torch.ones(1, dtype=torch.long),
            "lengths": torch.tensor([1], dtype=torch.long),
            "position_ids": torch.tensor([0], dtype=torch.long),
        }
    ]

    collated = collate_fn(batch)

    assert collated["hidden_states"].dtype == torch.bfloat16
    assert collated["verifier_last_hidden_states"].dtype == torch.bfloat16
    assert collated["input_ids"].dtype == torch.long


def test_collate_fn_length_truncation():
    """Test that lengths are truncated when they exceed max_len."""
    max_len = 11
    hidden_size = 8
    num_target_layers = 3
    collate_fn = CollateFn(
        max_len, hidden_size, num_target_layers=num_target_layers, dtype=torch.float32
    )

    batch = [
        {
            "input_ids": torch.arange(5, dtype=torch.long),
            "hidden_states": torch.randn(5, num_target_layers * hidden_size),
            "verifier_last_hidden_states": torch.randn(5, hidden_size),
            "loss_mask": torch.ones(5, dtype=torch.long),
            "lengths": torch.tensor([5], dtype=torch.long),
            "position_ids": torch.arange(5, dtype=torch.long),
        },
        {
            "input_ids": torch.arange(7, dtype=torch.long),
            "hidden_states": torch.randn(7, num_target_layers * hidden_size),
            "verifier_last_hidden_states": torch.randn(7, hidden_size),
            "loss_mask": torch.ones(7, dtype=torch.long),
            "lengths": torch.tensor([7], dtype=torch.long),
            "position_ids": torch.arange(7, dtype=torch.long),
        },
    ]

    collated = collate_fn(batch)

    # document_ids: doc 0 has length 5, doc 1 truncated to length 6, rest is padding
    expected_document_ids = torch.tensor(
        [[0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]], dtype=torch.long
    )
    assert torch.equal(collated["document_ids"], expected_document_ids)
    assert "lengths" not in collated

    for key in [
        "input_ids",
        "hidden_states",
        "verifier_last_hidden_states",
        "loss_mask",
        "position_ids",
    ]:
        assert collated[key].shape[0] == 1
        assert collated[key].shape[1] == max_len


def test_arrow_dataset_default_train_ratio_does_not_crash(tmp_path: Path):
    ds = Dataset.from_dict(
        {
            "input_ids": [[1, 2, 3]],
            "loss_mask": [[1, 1, 1]],
            "seq_len": [3],
        }
    )
    ds.save_to_disk(str(tmp_path / "data"))
    (tmp_path / "data" / "hidden_states").mkdir()

    arrow_ds = ArrowDataset(
        max_len=128,
        datapath=str(tmp_path / "data"),
        on_missing="skip",
    )

    # Should not raise AttributeError
    assert arrow_ds._map_to_file_idx(0) == 0
    assert arrow_ds._map_to_file_idx(5) == 5


def test_arrow_dataset_on_generate_cache_creates_hidden_states_dir(tmp_path: Path):
    """on_generate="cache" must create the cache dir when cache() is called —
    otherwise shutil.move into it raises FileNotFoundError, which _maybe_generate_hs
    downgrades to a warning, so caching silently fails for every sample."""
    ds = Dataset.from_dict(
        {
            "input_ids": [[1, 2, 3]],
            "loss_mask": [[1, 1, 1]],
            "seq_len": [3],
        }
    )
    ds.save_to_disk(str(tmp_path / "data"))

    arrow_ds = ArrowDataset(
        max_len=128,
        datapath=str(tmp_path / "data"),
        on_missing="generate",
        on_generate="cache",
    )

    assert hasattr(arrow_ds.transfer, "hidden_states_path")
    # Directory is created lazily when cache() is called
    assert not arrow_ds.transfer.hidden_states_path.exists()

    # Simulate caching a generated sample

    temp_file = tmp_path / "temp_hs.safetensors"
    save_file({"hidden_states": torch.zeros(1, 1)}, temp_file)

    arrow_ds.transfer.cache(str(temp_file), file_idx=0)

    # Now the directory should exist
    assert arrow_ds.transfer.hidden_states_path.is_dir()
    # And the cached file should exist
    assert (arrow_ds.transfer.hidden_states_path / "hs_0.safetensors").exists()


def test_build_client_item_decodes_json_serialized_messages():
    """Multimodal messages stored as a JSON string are decoded and forwarded."""
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Describe this image."},
                {"type": "image_url", "image_url": {"url": "file:///img.png"}},
            ],
        },
        {"role": "assistant", "content": "It is blank."},
    ]
    item = {
        "input_ids": torch.tensor([1, 2, 3], dtype=torch.long),
        "messages": json.dumps(messages),
    }

    client_item = build_client_item(item)

    assert client_item["input_ids"] == [1, 2, 3]
    assert client_item["messages"] == messages


def test_build_client_item_omits_text_only_json_messages():
    """Text-only conversations must not forward messages (input_ids wins)."""
    messages = [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi!"},
    ]
    item = {
        "input_ids": torch.tensor([1, 2], dtype=torch.long),
        "messages": json.dumps(messages),
    }

    assert "messages" not in build_client_item(item)


def test_build_client_item_accepts_structured_messages():
    """Datasets prepared before messages were JSON-serialized store structured
    rows; they must keep working."""
    messages = [
        {
            "role": "user",
            "content": [{"type": "image_url", "image_url": {"url": "file:///i.png"}}],
        },
    ]
    item = {"input_ids": torch.tensor([7], dtype=torch.long), "messages": messages}

    assert build_client_item(item)["messages"] == messages


def test_build_client_item_stringifies_dict_tool_call_arguments():
    """HF chat templates need dict tool-call arguments, but the OpenAI Chat
    Completions schema vLLM validates against requires a JSON string."""
    arguments = {"element": "Submit button", "x": 772, "y": 512}
    messages = [
        {
            "role": "user",
            "content": [{"type": "image_url", "image_url": {"url": "file:///i.png"}}],
        },
        {
            "role": "assistant",
            "content": "Clicking.",
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "click_web", "arguments": arguments},
                }
            ],
        },
    ]
    item = {
        "input_ids": torch.tensor([1], dtype=torch.long),
        "messages": json.dumps(messages),
    }

    sent = build_client_item(item)["messages"]

    sent_arguments = sent[1]["tool_calls"][0]["function"]["arguments"]
    assert isinstance(sent_arguments, str)
    assert json.loads(sent_arguments) == arguments
    # The rest of the payload is untouched
    assert sent[0] == messages[0]
    assert sent[1]["tool_calls"][0]["function"]["name"] == "click_web"


def test_build_client_item_keeps_string_tool_call_arguments():
    """Arguments already serialized as JSON strings pass through unchanged."""
    messages = [
        {
            "role": "user",
            "content": [{"type": "image_url", "image_url": {"url": "file:///i.png"}}],
        },
        {
            "role": "assistant",
            "content": "Clicking.",
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "click", "arguments": '{"x": 65, "y": 105}'},
                }
            ],
        },
    ]
    item = {
        "input_ids": torch.tensor([1], dtype=torch.long),
        "messages": json.dumps(messages),
    }

    assert build_client_item(item)["messages"] == messages


MM_MESSAGES = [
    {
        "role": "user",
        "content": [{"type": "image_url", "image_url": {"url": "file:///i.png"}}],
    },
]
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "click",
            "description": "Click at the given position.",
            "parameters": {"type": "object", "properties": {"x": {"type": "integer"}}},
        },
    }
]


def test_build_client_item_forwards_tools():
    """Tools are rendered into the prompt by the chat template, so they must be
    sent with the request for vLLM's re-render to reproduce input_ids."""
    item = {
        "input_ids": torch.tensor([1], dtype=torch.long),
        "messages": json.dumps(MM_MESSAGES),
        "tools": json.dumps(TOOLS),
    }

    assert build_client_item(item)["tools"] == TOOLS


def test_build_client_item_omits_empty_tools():
    """Conversations tokenized without tools keep the key out of the request."""
    item = {
        "input_ids": torch.tensor([1], dtype=torch.long),
        "messages": json.dumps(MM_MESSAGES),
        "tools": "",
    }

    assert "tools" not in build_client_item(item)


def test_build_client_item_omits_tools_for_text_only_messages():
    """Text-only conversations go through the Completions API, so neither
    messages nor tools may be forwarded."""
    item = {
        "input_ids": torch.tensor([1], dtype=torch.long),
        "messages": json.dumps([{"role": "user", "content": "Hi"}]),
        "tools": json.dumps(TOOLS),
    }

    client_item = build_client_item(item)

    assert "messages" not in client_item
    assert "tools" not in client_item
