"""Unit tests for data processing in speculators.train.data."""

import json
from pathlib import Path
from typing import Any, cast

import pytest
import torch
from datasets import Dataset
from safetensors.torch import load_file, save_file

from speculators.models.eagle3.data import shift_batch
from speculators.train.data import (
    ArrowDataset,
    SampleFileDataset,
    _maybe_load_hs_file,
    build_client_item,
    create_collate_fn,
    standardize_data_v1,
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


def test_standardize_data_v1():
    """Test v1 data format standardization."""
    v1_data = {
        "input_ids": torch.tensor([0, 1, 2, 3, 4], dtype=torch.long),
        "loss_mask": torch.tensor([0, 0, 1, 1, 1], dtype=torch.long),
        "hidden_states": [
            torch.tensor(
                [
                    [0.0, 0.1, 0.2],
                    [1.0, 1.1, 1.2],
                    [2.0, 2.1, 2.2],
                    [3.0, 3.1, 3.2],
                    [4.0, 4.1, 4.2],
                ]
            ),
            torch.tensor(
                [
                    [5.0, 5.1, 5.2],
                    [6.0, 6.1, 6.2],
                    [7.0, 7.1, 7.2],
                    [8.0, 8.1, 8.2],
                    [9.0, 9.1, 9.2],
                ]
            ),
            torch.tensor(
                [
                    [10.0, 10.1, 10.2],
                    [11.0, 11.1, 11.2],
                    [12.0, 12.1, 12.2],
                    [13.0, 13.1, 13.2],
                    [14.0, 14.1, 14.2],
                ]
            ),
            torch.tensor(
                [
                    [15.0, 15.1, 15.2],
                    [16.0, 16.1, 16.2],
                    [17.0, 17.1, 17.2],
                    [18.0, 18.1, 18.2],
                    [19.0, 19.1, 19.2],
                ]
            ),
        ],
    }

    expected_output = {
        "hidden_states": torch.tensor(
            [
                [0.0, 0.1, 0.2, 5.0, 5.1, 5.2, 10.0, 10.1, 10.2],
                [1.0, 1.1, 1.2, 6.0, 6.1, 6.2, 11.0, 11.1, 11.2],
                [2.0, 2.1, 2.2, 7.0, 7.1, 7.2, 12.0, 12.1, 12.2],
                [3.0, 3.1, 3.2, 8.0, 8.1, 8.2, 13.0, 13.1, 13.2],
                [4.0, 4.1, 4.2, 9.0, 9.1, 9.2, 14.0, 14.1, 14.2],
            ]
        ),
        "input_ids": torch.tensor([0, 1, 2, 3, 4], dtype=torch.long),
        "verifier_last_hidden_states": torch.tensor(
            [
                [15.0, 15.1, 15.2],
                [16.0, 16.1, 16.2],
                [17.0, 17.1, 17.2],
                [18.0, 18.1, 18.2],
                [19.0, 19.1, 19.2],
            ]
        ),
        "loss_mask": torch.tensor([0, 0, 1, 1, 1], dtype=torch.long),
    }

    standardized = standardize_data_v1(v1_data)

    for key, value in standardized.items():
        assert torch.allclose(value, expected_output[key])


def test_build_client_item_omits_text_only_message_parts():
    """Text-only content neither re-tokenizes nor parses the chat-only tools field."""
    dataset_item = {
        "input_ids": torch.tensor([1, 2, 3]),
        "messages": [
            {
                "role": "user",
                "content": [{"type": "text", "text": "Hello"}],
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": "Hi"}],
            },
        ],
        "tools": "not valid JSON, but irrelevant on the Completion path",
    }

    assert build_client_item(dataset_item) == {"input_ids": [1, 2, 3]}


def test_build_client_item_includes_actual_multimodal_parts():
    """Real media parts must be preserved for vLLM chat generation."""
    tools = [
        {
            "type": "function",
            "function": {
                "name": "describe_image",
                "parameters": {"type": "object", "properties": {}},
            },
        }
    ]
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Describe"},
                {"type": "image_url", "image_url": {"url": "file:///tmp/cat.png"}},
            ],
        }
    ]
    dataset_item = {
        "input_ids": [1, 2, 3],
        "messages": messages,
        "tools": json.dumps(tools, sort_keys=True, separators=(",", ":")),
    }

    assert build_client_item(dataset_item) == {
        "input_ids": [1, 2, 3],
        "messages": messages,
        "tools": tools,
    }


def test_collate_fn_basic():
    """Test basic collation functionality."""
    max_len = 10
    hidden_size = 1
    num_target_layers = 3
    collate_fn = create_collate_fn(
        max_len, hidden_size, num_target_layers=num_target_layers
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


def test_collate_fn_length_truncation():
    """Test that lengths are truncated when they exceed max_len."""
    max_len = 11
    hidden_size = 8
    num_target_layers = 3
    collate_fn = create_collate_fn(
        max_len, hidden_size, num_target_layers=num_target_layers
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


def test_collate_fn_empty_batch_uses_training_dtypes():
    """Test that all-skipped batches keep training-compatible dtypes."""
    max_len = 8
    hidden_size = 4
    collate_fn = create_collate_fn(
        max_len,
        hidden_size,
        hidden_states_dtype=torch.bfloat16,
    )

    collated = collate_fn([None])

    assert collated["hidden_states"].dtype == torch.bfloat16
    assert collated["verifier_last_hidden_states"].dtype == torch.bfloat16
    assert collated["input_ids"].dtype == torch.long
    assert collated["loss_mask"].dtype == torch.long
    assert collated["position_ids"].dtype == torch.long
    assert collated["document_ids"].dtype == torch.long


def test_dataset_getitem_v1_format(tmp_path: Path):
    """Test dataset __getitem__ with v1 data format and dtype conversion."""

    output_dtype = torch.float64
    file_dtype = torch.float32

    # Create v1 format data
    data = {
        "input_ids": torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=torch.long),
        "loss_mask": torch.tensor([0, 0, 1, 1, 1, 1, 1, 1, 1, 1], dtype=torch.long),
        "hidden_states": [
            torch.tensor(
                [
                    [0.0, 0.1],
                    [1.0, 1.1],
                    [2.0, 2.1],
                    [3.0, 3.1],
                    [4.0, 4.1],
                    [5.0, 5.1],
                    [6.0, 6.1],
                    [7.0, 7.1],
                    [8.0, 8.1],
                    [9.0, 9.1],
                ],
                dtype=file_dtype,
            ),
            torch.tensor(
                [
                    [10.0, 10.1],
                    [11.0, 11.1],
                    [12.0, 12.1],
                    [13.0, 13.1],
                    [14.0, 14.1],
                    [15.0, 15.1],
                    [16.0, 16.1],
                    [17.0, 17.1],
                    [18.0, 18.1],
                    [19.0, 19.1],
                ],
                dtype=file_dtype,
            ),
            torch.tensor(
                [
                    [20.0, 20.1],
                    [21.0, 21.1],
                    [22.0, 22.1],
                    [23.0, 23.1],
                    [24.0, 24.1],
                    [25.0, 25.1],
                    [26.0, 26.1],
                    [27.0, 27.1],
                    [28.0, 28.1],
                    [29.0, 29.1],
                ],
                dtype=file_dtype,
            ),
            torch.tensor(
                [
                    [30.0, 30.1],
                    [31.0, 31.1],
                    [32.0, 32.1],
                    [33.0, 33.1],
                    [34.0, 34.1],
                    [35.0, 35.1],
                    [36.0, 36.1],
                    [37.0, 37.1],
                    [38.0, 38.1],
                    [39.0, 39.1],
                ],
                dtype=file_dtype,
            ),
        ],
    }
    expected_output = {
        "input_ids": torch.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=torch.long),
        "hidden_states": torch.tensor(
            [
                [0.0, 0.1, 10.0, 10.1, 20.0, 20.1],
                [1.0, 1.1, 11.0, 11.1, 21.0, 21.1],
                [2.0, 2.1, 12.0, 12.1, 22.0, 22.1],
                [3.0, 3.1, 13.0, 13.1, 23.0, 23.1],
                [4.0, 4.1, 14.0, 14.1, 24.0, 24.1],
                [5.0, 5.1, 15.0, 15.1, 25.0, 25.1],
                [6.0, 6.1, 16.0, 16.1, 26.0, 26.1],
                [7.0, 7.1, 17.0, 17.1, 27.0, 27.1],
                [8.0, 8.1, 18.0, 18.1, 28.0, 28.1],
            ],
            dtype=output_dtype,
        ),
        "verifier_last_hidden_states": torch.tensor(
            [
                [31.0, 31.1],
                [32.0, 32.1],
                [33.0, 33.1],
                [34.0, 34.1],
                [35.0, 35.1],
                [36.0, 36.1],
                [37.0, 37.1],
                [38.0, 38.1],
                [39.0, 39.1],
            ],
            dtype=output_dtype,
        ),
        "loss_mask": torch.tensor([0, 1, 1, 1, 1, 1, 1, 1, 1], dtype=torch.long),
        "lengths": torch.tensor([9], dtype=torch.long),
        "position_ids": torch.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=torch.long),
    }

    file_path = tmp_path / f"data_{0}.pt"
    torch.save(data, file_path)

    dataset = SampleFileDataset(
        max_len=12, file_list=[str(file_path)], hidden_states_dtype=output_dtype
    )

    raw_item = dataset[0]
    assert raw_item is not None
    item = shift_batch(raw_item)

    for key, value in item.items():
        assert torch.allclose(value, expected_output[key]), (
            f"Key {key} does not match expected output"
        )


def test_dataset_loads_lengths_from_sample_lengths_json(tmp_path: Path):
    """Test that approx_lengths are loaded from sample_lengths.json when present."""
    for i in range(3):
        seq_len = 10 + i * 5  # 10, 15, 20
        data = {
            "input_ids": torch.arange(seq_len, dtype=torch.long),
            "loss_mask": torch.ones(seq_len, dtype=torch.long),
            "hidden_states": [
                torch.randn(seq_len, 2, dtype=torch.float32) for _ in range(4)
            ],
        }
        torch.save(data, tmp_path / f"data_{i}.pt")

    # Create sample_lengths.json with exact lengths (after shift_batch reduces by 1)
    expected_lengths = {"0": 9, "1": 14, "2": 19}
    with (tmp_path / "sample_lengths.json").open("w") as f:
        json.dump(expected_lengths, f)

    file_list = sorted([str(f) for f in tmp_path.glob("data_*.pt")])
    dataset = SampleFileDataset(max_len=50, file_list=file_list)

    assert dataset.approx_lengths == [9, 14, 19], (
        f"Expected [9, 14, 19], got {dataset.approx_lengths}"
    )


def test_dataset_fallback_when_sample_lengths_json_missing(tmp_path: Path):
    """Test fallback to file-size approximation when sample_lengths.json is missing."""
    seq_len = 10
    data = {
        "input_ids": torch.arange(seq_len, dtype=torch.long),
        "loss_mask": torch.ones(seq_len, dtype=torch.long),
        "hidden_states": [
            torch.randn(seq_len, 2, dtype=torch.float32) for _ in range(4)
        ],
    }
    torch.save(data, tmp_path / "data_0.pt")

    file_list = [str(tmp_path / "data_0.pt")]
    dataset = SampleFileDataset(max_len=50, file_list=file_list)

    # Should use fallback and return a list with one length
    assert len(dataset.approx_lengths) == 1
    assert dataset.approx_lengths[0] == seq_len


def test_dataset_fallback_when_sample_lengths_json_malformed(tmp_path: Path):
    """Test fallback when sample_lengths.json has missing keys."""
    for i in range(2):
        seq_len = 10
        data = {
            "input_ids": torch.arange(seq_len, dtype=torch.long),
            "loss_mask": torch.ones(seq_len, dtype=torch.long),
            "hidden_states": [
                torch.randn(seq_len, 2, dtype=torch.float32) for _ in range(4)
            ],
        }
        torch.save(data, tmp_path / f"data_{i}.pt")

    # Create malformed sample_lengths.json (missing key "1")
    with (tmp_path / "sample_lengths.json").open("w") as f:
        json.dump({"0": 9}, f)

    file_list = sorted([str(f) for f in tmp_path.glob("data_*.pt")])
    dataset = SampleFileDataset(max_len=50, file_list=file_list)
    assert len(dataset.approx_lengths) == 2


def test_arrow_dataset_default_split_ratio_does_not_crash(tmp_path: Path):
    """ArrowDataset with default split_ratio=1.0 should support indexing."""
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


_TOKENS = (1, 2, 3)


def _arrow_data(tmp_path: Path, input_ids=_TOKENS) -> Path:
    data_path = tmp_path / "data"
    Dataset.from_dict(
        {
            "input_ids": [list(input_ids)],
            "loss_mask": [[1] * len(input_ids)],
            "seq_len": [len(input_ids)],
        }
    ).save_to_disk(str(data_path))
    return data_path


def _hidden_state_file(
    path: Path,
    *,
    token_ids=_TOKENS,
    token_dtype=torch.long,
    hidden_states=None,
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    save_file(
        {
            "hidden_states": hidden_states
            if hidden_states is not None
            else torch.randn(len(token_ids), 4, 2),
            "token_ids": torch.tensor(token_ids, dtype=token_dtype),
        },
        path,
    )
    return path


def _arrow_dataset(
    tmp_path: Path,
    *,
    input_ids=_TOKENS,
    on_missing="raise",
    on_generate="delete",
    create_hidden_root=True,
):
    data_path = _arrow_data(tmp_path, input_ids)
    hidden_root = tmp_path / "hidden_states"
    if create_hidden_root:
        hidden_root.mkdir()
    dataset = ArrowDataset(
        max_len=128,
        datapath=str(data_path),
        hidden_states_path=hidden_root,
        on_missing=on_missing,
        on_generate=on_generate,
    )
    return dataset, hidden_root


def _mock_generation(dataset, monkeypatch, generated):
    def generate(*args, **kwargs):
        del args, kwargs
        if isinstance(generated, Exception):
            raise generated
        return str(generated)

    monkeypatch.setattr("speculators.train.data.generate_hidden_states", generate)
    dataset.client = cast("Any", object())
    dataset.model = "dummy-model"


@pytest.mark.parametrize("on_missing", ["skip", "warn", "raise"])
def test_arrow_dataset_missing_hidden_states_dir_preserves_on_missing_semantics(
    tmp_path: Path,
    on_missing,
):
    missing_root = tmp_path / "missing-hidden-states"
    data_path = _arrow_data(tmp_path)
    arrow_ds = ArrowDataset(
        max_len=128,
        datapath=str(data_path),
        hidden_states_path=missing_root,
        on_missing=on_missing,
    )

    if on_missing == "warn":
        with pytest.warns(UserWarning, match="Failed to load hidden states"):
            assert arrow_ds[0] is None
    elif on_missing == "raise":
        with pytest.raises(RuntimeError, match="Failed to load hidden states"):
            arrow_ds[0]
    else:
        assert arrow_ds[0] is None

    assert not missing_root.exists()


def test_arrow_dataset_on_generate_cache_creates_hidden_states_dir(tmp_path: Path):
    """on_generate="cache" creates its destination before the first write."""
    data_path = _arrow_data(tmp_path)
    arrow_ds = ArrowDataset(
        max_len=128,
        datapath=str(data_path),
        on_missing="generate",
        on_generate="cache",
    )

    assert arrow_ds.hidden_states_path.is_dir()


def test_maybe_load_hs_file_allows_lock_before_output_leaf(
    tmp_path: Path,
    monkeypatch,
):
    hidden_states_path = tmp_path / "hidden_states"
    hidden_states_path.mkdir()
    pending_path = hidden_states_path / "pending.safetensors"
    lock_path = Path(str(pending_path) + ".lock")
    lock_path.write_bytes(b"")

    def finish_write(observed_lock_path):
        assert Path(observed_lock_path) == lock_path
        _hidden_state_file(pending_path)
        lock_path.unlink()

    monkeypatch.setattr("speculators.train.data.wait_for_lock", finish_write)

    loaded = _maybe_load_hs_file(pending_path, hidden_states_path)

    assert loaded is not None
    assert loaded["token_ids"].tolist() == [1, 2, 3]


def test_maybe_load_hs_file_revalidates_after_wait(tmp_path: Path, monkeypatch):
    hidden_states_path = tmp_path / "hidden_states"
    hidden_states_path.mkdir()
    pending_path = hidden_states_path / "pending.safetensors"
    lock_path = Path(str(pending_path) + ".lock")
    lock_path.write_bytes(b"")
    outside_path = tmp_path / "outside.safetensors"
    _hidden_state_file(outside_path)

    def replace_with_symlink(observed_lock_path):
        assert Path(observed_lock_path) == lock_path
        pending_path.symlink_to(outside_path)
        lock_path.unlink()

    monkeypatch.setattr(
        "speculators.train.data.wait_for_lock", replace_with_symlink
    )

    with pytest.raises(ValueError, match="symlink component"):
        _maybe_load_hs_file(pending_path, hidden_states_path)

    assert outside_path.is_file()


def test_arrow_dataset_generate_failure_raises(tmp_path: Path, monkeypatch):
    """on_missing=generate should fail fast when vLLM generation fails."""
    arrow_ds, _ = _arrow_dataset(tmp_path, on_missing="generate")
    _mock_generation(arrow_ds, monkeypatch, RuntimeError("vLLM request failed"))

    with pytest.raises(RuntimeError, match="vLLM request failed"):
        arrow_ds[0]


def test_arrow_dataset_generate_accepts_list_backed_input_ids(
    tmp_path: Path, monkeypatch
):
    """Generated hidden states align when HF returns input_ids as a Python list."""
    arrow_ds, hidden_states_path = _arrow_dataset(
        tmp_path, on_missing="generate"
    )
    generated_path = _hidden_state_file(
        hidden_states_path / "generated.safetensors"
    )
    _mock_generation(arrow_ds, monkeypatch, generated_path)
    assert isinstance(arrow_ds.data[0]["input_ids"], list)

    item = arrow_ds[0]

    assert item is not None
    assert item["input_ids"].tolist() == [1, 2, 3]
    assert not generated_path.exists()


def test_arrow_dataset_generate_cache_moves_valid_source_within_root(
    tmp_path: Path,
    monkeypatch,
):
    arrow_ds, hidden_states_path = _arrow_dataset(
        tmp_path, on_missing="generate", on_generate="cache"
    )
    generated_path = _hidden_state_file(
        hidden_states_path / "response.safetensors"
    )
    _mock_generation(arrow_ds, monkeypatch, generated_path)

    item = arrow_ds[0]

    assert item is not None
    assert not generated_path.exists()
    assert (hidden_states_path / "hs_0.safetensors").is_file()


@pytest.mark.parametrize("on_generate", ["cache", "delete"])
def test_arrow_dataset_rejects_other_managed_cache_as_generated_source(
    tmp_path: Path,
    monkeypatch,
    on_generate,
):
    arrow_ds, hidden_states_path = _arrow_dataset(
        tmp_path, on_missing="generate", on_generate=on_generate
    )
    other_target = _hidden_state_file(hidden_states_path / "hs_1.safetensors")
    _mock_generation(arrow_ds, monkeypatch, other_target)

    with pytest.raises(ValueError, match="another managed cache entry"):
        arrow_ds[0]

    assert other_target.is_file()
    assert not (hidden_states_path / "hs_0.safetensors").exists()


@pytest.mark.parametrize("on_generate", ["cache", "delete"])
def test_arrow_dataset_current_target_requires_actual_prefix_truncation(
    tmp_path: Path,
    monkeypatch,
    on_generate,
):
    arrow_ds, hidden_states_path = _arrow_dataset(
        tmp_path, on_missing="generate", on_generate=on_generate
    )
    current_target = _hidden_state_file(hidden_states_path / "hs_0.safetensors")
    _mock_generation(arrow_ds, monkeypatch, current_target)

    with pytest.raises(ValueError, match="only for an explicit in-place"):
        arrow_ds._maybe_generate_hs(0)

    assert current_target.is_file()
    assert load_file(current_target)["token_ids"].tolist() == [1, 2, 3]


def test_arrow_dataset_truncated_cache_atomically_rewrites_same_source_target(
    tmp_path: Path,
    monkeypatch,
):
    arrow_ds, hidden_states_path = _arrow_dataset(
        tmp_path, on_missing="generate", on_generate="cache"
    )
    target_path = _hidden_state_file(
        hidden_states_path / "hs_0.safetensors", token_ids=(1, 2, 3, 4)
    )
    _mock_generation(arrow_ds, monkeypatch, target_path)
    monkeypatch.setattr(
        "speculators.train.data.build_client_item",
        lambda dataset_item: {
            "input_ids": list(dataset_item["input_ids"]),
            "messages": [{"role": "user", "content": "multimodal"}],
        },
    )
    generated = arrow_ds._maybe_generate_hs(0)

    assert generated["token_ids"].tolist() == [1, 2, 3]
    assert load_file(target_path)["token_ids"].tolist() == [1, 2, 3]
    assert not list(hidden_states_path.glob(".*.tmp"))


@pytest.mark.parametrize("on_generate", ["cache", "delete"])
def test_arrow_dataset_rejects_generated_source_outside_allowed_root(
    tmp_path: Path,
    monkeypatch,
    on_generate,
):
    arrow_ds, hidden_states_path = _arrow_dataset(
        tmp_path, on_missing="generate", on_generate=on_generate
    )
    outside_path = _hidden_state_file(tmp_path / "outside.safetensors")
    _mock_generation(arrow_ds, monkeypatch, outside_path)

    with pytest.raises(ValueError, match="outside the allowed root"):
        arrow_ds[0]

    assert outside_path.is_file()
    assert not (hidden_states_path / "hs_0.safetensors").exists()


@pytest.mark.parametrize("on_generate", ["cache", "delete"])
def test_arrow_dataset_revalidates_source_before_move_or_delete(
    tmp_path: Path,
    monkeypatch,
    on_generate,
):
    arrow_ds, hidden_states_path = _arrow_dataset(
        tmp_path, on_missing="generate", on_generate=on_generate
    )
    generated_path = _hidden_state_file(
        hidden_states_path / "response.safetensors"
    )
    outside_path = _hidden_state_file(tmp_path / "outside.safetensors")
    _mock_generation(arrow_ds, monkeypatch, generated_path)

    def swap_source_after_load(data, tokens, *, allow_prefix_truncation):
        del tokens, allow_prefix_truncation
        generated_path.unlink()
        generated_path.symlink_to(outside_path)
        return data, False

    monkeypatch.setattr(
        "speculators.train.data.align_hidden_states_to_tokens",
        swap_source_after_load,
    )
    with pytest.raises(ValueError, match="symlink component"):
        arrow_ds[0]

    assert outside_path.is_file()
    assert generated_path.is_symlink()
    assert not (hidden_states_path / "hs_0.safetensors").exists()


def test_arrow_dataset_token_id_mismatch_raises(tmp_path: Path):
    """Loaded/generated hidden states must match the preprocessed token IDs."""
    arrow_ds, hidden_states_path = _arrow_dataset(tmp_path)
    _hidden_state_file(
        hidden_states_path / "hs_0.safetensors", token_ids=(1, 2, 4)
    )

    with pytest.raises(RuntimeError, match="don't match input ids"):
        arrow_ds[0]


def test_arrow_dataset_rejects_nonfinite_cached_hidden_states(tmp_path: Path):
    arrow_ds, hidden_states_path = _arrow_dataset(tmp_path)
    hidden_states = torch.randn(3, 4, 2)
    hidden_states[1, 2, 0] = float("inf")
    _hidden_state_file(
        hidden_states_path / "hs_0.safetensors", hidden_states=hidden_states
    )

    with pytest.raises(RuntimeError, match="NaN or Inf"):
        arrow_ds[0]


def test_arrow_dataset_rejects_narrow_unsigned_token_ids_before_wraparound(
    tmp_path: Path,
):
    arrow_ds, hidden_states_path = _arrow_dataset(tmp_path, input_ids=(257,))
    # Dataset token 257 aliases cached uint8 value 1 after a narrowing cast.
    _hidden_state_file(
        hidden_states_path / "hs_0.safetensors",
        token_ids=(1,),
        token_dtype=torch.uint8,
    )

    with pytest.raises(RuntimeError, match="torch.int32 or torch.int64"):
        arrow_ds[0]
