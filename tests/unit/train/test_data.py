"""Unit tests for data processing in speculators.train.data."""

import json
from pathlib import Path

import pytest
import torch

from speculators.train.data import (
    Eagle3SampleFileDataset,
    create_collate_fn,
    shift_batch,
    standardize_data_v0,
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


def test_shift_batch_preserves_input_embeds_override():
    """Test shift_batch keeps input_embeds_override aligned to shifted tokens."""
    batch = {
        "input_ids": torch.tensor([0, 1, 2, 3], dtype=torch.long),
        "hidden_states": torch.randn(4, 3),
        "verifier_last_hidden_states": torch.randn(4, 2),
        "loss_mask": torch.tensor([0, 1, 1, 1], dtype=torch.long),
        "lengths": torch.tensor([4], dtype=torch.long),
        "position_ids": torch.tensor([0, 1, 2, 3], dtype=torch.long),
        "input_embeds_override": torch.tensor(
            [
                [10.0, 10.1],
                [11.0, 11.1],
                [12.0, 12.1],
                [13.0, 13.1],
            ]
        ),
    }

    shifted = shift_batch(batch)

    # Verify embedding override aligns with shifted token indices.
    assert torch.equal(
        shifted["input_embeds_override"],
        torch.tensor([[11.0, 11.1], [12.0, 12.1], [13.0, 13.1]]),
    )


def test_standardize_data_v0():
    """Test v0 data format standardization."""
    v0_data = {
        "input_ids": torch.tensor([0, 1, 2, 3, 4], dtype=torch.long),
        "loss_mask": torch.tensor([0, 0, 1, 1, 1], dtype=torch.long),
        "hidden_state": torch.tensor(
            [
                [0.0, 0.1, 0.2],
                [1.0, 1.1, 1.2],
                [2.0, 2.1, 2.2],
                [3.0, 3.1, 3.2],
                [4.0, 4.1, 4.2],
            ]
        ),
        "target": torch.tensor([[10.0], [11.0], [12.0], [13.0], [14.0]]),
    }

    expected_output = {
        "hidden_states": torch.tensor(
            [
                [0.0, 0.1, 0.2],
                [1.0, 1.1, 1.2],
                [2.0, 2.1, 2.2],
                [3.0, 3.1, 3.2],
                [4.0, 4.1, 4.2],
            ]
        ),
        "input_ids": torch.tensor([0, 1, 2, 3, 4], dtype=torch.long),
        "verifier_last_hidden_states": torch.tensor(
            [[10.0], [11.0], [12.0], [13.0], [14.0]]
        ),
        "loss_mask": torch.tensor([0, 0, 1, 1, 1], dtype=torch.long),
    }

    standardized = standardize_data_v0(v0_data)

    for key, value in standardized.items():
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


def test_standardize_data_v1_with_input_embeds():
    """Test v1 standardization preserves input_embeds."""
    v1_data = {
        "input_ids": torch.tensor([0, 1], dtype=torch.long),
        "loss_mask": torch.tensor([0, 1], dtype=torch.long),
        "hidden_states": [
            torch.tensor([[0.0, 0.1], [1.0, 1.1]]),
            torch.tensor([[2.0, 2.1], [3.0, 3.1]]),
        ],
        "input_embeds": torch.tensor([[9.0, 9.1], [8.0, 8.1]]),
    }

    standardized = standardize_data_v1(v1_data)

    # Verify auxiliary embeddings are mapped to the training override key.
    assert torch.equal(
        standardized["input_embeds_override"],
        torch.tensor([[9.0, 9.1], [8.0, 8.1]]),
    )


def test_collate_fn_basic():
    """Test basic collation functionality."""
    max_len = 10
    collate_fn = create_collate_fn(max_len)

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
        "lengths": torch.tensor([2, 6], dtype=torch.long),
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
    collate_fn = create_collate_fn(max_len)

    batch = [
        {
            "input_ids": torch.arange(5, dtype=torch.long),
            "hidden_states": torch.randn(5, 24),
            "verifier_last_hidden_states": torch.randn(5, 8),
            "loss_mask": torch.ones(5, dtype=torch.long),
            "lengths": torch.tensor([5], dtype=torch.long),
            "position_ids": torch.arange(5, dtype=torch.long),
        },
        {
            "input_ids": torch.arange(7, dtype=torch.long),
            "hidden_states": torch.randn(7, 24),
            "verifier_last_hidden_states": torch.randn(7, 8),
            "loss_mask": torch.ones(7, dtype=torch.long),
            "lengths": torch.tensor([7], dtype=torch.long),
            "position_ids": torch.arange(7, dtype=torch.long),
        },
    ]

    collated = collate_fn(batch)

    # Last length is truncated to fit in max_len
    expected_lengths = torch.tensor([5, 6], dtype=torch.long)

    # All tensors (other than lengths) are concatenated then truncated to max_len
    assert torch.equal(collated["lengths"], expected_lengths)
    for key in [
        "input_ids",
        "hidden_states",
        "verifier_last_hidden_states",
        "loss_mask",
        "position_ids",
    ]:
        assert collated[key].shape[0] == 1
        assert collated[key].shape[1] == max_len


def test_collate_fn_raises_on_mixed_input_embeds_override_presence():
    """Test mixed schema batches fail fast instead of dropping override tensors."""
    collate_fn = create_collate_fn(max_len=8)
    batch = [
        {
            "input_ids": torch.tensor([0, 1], dtype=torch.long),
            "hidden_states": torch.randn(2, 4),
            "verifier_last_hidden_states": torch.randn(2, 2),
            "loss_mask": torch.ones(2, dtype=torch.long),
            "lengths": torch.tensor([2], dtype=torch.long),
            "position_ids": torch.tensor([0, 1], dtype=torch.long),
        },
        {
            "input_ids": torch.tensor([2], dtype=torch.long),
            "hidden_states": torch.randn(1, 4),
            "verifier_last_hidden_states": torch.randn(1, 2),
            "loss_mask": torch.ones(1, dtype=torch.long),
            "lengths": torch.tensor([1], dtype=torch.long),
            "position_ids": torch.tensor([0], dtype=torch.long),
            "input_embeds_override": torch.randn(1, 4),
        },
    ]

    # Mixed old/new sample schemas should fail fast with a clear error.
    with pytest.raises(ValueError, match="missing key 'input_embeds_override'"):
        collate_fn(batch)


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

    dataset = Eagle3SampleFileDataset(
        max_len=12,
        file_list=[str(file_path)],
        standardize_fn=standardize_data_v1,
        hidden_states_dtype=output_dtype,
    )

    item = dataset[0]

    for key, value in item.items():
        assert torch.allclose(value, expected_output[key]), (
            f"Key {key} does not match expected output"
        )


def test_dataset_getitem_v1_with_input_embeds_override(tmp_path: Path):
    """Test dataset converts input_embeds_override to hidden_states_dtype."""

    output_dtype = torch.float64
    file_dtype = torch.float32
    data = {
        "input_ids": torch.tensor([0, 1, 2, 3], dtype=torch.long),
        "loss_mask": torch.tensor([0, 1, 1, 1], dtype=torch.long),
        "hidden_states": [
            torch.randn(4, 2, dtype=file_dtype),
            torch.randn(4, 2, dtype=file_dtype),
        ],
        "input_embeds": torch.randn(4, 2, dtype=file_dtype),
    }
    file_path = tmp_path / "data_0.pt"
    torch.save(data, file_path)

    dataset = Eagle3SampleFileDataset(
        max_len=8,
        file_list=[str(file_path)],
        standardize_fn=standardize_data_v1,
        hidden_states_dtype=output_dtype,
    )
    item = dataset[0]

    # Verify override tensor exists, is shifted, and follows configured dtype.
    assert "input_embeds_override" in item
    assert item["input_embeds_override"].dtype == output_dtype
    assert item["input_embeds_override"].shape[0] == 3


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
    dataset = Eagle3SampleFileDataset(
        max_len=50,
        file_list=file_list,
        standardize_fn=standardize_data_v1,
    )

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
    dataset = Eagle3SampleFileDataset(
        max_len=50,
        file_list=file_list,
        standardize_fn=standardize_data_v1,
    )

    # Should use fallback and return a list with one length
    assert len(dataset.approx_lengths) == 1
    assert dataset.approx_lengths[0] == seq_len - 1  # After shift_batch


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
    dataset = Eagle3SampleFileDataset(
        max_len=50,
        file_list=file_list,
        standardize_fn=standardize_data_v1,
    )
    assert len(dataset.approx_lengths) == 2
