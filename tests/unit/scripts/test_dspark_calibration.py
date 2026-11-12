"""Behavior at the offline cache to calibration batch boundary."""

import pytest
import torch
from datasets import Dataset
from safetensors.torch import save_file

from examples.quantization._dspark_calibration import calibration_batches


def test_calibration_batches_require_matching_cached_tokens(tmp_path):
    records = Dataset.from_dict(
        {
            "input_ids": [[1, 2, 3, 4, 5, 6]],
            "loss_mask": [[0, 1, 1, 1, 1, 1]],
            "seq_len": [6],
        }
    ).with_format("torch")
    records.save_to_disk(tmp_path / "records")
    cache = tmp_path / "cache"
    cache.mkdir()
    hidden = torch.tensor([[[2.0, 2.0], [5.0, 5.0], [99.0, 99.0]]] * 6)
    path = cache / "hs_0.safetensors"
    save_file(
        {"token_ids": torch.tensor([1, 2, 3, 4, 5, 6]), "hidden_states": hidden}, path
    )

    batches = calibration_batches(
        tmp_path / "records",
        cache,
        count=1,
        max_length=6,
        max_anchors=10,
        layer_ids=[1, 2],
        captured_layer_ids=[1, 2, 3],
        width=2,
        block_size=2,
    )
    batch = next(iter(batches))
    assert batch["hidden_states"].shape == (1, 6, 4)
    assert batch["hidden_states"][0, 0].tolist() == [2, 2, 5, 5]
    assert batch["max_anchors"] == 3

    save_file(
        {"token_ids": torch.tensor([1, 2, 3, 4, 5, 7]), "hidden_states": hidden}, path
    )
    with pytest.raises(ValueError, match="Token ids don't match"):
        next(iter(batches))
