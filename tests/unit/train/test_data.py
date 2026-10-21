"""Unit tests for data processing in speculators.train.data."""

import logging
from pathlib import Path

import pytest
import torch
from datasets import Dataset

import speculators.train.hidden_states_source as source_module
from speculators.models.eagle3.data import shift_batch
from speculators.train.data import (
    ArrowDataset,
    SpeculatorsDataset,
    _add_document_ids,
    _pad_samples,
)
from speculators.train.data_types import DatasetRecord
from speculators.train.recovery import (
    RECOVERY_METADATA_KEY,
    RecoveryMetadata,
    SampleUnavailable,
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


def test_pad_samples_basic():
    """Test padding and document-ID construction."""
    max_len = 10

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

    collated = _pad_samples(batch, max_len, torch.float32)
    _add_document_ids(collated, max_len)

    for key, value in collated.items():
        if isinstance(value, torch.Tensor):
            assert isinstance(expected_output[key], torch.Tensor)
            assert value.shape == expected_output[key].shape  # type: ignore[attr-defined]

            is_masking = expected_output[key] == -1
            assert torch.all(
                torch.isclose(value[~is_masking], expected_output[key][~is_masking])  # type: ignore[index]
            )

        else:
            assert value == expected_output[key]


def test_pad_samples_casts_hidden_states_dtype():
    """Test that hidden-states keys are cast to the target dtype during padding."""
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

    collated = _pad_samples(batch, 4, torch.bfloat16)

    assert collated["hidden_states"].dtype == torch.bfloat16
    assert collated["verifier_last_hidden_states"].dtype == torch.bfloat16
    assert collated["input_ids"].dtype == torch.long


def test_pad_samples_length_truncation():
    """Test that lengths are truncated when they exceed max_len."""
    max_len = 11
    hidden_size = 8
    num_target_layers = 3

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

    collated = _pad_samples(batch, max_len, torch.float32)
    _add_document_ids(collated, max_len)

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
        datapath=str(tmp_path / "data"),
    )

    # Should not raise AttributeError
    assert arrow_ds._map_to_storage_index(0) == 0
    assert arrow_ds._map_to_storage_index(5) == 5


class _SequenceTransfer:
    """Minimal transfer fake which returns or raises queued generated results."""

    def __init__(self, generated_results):
        self.generated_results = list(generated_results)
        self.deleted: list[str] = []

    def setup(self):
        return None

    def get_cached(self, _file_idx):
        return None

    def get_generated(self, _hs_transfer_params):
        result = self.generated_results.pop(0)
        if isinstance(result, Exception):
            raise result
        return result

    def delete(self, handle):
        self.deleted.append(handle)


def _make_generation_dataset(
    tmp_path: Path,
    transfer: _SequenceTransfer,
    num_records: int = 1,
    **kwargs,
) -> SpeculatorsDataset:
    ds = Dataset.from_dict(
        {
            "input_ids": [[1, 2, 3]] * num_records,
            "loss_mask": [[1, 1, 1]] * num_records,
            "seq_len": [3] * num_records,
        }
    )
    ds.save_to_disk(str(tmp_path / "data"))
    records = ArrowDataset(
        datapath=str(tmp_path / "data"),
    )
    records.data.set_format(type="torch")
    source = source_module.VllmHiddenStatesSource(
        transfer=transfer,  # type: ignore[arg-type]
        model="model",
        generation_validation_retries=kwargs.pop("generation_validation_retries", 2),
        max_consecutive_generation_failures=kwargs.pop(
            "max_consecutive_generation_failures", 20
        ),
    )
    dataset = SpeculatorsDataset(
        records=records,
        max_len=8,
        hidden_states=source,
        hidden_size=4,
        num_target_layers=kwargs.pop("num_target_layers", 1),
    )
    source.client = object()  # type: ignore[assignment]
    return dataset


def test_arrow_dataset_builds_explicit_dataset_record(tmp_path):
    dataset = _make_generation_dataset(tmp_path, _SequenceTransfer([]))

    record = dataset.records[0]

    assert torch.equal(record.input_ids, torch.tensor([1, 2, 3]))
    assert torch.equal(record.loss_mask, torch.tensor([1, 1, 1]))
    assert record.seq_len == 3
    assert record.storage_index == 0
    assert record.messages is None


def _valid_generated_sample() -> dict[str, torch.Tensor]:
    return {
        "hidden_states": torch.ones(3, 2, 4, dtype=torch.bfloat16),
        "token_ids": torch.tensor([1, 2, 3], dtype=torch.long),
    }


def _generated_sample(num_layers: int, hidden_size: int = 4) -> dict[str, torch.Tensor]:
    """Build a payload whose layer values make truncation observable."""
    capture = torch.arange(1, num_layers + 1, dtype=torch.bfloat16)
    return {
        "hidden_states": capture.view(1, -1, 1).expand(3, num_layers, hidden_size),
        "token_ids": torch.tensor([1, 2, 3], dtype=torch.long),
    }


def test_speculators_dataset_getitems_builds_collated_batch(tmp_path, monkeypatch):
    transfer = _SequenceTransfer([_valid_generated_sample()])
    dataset = _make_generation_dataset(tmp_path, transfer)
    handles = iter(["handle"])
    monkeypatch.setattr(
        source_module,
        "generate_hidden_states",
        lambda *_args, **_kwargs: _next_generation_result(handles),
    )

    batch = dataset.__getitems__([0])

    assert batch["input_ids"].shape == (1, 8)
    assert batch["hidden_states"].shape == (1, 8, 4)
    assert batch["document_ids"].shape == (1, 8)
    assert batch["error_records"] == 0


def test_offline_hidden_states_source_loads_cached_sample():
    cached = _valid_generated_sample()

    class CachedTransfer:
        def __init__(self):
            self.requested_indices = []

        def get_cached(self, file_index):
            self.requested_indices.append(file_index)
            return cached

    transfer = CachedTransfer()
    source = source_module.OfflineHiddenStatesSource(
        transfer=transfer,  # type: ignore[arg-type]
        on_missing="raise",
    )

    record = DatasetRecord(
        input_ids=torch.tensor([1, 2, 3]),
        loss_mask=torch.ones(3, dtype=torch.bool),
        seq_len=3,
        storage_index=17,
    )
    assert source.get_one(record) is cached
    assert transfer.requested_indices == [17]


@pytest.mark.parametrize("on_missing", ["skip", "warn"])
def test_offline_hidden_states_source_handles_cache_miss(on_missing):
    class MissingTransfer:
        def get_cached(self, _storage_index):
            return None

    record = DatasetRecord(
        input_ids=torch.tensor([1, 2, 3]),
        loss_mask=torch.ones(3, dtype=torch.bool),
        seq_len=3,
        storage_index=17,
    )
    source = source_module.OfflineHiddenStatesSource(
        transfer=MissingTransfer(),  # type: ignore[arg-type]
        on_missing=on_missing,
    )

    if on_missing == "warn":
        with pytest.warns(UserWarning, match="sample 17"):
            result = source.get_one(record)
    else:
        result = source.get_one(record)

    assert isinstance(result, SampleUnavailable)


def test_offline_hidden_states_source_raises_on_cache_miss():
    class MissingTransfer:
        def get_cached(self, _storage_index):
            return None

    record = DatasetRecord(
        input_ids=torch.tensor([1, 2, 3]),
        loss_mask=torch.ones(3, dtype=torch.bool),
        seq_len=3,
        storage_index=17,
    )
    source = source_module.OfflineHiddenStatesSource(
        transfer=MissingTransfer(),  # type: ignore[arg-type]
        on_missing="raise",
    )

    with pytest.raises(RuntimeError, match="sample 17"):
        source.get_one(record)


def test_vllm_hidden_states_source_close_is_idempotent():
    class CloseableClient:
        def __init__(self):
            self.close_calls = 0

        def close(self):
            self.close_calls += 1

    source = source_module.VllmHiddenStatesSource(
        transfer=object(),  # type: ignore[arg-type]
    )
    client = CloseableClient()
    source.client = client  # type: ignore[assignment]

    source.close()
    source.close()

    assert client.close_calls == 1
    assert source.client is None


def _next_generation_result(handles):
    handle = next(handles)
    return handle, {"handle": handle}


def test_arrow_dataset_retries_nonfinite_read_then_recovers(
    tmp_path, monkeypatch, caplog
):
    corrupt = _valid_generated_sample()
    corrupt["hidden_states"][1, 0, 0] = float("nan")
    transfer = _SequenceTransfer([corrupt, _valid_generated_sample()])
    arrow_ds = _make_generation_dataset(
        tmp_path,
        transfer,
        generation_validation_retries=1,
    )
    handles = iter(["bad-handle", "good-handle"])
    monkeypatch.setattr(
        source_module,
        "generate_hidden_states",
        lambda *_args, **_kwargs: _next_generation_result(handles),
    )

    with caplog.at_level(logging.WARNING, logger="speculators"):
        item = arrow_ds[0]

    assert isinstance(item, dict)
    assert "non-finite" in caplog.text
    assert item["hidden_states"].shape == (3, 4)
    assert torch.isfinite(item["hidden_states"]).all()
    assert transfer.deleted == ["bad-handle", "good-handle"]
    assert isinstance(arrow_ds.hidden_states, source_module.VllmHiddenStatesSource)
    assert arrow_ds.hidden_states.generation_recovery.consecutive_failures == 0


def test_arrow_dataset_payload_matching_target_layers_feeds_all_entries(
    tmp_path, monkeypatch
):
    num_target_layers = 3
    hidden_size = 4
    sample = _generated_sample(num_target_layers, hidden_size)
    transfer = _SequenceTransfer([sample])
    arrow_ds = _make_generation_dataset(
        tmp_path, transfer, num_target_layers=num_target_layers
    )
    monkeypatch.setattr(
        source_module,
        "generate_hidden_states",
        lambda *_args, **_kwargs: ("handle-0", {"handle": "handle-0"}),
    )

    item = arrow_ds[0]

    assert isinstance(item, dict)
    assert item["hidden_states"].shape == (3, num_target_layers * hidden_size)
    assert torch.equal(item["hidden_states"], sample["hidden_states"].flatten(1))
    assert torch.equal(
        item["verifier_last_hidden_states"], sample["hidden_states"][:, -1]
    )


def test_arrow_dataset_payload_one_past_target_layers_drops_trailing(
    tmp_path, monkeypatch
):
    num_target_layers = 3
    hidden_size = 4
    sample = _generated_sample(num_target_layers + 1, hidden_size)
    transfer = _SequenceTransfer([sample])
    arrow_ds = _make_generation_dataset(
        tmp_path, transfer, num_target_layers=num_target_layers
    )
    monkeypatch.setattr(
        source_module,
        "generate_hidden_states",
        lambda *_args, **_kwargs: ("handle-0", {"handle": "handle-0"}),
    )

    item = arrow_ds[0]

    assert isinstance(item, dict)
    assert item["hidden_states"].shape == (3, num_target_layers * hidden_size)
    assert torch.equal(
        item["hidden_states"], sample["hidden_states"][:, :-1].flatten(1)
    )
    assert torch.equal(
        item["verifier_last_hidden_states"], sample["hidden_states"][:, -1]
    )


def test_exhausted_generation_produces_locally_empty_zero_loss_batch(
    tmp_path, monkeypatch, caplog
):
    transfer = _SequenceTransfer(
        [ValueError("checksum mismatch"), ValueError("checksum mismatch")]
    )
    arrow_ds = _make_generation_dataset(
        tmp_path,
        transfer,
        generation_validation_retries=1,
        max_consecutive_generation_failures=10,
    )
    handles = iter(["bad-1", "bad-2"])
    monkeypatch.setattr(
        source_module,
        "generate_hidden_states",
        lambda *_args, **_kwargs: _next_generation_result(handles),
    )

    with caplog.at_level(logging.WARNING, logger="speculators"):
        collated = arrow_ds.__getitems__([0])

    assert "checksum mismatch" in caplog.text
    assert collated["error_records"] == 1
    metadata = collated[RECOVERY_METADATA_KEY]
    assert isinstance(metadata, RecoveryMetadata)
    assert metadata.locally_empty
    assert metadata.failure_count == 1
    assert not metadata.fatal
    assert not collated["loss_mask"].bool().any()
    assert torch.equal(collated["document_ids"], torch.full((1, 8), -1))
    assert collated["hidden_states"].shape == (1, 8, 4)
    assert collated["hidden_states"].dtype == torch.bfloat16


def test_consecutive_generation_failures_trip_circuit_breaker(
    tmp_path, monkeypatch, caplog
):
    transfer = _SequenceTransfer(
        [
            ValueError("bad read 1"),
            ValueError("bad read 2"),
            ValueError("bad read 3"),
            ValueError("bad read 4"),
        ]
    )
    arrow_ds = _make_generation_dataset(
        tmp_path,
        transfer,
        generation_validation_retries=1,
        max_consecutive_generation_failures=2,
    )
    handles = iter(["bad-1", "bad-2", "bad-3", "bad-4"])
    monkeypatch.setattr(
        source_module,
        "generate_hidden_states",
        lambda *_args, **_kwargs: _next_generation_result(handles),
    )

    with caplog.at_level(logging.WARNING, logger="speculators"):
        first_batch = arrow_ds.__getitems__([0])
        collated = arrow_ds.__getitems__([0])

    assert "consecutive failures=2/2" in caplog.text
    first_metadata = first_batch[RECOVERY_METADATA_KEY]
    assert isinstance(first_metadata, RecoveryMetadata)
    assert not first_metadata.fatal
    metadata = collated[RECOVERY_METADATA_KEY]
    assert isinstance(metadata, RecoveryMetadata)
    assert metadata.fatal
    assert "bad read 4" in metadata.error


def test_getitems_keeps_valid_samples_when_one_generation_fails(tmp_path, monkeypatch):
    transfer = _SequenceTransfer(
        [ValueError("transient read"), _valid_generated_sample()]
    )
    dataset = _make_generation_dataset(
        tmp_path,
        transfer,
        num_records=2,
        generation_validation_retries=0,
    )
    handles = iter(["bad", "good"])
    monkeypatch.setattr(
        source_module,
        "generate_hidden_states",
        lambda *_args, **_kwargs: _next_generation_result(handles),
    )

    collated = dataset.__getitems__([0, 1])

    assert collated["error_records"] == 1
    metadata = collated[RECOVERY_METADATA_KEY]
    assert isinstance(metadata, RecoveryMetadata)
    assert not metadata.locally_empty
    assert metadata.failure_count == 1
    assert collated["loss_mask"].sum() == 3
    assert torch.equal(collated["input_ids"][0, :3], torch.tensor([1, 2, 3]))
