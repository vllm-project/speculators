from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
import torch
from datasets import Dataset
from safetensors.torch import load_file, save_file

import speculators.train.data as data_module
import speculators.train.dataloader as dataloader_module
from speculators.train.data import ArrowDataset

if TYPE_CHECKING:
    from pathlib import Path


def _write_dataset(path: Path) -> None:
    dataset = Dataset.from_dict(
        {
            "input_ids": [[1, 2, 3]],
            "loss_mask": [[0, 1, 1]],
            "seq_len": [3],
        }
    ).with_format("torch")
    dataset.save_to_disk(path)


def _hidden_states() -> dict[str, torch.Tensor]:
    return {
        "token_ids": torch.tensor([1, 2, 3]),
        "hidden_states": torch.arange(12, dtype=torch.float32).reshape(3, 4),
    }


def _arrow_dataset(
    data_path: Path,
    *,
    shared_path: Path | None,
    hidden_states_path: Path,
    on_generate: str = "delete",
) -> ArrowDataset:
    dataset = ArrowDataset(
        max_len=128,
        datapath=data_path,
        hidden_states_path=hidden_states_path,
        model="model",
        on_missing="generate",
        on_generate=on_generate,
        shared_artifacts_path=shared_path,
        shared_artifacts_namespace=("layers:2,18,33" if shared_path else None),
        shared_artifacts_ttl_seconds=None,
    )
    dataset.client = object()
    return dataset


def test_shared_dataset_requires_identity_namespace(tmp_path):
    data_path = tmp_path / "data"
    _write_dataset(data_path)

    with pytest.raises(ValueError, match="shared_artifacts_namespace is required"):
        ArrowDataset(
            max_len=128,
            datapath=data_path,
            model="model",
            shared_artifacts_path=tmp_path / "shared",
        )


def _successful_generator(service_path: Path, calls: list[Path]):
    def generate(*_args, **_kwargs):
        path = service_path / f"request-{len(calls)}.safetensors"
        save_file(_hidden_states(), path)
        (path.parent / f"{path.name}.lock").touch()
        calls.append(path)
        return str(path)

    return generate


def test_shared_dataset_requests_publish_once_and_delete_service_temporary(
    tmp_path, monkeypatch
):
    data_path = tmp_path / "data"
    service_path = tmp_path / "service"
    shared_path = tmp_path / "shared"
    service_path.mkdir()
    _write_dataset(data_path)
    calls: list[Path] = []
    monkeypatch.setattr(
        data_module,
        "generate_hidden_states",
        _successful_generator(service_path, calls),
    )
    first = _arrow_dataset(
        data_path,
        shared_path=shared_path,
        hidden_states_path=tmp_path / "first-index",
    )
    second = _arrow_dataset(
        data_path,
        shared_path=shared_path,
        hidden_states_path=tmp_path / "second-index",
    )

    first_data = first._maybe_generate_hs(0)
    second_data = second._maybe_generate_hs(0)

    assert first_data is not None
    assert second_data is not None
    assert torch.equal(first_data["hidden_states"], second_data["hidden_states"])
    assert len(calls) == 1
    assert not calls[0].exists()
    assert not (calls[0].parent / f"{calls[0].name}.lock").exists()
    assert first.artifact_cache is not None
    stats = first.artifact_cache.snapshot_stats()
    assert stats["logical_requests"] == 2
    assert stats["misses"] == 1
    assert stats["hits"] == 1
    assert stats["publishes"] == 1


def test_unconfigured_dataset_keeps_existing_per_request_delete_behavior(
    tmp_path, monkeypatch
):
    data_path = tmp_path / "data"
    service_path = tmp_path / "service"
    service_path.mkdir()
    _write_dataset(data_path)
    calls: list[Path] = []
    monkeypatch.setattr(
        data_module,
        "generate_hidden_states",
        _successful_generator(service_path, calls),
    )
    first = _arrow_dataset(
        data_path,
        shared_path=None,
        hidden_states_path=tmp_path / "first-index",
    )
    second = _arrow_dataset(
        data_path,
        shared_path=None,
        hidden_states_path=tmp_path / "second-index",
    )

    assert first._maybe_generate_hs(0) is not None
    assert second._maybe_generate_hs(0) is not None

    assert first.artifact_cache is None
    assert len(calls) == 2
    assert all(not path.exists() for path in calls)


def test_shared_dataset_does_not_publish_partial_service_artifact(
    tmp_path, monkeypatch
):
    data_path = tmp_path / "data"
    service_path = tmp_path / "service"
    shared_path = tmp_path / "shared"
    service_path.mkdir()
    _write_dataset(data_path)
    calls: list[Path] = []

    def generate(*_args, **_kwargs):
        path = service_path / f"request-{len(calls)}.safetensors"
        if not calls:
            path.write_bytes(b"partial")
        else:
            save_file(_hidden_states(), path)
        calls.append(path)
        return str(path)

    monkeypatch.setattr(data_module, "generate_hidden_states", generate)
    dataset = _arrow_dataset(
        data_path,
        shared_path=shared_path,
        hidden_states_path=tmp_path / "index",
    )

    with pytest.warns(UserWarning, match="Failed to load/cache"):
        assert dataset._maybe_generate_hs(0) is None
    assert not calls[0].exists()
    assert list((shared_path / "artifacts").glob("*/*.safetensors")) == []

    loaded = dataset._maybe_generate_hs(0)

    assert loaded is not None
    assert len(calls) == 2
    assert not calls[1].exists()
    assert dataset.artifact_cache is not None
    stats = dataset.artifact_cache.snapshot_stats()
    assert stats["generation_failures"] == 1
    assert stats["publishes"] == 1


def test_shared_dataset_preserves_service_artifact_after_lock_timeout(
    tmp_path, monkeypatch
):
    data_path = tmp_path / "data"
    service_path = tmp_path / "service"
    shared_path = tmp_path / "shared"
    service_path.mkdir()
    _write_dataset(data_path)
    artifact_path = service_path / "request.safetensors"
    artifact_path.write_bytes(b"partial")
    lock_path = artifact_path.parent / f"{artifact_path.name}.lock"
    lock_path.touch()

    def time_out_waiting_for_lock(*_args, **_kwargs):
        raise TimeoutError("active")

    monkeypatch.setattr(
        data_module,
        "generate_hidden_states",
        lambda *_args, **_kwargs: str(artifact_path),
    )
    monkeypatch.setattr(
        data_module,
        "wait_for_lock",
        time_out_waiting_for_lock,
    )
    dataset = _arrow_dataset(
        data_path,
        shared_path=shared_path,
        hidden_states_path=tmp_path / "index",
    )

    with pytest.warns(UserWarning, match="Failed to load/cache"):
        assert dataset._maybe_generate_hs(0) is None

    assert artifact_path.exists()
    assert lock_path.exists()
    assert list((shared_path / "artifacts").glob("*/*.safetensors")) == []
    assert dataset.artifact_cache is not None
    stats = dataset.artifact_cache.snapshot_stats()
    assert stats["logical_requests"] == 1
    assert stats["misses"] == 1
    assert stats["generation_failures"] == 1
    assert stats["publishes"] == 0


def test_shared_dataset_preserves_on_generate_index_cache(tmp_path, monkeypatch):
    data_path = tmp_path / "data"
    service_path = tmp_path / "service"
    shared_path = tmp_path / "shared"
    index_path = tmp_path / "index"
    service_path.mkdir()
    _write_dataset(data_path)
    calls: list[Path] = []
    monkeypatch.setattr(
        data_module,
        "generate_hidden_states",
        _successful_generator(service_path, calls),
    )
    dataset = _arrow_dataset(
        data_path,
        shared_path=shared_path,
        hidden_states_path=index_path,
        on_generate="cache",
    )

    assert dataset._maybe_generate_hs(0) is not None

    indexed = index_path / "hs_0.safetensors"
    assert indexed.is_file()
    assert load_file(indexed)["token_ids"].tolist() == [1, 2, 3]
    assert len(calls) == 1


def test_train_and_validation_loaders_share_artifact_configuration(monkeypatch):
    dataset_kwargs = []

    def fake_arrow_dataset(**kwargs):
        dataset_kwargs.append(kwargs)
        return object()

    monkeypatch.setattr(dataloader_module, "ArrowDataset", fake_arrow_dataset)
    monkeypatch.setattr(
        dataloader_module,
        "_setup_dataloader",
        lambda dataset, *_args, **_kwargs: dataset,
    )

    dataloader_module.create_train_val_loaders(
        data_path="data",
        train_data_ratio=0.9,
        total_seq_len=128,
        hidden_states_dtype=torch.bfloat16,
        noise_std=0.0,
        legacy_data=False,
        hidden_states_path=None,
        vllm_endpoint="http://producer/v1",
        on_missing="generate",
        on_generate="delete",
        verifier_name_or_path="model",
        request_timeout=10,
        max_retries=2,
        hidden_size=4,
        num_target_layers=3,
        num_workers=0,
        prefetch_factor=1,
        preprocess=None,
        shared_artifacts_path="shared",
        shared_artifacts_namespace="layers:2,18,33",
        shared_artifacts_ttl_seconds=None,
        shared_artifacts_lock_timeout_seconds=45,
    )

    assert len(dataset_kwargs) == 2
    for kwargs in dataset_kwargs:
        assert kwargs["shared_artifacts_path"] == "shared"
        assert kwargs["shared_artifacts_namespace"] == "layers:2,18,33"
        assert kwargs["shared_artifacts_ttl_seconds"] is None
        assert kwargs["shared_artifacts_lock_timeout_seconds"] == 45
