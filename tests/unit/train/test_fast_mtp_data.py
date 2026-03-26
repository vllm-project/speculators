"""Unit tests for FastMTP dataset utilities."""

from pathlib import Path

import pytest
import torch

from speculators.train.fast_mtp_data import (
    FastMTPSampleFileDataset,
    _shift_batch_fastmtp,
    make_fast_mtp_dataloader,
)

# ---------------------------------------------------------------------------
# _shift_batch_fastmtp
# ---------------------------------------------------------------------------


def _make_tensors(
    seq_len: int,
    hidden_size: int = 16,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    input_ids = torch.arange(seq_len, dtype=torch.long)
    hidden_states = torch.randn(seq_len, hidden_size)
    loss_mask = torch.ones(seq_len, dtype=torch.long)
    return input_ids, hidden_states, loss_mask


def test_shift_input_ids_by_one_position() -> None:
    ids, hs, mask = _make_tensors(10)
    result = _shift_batch_fastmtp(ids, hs, mask)
    assert torch.equal(result["input_ids"], ids[1:])


def test_shift_hidden_states_truncated_by_one() -> None:
    ids, hs, mask = _make_tensors(10)
    result = _shift_batch_fastmtp(ids, hs, mask)
    assert torch.equal(result["hidden_states"], hs[:-1])


def test_shift_loss_mask_by_one_position() -> None:
    ids, hs, mask = _make_tensors(10)
    result = _shift_batch_fastmtp(ids, hs, mask)
    assert torch.equal(result["loss_mask"], mask[1:])


def test_shift_output_seq_len_is_input_minus_one() -> None:
    seq_len = 10
    ids, hs, mask = _make_tensors(seq_len)
    result = _shift_batch_fastmtp(ids, hs, mask)
    assert result["input_ids"].shape[0] == seq_len - 1
    assert result["hidden_states"].shape[0] == seq_len - 1
    assert result["loss_mask"].shape[0] == seq_len - 1


def test_shift_lengths_is_seq_len_minus_one() -> None:
    ids, hs, mask = _make_tensors(10)
    result = _shift_batch_fastmtp(ids, hs, mask)
    assert result["lengths"].item() == 9


def test_shift_position_ids_correct() -> None:
    ids, hs, mask = _make_tensors(10)
    result = _shift_batch_fastmtp(ids, hs, mask)
    expected = torch.arange(9, dtype=torch.long)
    assert torch.equal(result["position_ids"], expected)


# ---------------------------------------------------------------------------
# FastMTPSampleFileDataset
# ---------------------------------------------------------------------------


def _save_sample(path: Path, seq_len: int = 10, hidden_size: int = 16) -> None:
    data = {
        "input_ids": torch.arange(seq_len, dtype=torch.long),
        "hidden_states": torch.randn(seq_len, hidden_size),
        "loss_mask": torch.ones(seq_len, dtype=torch.long),
    }
    torch.save(data, path)


def test_dataset_loads_pt_file(tmp_path: Path) -> None:
    sample_path = tmp_path / "data_0.pt"
    _save_sample(sample_path)

    ds = FastMTPSampleFileDataset(max_len=64, file_list=[str(sample_path)])
    assert len(ds) == 1
    item = ds[0]
    assert "input_ids" in item
    assert "hidden_states" in item
    assert "loss_mask" in item
    assert "lengths" in item
    assert "position_ids" in item


def test_dataset_applies_shift(tmp_path: Path) -> None:
    seq_len = 10
    sample_path = tmp_path / "data_0.pt"
    _save_sample(sample_path, seq_len=seq_len)

    ds = FastMTPSampleFileDataset(max_len=64, file_list=[str(sample_path)])
    item = ds[0]
    # After shift, seq_len decreases by 1
    assert item["input_ids"].shape[0] == seq_len - 1
    assert item["hidden_states"].shape[0] == seq_len - 1


def test_dataset_missing_key_raises(tmp_path: Path) -> None:
    path = tmp_path / "data_0.pt"
    # Missing loss_mask
    torch.save(
        {"input_ids": torch.arange(5), "hidden_states": torch.randn(5, 16)}, path
    )

    ds = FastMTPSampleFileDataset(max_len=64, file_list=[str(path)])
    with pytest.raises(KeyError, match="loss_mask"):
        ds[0]


def test_dataset_corrupted_file_raises(tmp_path: Path) -> None:
    path = tmp_path / "data_0.pt"
    path.write_bytes(b"not a valid pt file")

    ds = FastMTPSampleFileDataset(max_len=64, file_list=[str(path)])
    with pytest.raises(RuntimeError, match="Failed to load sample"):
        ds[0]


def test_dataset_dtype_casting(tmp_path: Path) -> None:
    path = tmp_path / "data_0.pt"
    _save_sample(path)

    ds = FastMTPSampleFileDataset(
        max_len=64, file_list=[str(path)], hidden_states_dtype=torch.bfloat16
    )
    item = ds[0]
    assert item["hidden_states"].dtype == torch.bfloat16


def test_dataset_datapath_vs_file_list_mutual_exclusive(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="not both"):
        FastMTPSampleFileDataset(
            max_len=64,
            datapath=str(tmp_path),
            file_list=["dummy.pt"],
        )


def test_dataset_neither_datapath_nor_file_list_raises() -> None:
    with pytest.raises(ValueError, match="Either datapath or file_list"):
        FastMTPSampleFileDataset(max_len=64)


# ---------------------------------------------------------------------------
# make_fast_mtp_dataloader
# ---------------------------------------------------------------------------


def _write_samples(data_dir: Path, n: int, seq_len: int = 10) -> None:
    for i in range(n):
        _save_sample(data_dir / f"data_{i:04d}.pt", seq_len=seq_len)


def test_make_dataloader_returns_two_loaders(tmp_path: Path) -> None:
    _write_samples(tmp_path, n=10)
    train_loader, val_loader = make_fast_mtp_dataloader(
        data_dir=tmp_path, max_len=64, batch_size=2, seed=42
    )
    assert train_loader is not None
    assert val_loader is not None


def test_make_dataloader_train_val_split(tmp_path: Path) -> None:
    _write_samples(tmp_path, n=10)
    train_loader, val_loader = make_fast_mtp_dataloader(
        data_dir=tmp_path, max_len=64, batch_size=1, train_ratio=0.8, seed=0
    )
    assert len(train_loader.dataset) == 8  # type: ignore[arg-type]
    assert len(val_loader.dataset) == 2  # type: ignore[arg-type]


def test_make_dataloader_seed_reproducible(tmp_path: Path) -> None:
    _write_samples(tmp_path, n=10)
    loader_a, _ = make_fast_mtp_dataloader(
        data_dir=tmp_path, max_len=64, batch_size=1, seed=42
    )
    loader_b, _ = make_fast_mtp_dataloader(
        data_dir=tmp_path, max_len=64, batch_size=1, seed=42
    )
    files_a = loader_a.dataset.file_paths  # type: ignore[attr-defined]
    files_b = loader_b.dataset.file_paths  # type: ignore[attr-defined]
    assert files_a == files_b


def test_make_dataloader_different_seeds_differ(tmp_path: Path) -> None:
    _write_samples(tmp_path, n=20)
    loader_a, _ = make_fast_mtp_dataloader(
        data_dir=tmp_path, max_len=64, batch_size=1, seed=0
    )
    loader_b, _ = make_fast_mtp_dataloader(
        data_dir=tmp_path, max_len=64, batch_size=1, seed=99
    )
    files_a = loader_a.dataset.file_paths  # type: ignore[attr-defined]
    files_b = loader_b.dataset.file_paths  # type: ignore[attr-defined]
    assert files_a != files_b
