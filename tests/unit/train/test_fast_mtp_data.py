"""Tests for FastMTP dataset utilities."""

from pathlib import Path

import pytest
import torch

from speculators.train.fast_mtp_data import (
    FastMTPSampleFileDataset,
    _shift_batch_fastmtp,
    make_fast_mtp_dataloader,
)


def _save_sample(path: Path, seq_len: int = 10, hidden_size: int = 16) -> None:
    torch.save(
        {
            "input_ids": torch.arange(seq_len, dtype=torch.long),
            "hidden_states": torch.randn(seq_len, hidden_size),
            "loss_mask": torch.ones(seq_len, dtype=torch.long),
        },
        path,
    )


# ---------------------------------------------------------------------------
# _shift_batch_fastmtp
# ---------------------------------------------------------------------------


def test_shift_batch_contract() -> None:
    seq_len = 10
    ids = torch.arange(seq_len, dtype=torch.long)
    hs = torch.randn(seq_len, 16)
    mask = torch.ones(seq_len, dtype=torch.long)

    out = _shift_batch_fastmtp(ids, hs, mask)

    assert torch.equal(out["input_ids"], ids[1:])
    assert torch.equal(out["hidden_states"], hs[:-1])
    assert torch.equal(out["loss_mask"], mask[1:])
    assert out["lengths"].item() == seq_len - 1
    assert torch.equal(out["position_ids"], torch.arange(seq_len - 1, dtype=torch.long))


# ---------------------------------------------------------------------------
# FastMTPSampleFileDataset
# ---------------------------------------------------------------------------


def test_dataset_loads_and_shifts(tmp_path: Path) -> None:
    seq_len = 10
    _save_sample(tmp_path / "data_0.pt", seq_len=seq_len)
    ds = FastMTPSampleFileDataset(max_len=64, file_list=[str(tmp_path / "data_0.pt")])

    item = ds[0]
    assert set(item.keys()) == {
        "input_ids",
        "hidden_states",
        "loss_mask",
        "lengths",
        "position_ids",
    }
    assert item["input_ids"].shape[0] == seq_len - 1


def test_dataset_dtype_cast(tmp_path: Path) -> None:
    _save_sample(tmp_path / "data_0.pt")
    ds = FastMTPSampleFileDataset(
        max_len=64,
        file_list=[str(tmp_path / "data_0.pt")],
        hidden_states_dtype=torch.bfloat16,
    )
    assert ds[0]["hidden_states"].dtype == torch.bfloat16


def test_dataset_missing_key_raises(tmp_path: Path) -> None:
    path = tmp_path / "data_0.pt"
    torch.save(
        {"input_ids": torch.arange(5), "hidden_states": torch.randn(5, 16)}, path
    )
    with pytest.raises(KeyError, match="loss_mask"):
        FastMTPSampleFileDataset(max_len=64, file_list=[str(path)])[0]


def test_dataset_corrupted_file_raises(tmp_path: Path) -> None:
    path = tmp_path / "data_0.pt"
    path.write_bytes(b"not a valid pt file")
    with pytest.raises(RuntimeError, match="Failed to load sample"):
        FastMTPSampleFileDataset(max_len=64, file_list=[str(path)])[0]


def test_dataset_mutual_exclusion_raises(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="not both"):
        FastMTPSampleFileDataset(
            max_len=64, datapath=str(tmp_path), file_list=["dummy.pt"]
        )


def test_dataset_no_source_raises() -> None:
    with pytest.raises(ValueError, match="Either datapath or file_list"):
        FastMTPSampleFileDataset(max_len=64)


# ---------------------------------------------------------------------------
# make_fast_mtp_dataloader
# ---------------------------------------------------------------------------


def _write_samples(data_dir: Path, n: int) -> None:
    for i in range(n):
        _save_sample(data_dir / f"data_{i:04d}.pt")


def test_dataloader_train_val_split(tmp_path: Path) -> None:
    _write_samples(tmp_path, n=10)
    train, val = make_fast_mtp_dataloader(
        data_dir=tmp_path,
        max_len=64,
        batch_size=1,
        hidden_size=16,
        train_ratio=0.8,
        seed=0,
    )
    assert len(train.dataset) == 8  # type: ignore[arg-type]
    assert len(val.dataset) == 2  # type: ignore[arg-type]


def test_dataloader_seed_determinism(tmp_path: Path) -> None:
    _write_samples(tmp_path, n=20)
    a, _ = make_fast_mtp_dataloader(
        data_dir=tmp_path, max_len=64, batch_size=1, hidden_size=16, seed=42
    )
    b, _ = make_fast_mtp_dataloader(
        data_dir=tmp_path, max_len=64, batch_size=1, hidden_size=16, seed=42
    )
    c, _ = make_fast_mtp_dataloader(
        data_dir=tmp_path, max_len=64, batch_size=1, hidden_size=16, seed=99
    )
    assert a.dataset.file_paths == b.dataset.file_paths  # type: ignore[attr-defined]
    assert a.dataset.file_paths != c.dataset.file_paths  # type: ignore[attr-defined]
