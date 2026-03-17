"""Unit tests for FastMTP dataset utilities (train/fast_mtp_data.py)."""

from pathlib import Path

import pytest
import torch

from speculators.train.fast_mtp_data import (
    FastMTPSampleFileDataset,
    _shift_batch_fastmtp,
    make_fast_mtp_dataloader,
)

H = 8
SEQ_LEN = 10
MAX_LEN = 8


# ── Helpers ───────────────────────────────────────────────────────────────────


def _make_raw(seq_len: int = SEQ_LEN, hidden_size: int = H) -> dict:
    return {
        "input_ids": torch.arange(seq_len, dtype=torch.long),
        "hidden_states": torch.randn(seq_len, hidden_size),
        "loss_mask": torch.ones(seq_len, dtype=torch.long),
    }


def _save_pt(path: Path, seq_len: int = SEQ_LEN, hidden_size: int = H) -> Path:
    pt_path = path / f"data_{seq_len}.pt"
    torch.save(_make_raw(seq_len=seq_len, hidden_size=hidden_size), str(pt_path))
    return pt_path


# ── _shift_batch_fastmtp ──────────────────────────────────────────────────────


class TestShiftBatch:
    def _full_batch(self, seq_len: int = SEQ_LEN) -> dict:
        return {
            "input_ids": torch.arange(seq_len, dtype=torch.long),
            "hidden_states": torch.arange(seq_len * H, dtype=torch.float).reshape(
                seq_len, H
            ),
            "loss_mask": torch.ones(seq_len, dtype=torch.long),
            "lengths": torch.tensor([seq_len], dtype=torch.long),
            "position_ids": torch.arange(seq_len, dtype=torch.long),
        }

    def test_input_ids_shifted_by_one(self) -> None:
        batch = self._full_batch()
        shifted = _shift_batch_fastmtp(batch)
        assert torch.equal(shifted["input_ids"], batch["input_ids"][1:])

    def test_hidden_states_drops_last(self) -> None:
        batch = self._full_batch()
        shifted = _shift_batch_fastmtp(batch)
        assert torch.equal(shifted["hidden_states"], batch["hidden_states"][:-1])

    def test_alignment(self) -> None:
        """Semantic contract: after shift, input_ids[n] == x_{n+1} and
        hidden_states[n] == g_n, so the model at step 0 reads embed(x_1) with
        context g_0 to predict x_2.
        """
        batch = {
            "input_ids": torch.tensor([10, 20, 30, 40, 50], dtype=torch.long),
            "hidden_states": torch.arange(5 * H, dtype=torch.float).reshape(5, H),
            "loss_mask": torch.ones(5, dtype=torch.long),
            "lengths": torch.tensor([5], dtype=torch.long),
            "position_ids": torch.arange(5, dtype=torch.long),
        }
        shifted = _shift_batch_fastmtp(batch)
        assert shifted["input_ids"][0].item() == 20
        assert torch.equal(shifted["hidden_states"][0], batch["hidden_states"][0])


# ── FastMTPSampleFileDataset ───────────────────────────────────────────────────


@pytest.fixture
def ds(tmp_path: Path) -> FastMTPSampleFileDataset:
    """Dataset backed by a single SEQ_LEN=10 sample file."""
    return FastMTPSampleFileDataset(
        max_len=MAX_LEN, file_list=[str(_save_pt(tmp_path))]
    )


class TestFastMTPSampleFileDataset:
    def test_getitem_returns_expected_keys(self, ds: FastMTPSampleFileDataset) -> None:
        """Key contract for the trainer: exactly these keys, nothing else."""
        assert set(ds[0].keys()) == {
            "input_ids",
            "hidden_states",
            "loss_mask",
            "labels",
            "lengths",
            "position_ids",
        }

    def test_sequence_lengths_after_shift(self, ds: FastMTPSampleFileDataset) -> None:
        """Shift is applied: seq_len shrinks by 1; hidden_states has correct H dim."""
        sample = ds[0]
        assert sample["input_ids"].shape == (SEQ_LEN - 1,)
        assert sample["hidden_states"].shape == (SEQ_LEN - 1, H)

    def test_hidden_states_dtype_cast(self, tmp_path: Path) -> None:
        """hidden_states_dtype= is honoured; critical for bfloat16 memory savings."""
        ds = FastMTPSampleFileDataset(
            max_len=MAX_LEN,
            file_list=[str(_save_pt(tmp_path))],
            hidden_states_dtype=torch.bfloat16,
        )
        assert ds[0]["hidden_states"].dtype == torch.bfloat16

    def test_requires_datapath_or_file_list(self) -> None:
        with pytest.raises(ValueError):
            FastMTPSampleFileDataset(max_len=MAX_LEN)

    def test_raises_if_both_provided(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError):
            FastMTPSampleFileDataset(
                max_len=MAX_LEN,
                datapath=str(tmp_path),
                file_list=[str(tmp_path / "data_0.pt")],
            )

    def test_datapath_discovers_files(self, tmp_path: Path) -> None:
        for i in range(4):
            torch.save(_make_raw(seq_len=6), str(tmp_path / f"data_{i}.pt"))
        assert (
            len(FastMTPSampleFileDataset(max_len=MAX_LEN, datapath=str(tmp_path))) == 4
        )


# ── make_fast_mtp_dataloader ──────────────────────────────────────────────────


class TestMakeFastMtpDataloader:
    def _populate_dir(self, data_dir: Path, n: int = 10) -> None:
        for i in range(n):
            torch.save(_make_raw(seq_len=SEQ_LEN), str(data_dir / f"data_{i}.pt"))

    def test_train_val_split_sizes(self, tmp_path: Path) -> None:
        self._populate_dir(tmp_path, n=10)
        train_dl, val_dl = make_fast_mtp_dataloader(
            tmp_path, max_len=MAX_LEN, batch_size=1, train_ratio=0.9
        )
        assert len(train_dl.dataset) == 9  # type: ignore[arg-type]
        assert len(val_dl.dataset) == 1  # type: ignore[arg-type]

    def test_batch_shapes(self, tmp_path: Path) -> None:
        """End-to-end: batch from the DataLoader has correct ranks and H dim."""
        self._populate_dir(tmp_path, n=4)
        train_dl, _ = make_fast_mtp_dataloader(
            tmp_path, max_len=MAX_LEN, batch_size=2, train_ratio=0.75
        )
        batch = next(iter(train_dl))
        assert batch["input_ids"].ndim == 2
        assert batch["hidden_states"].ndim == 3
        assert batch["hidden_states"].shape[-1] == H

    def test_hidden_states_dtype(self, tmp_path: Path) -> None:
        """dtype flows from make_fast_mtp_dataloader all the way to the batch tensor."""
        self._populate_dir(tmp_path, n=4)
        train_dl, _ = make_fast_mtp_dataloader(
            tmp_path,
            max_len=MAX_LEN,
            batch_size=2,
            train_ratio=0.75,
            hidden_states_dtype=torch.bfloat16,
        )
        assert next(iter(train_dl))["hidden_states"].dtype == torch.bfloat16
