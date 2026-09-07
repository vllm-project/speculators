"""Tests for training CLI vocabulary-mapping resolution and caching."""

import argparse
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import torch

from speculators.train import cli as train_cli

DRAFT_VOCAB_SIZE = 32000


def _args(data_path: Path, draft_vocab_size: int | None) -> argparse.Namespace:
    return argparse.Namespace(
        d2t_path=None,
        t2d_path=None,
        data_path=str(data_path),
        token_freq_path=None,
        draft_vocab_size=draft_vocab_size,
        verifier_name_or_path="dummy-verifier",
        trust_remote_code=False,
    )


def _write_mappings(directory: Path, stem: str, draft_vocab_size: int) -> None:
    np.save(directory / f"d2t{stem}.npy", np.zeros(draft_vocab_size, dtype=np.int64))
    np.save(directory / f"t2d{stem}.npy", np.zeros(16, dtype=np.bool_))


def test_load_mappings_accepts_matching_draft_vocab_size(tmp_path: Path):
    _write_mappings(tmp_path, "", DRAFT_VOCAB_SIZE)

    d2t, t2d, draft_vocab_size = train_cli._load_mappings(
        tmp_path / "d2t.npy",
        tmp_path / "t2d.npy",
        DRAFT_VOCAB_SIZE,
    )

    assert d2t.shape == (DRAFT_VOCAB_SIZE,)
    assert t2d.shape == (16,)
    assert draft_vocab_size == DRAFT_VOCAB_SIZE


def test_load_mappings_reports_actual_and_requested_sizes(tmp_path: Path):
    _write_mappings(tmp_path, "", DRAFT_VOCAB_SIZE)

    with pytest.raises(
        ValueError,
        match=(
            r"Vocab mapping d2t has size 32000, but "
            r"--draft-vocab-size requires 151936\."
        ),
    ):
        train_cli._load_mappings(
            tmp_path / "d2t.npy",
            tmp_path / "t2d.npy",
            151936,
        )


def test_parse_vocab_mappings_uses_size_keyed_default_files(tmp_path: Path):
    _write_mappings(tmp_path, f"-{DRAFT_VOCAB_SIZE}", DRAFT_VOCAB_SIZE)
    _write_mappings(tmp_path, "", 151936)

    d2t, t2d, draft_vocab_size = train_cli.parse_vocab_mappings(
        _args(tmp_path, DRAFT_VOCAB_SIZE)
    )

    assert d2t.shape == (DRAFT_VOCAB_SIZE,)
    assert t2d.shape == (16,)
    assert draft_vocab_size == DRAFT_VOCAB_SIZE


def test_parse_vocab_mappings_ignores_legacy_default_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    _write_mappings(tmp_path, "", DRAFT_VOCAB_SIZE)
    monkeypatch.setattr(
        train_cli.AutoConfig,
        "from_pretrained",
        lambda *_args, **_kwargs: SimpleNamespace(vocab_size=151936),
    )

    d2t, t2d, draft_vocab_size = train_cli.parse_vocab_mappings(
        _args(tmp_path, DRAFT_VOCAB_SIZE)
    )

    assert d2t is None
    assert t2d is None
    assert draft_vocab_size == 151936


def test_parse_vocab_mappings_caches_generated_files_by_size(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    torch.save({0: 3, 1: 2, 2: 1}, tmp_path / "token_freq.pt")
    monkeypatch.setattr(train_cli, "get_target_vocab_size", lambda *_args, **_kwargs: 8)

    d2t, t2d, draft_vocab_size = train_cli.parse_vocab_mappings(_args(tmp_path, 4))

    assert d2t.shape == (4,)
    assert t2d.shape == (8,)
    assert draft_vocab_size == 4
    assert (tmp_path / "d2t-4.npy").exists()
    assert (tmp_path / "t2d-4.npy").exists()
    assert not (tmp_path / "d2t.npy").exists()
    assert not (tmp_path / "t2d.npy").exists()


def test_parse_vocab_mappings_rank_zero_writes_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    torch.save({0: 3, 1: 2, 2: 1}, tmp_path / "token_freq.pt")
    monkeypatch.setattr(train_cli, "get_target_vocab_size", lambda *_args, **_kwargs: 8)
    monkeypatch.setattr(train_cli, "get_rank", lambda: 0)
    monkeypatch.setattr(train_cli, "is_distributed", lambda: True)
    broadcast = Mock()
    monkeypatch.setattr(train_cli.dist, "broadcast_object_list", broadcast)

    d2t, t2d, draft_vocab_size = train_cli.parse_vocab_mappings(_args(tmp_path, 4))

    assert d2t.shape == (4,)
    assert t2d.shape == (8,)
    assert draft_vocab_size == 4
    assert (tmp_path / "d2t-4.npy").exists()
    assert (tmp_path / "t2d-4.npy").exists()
    broadcast.assert_called_once()


def test_parse_vocab_mappings_broadcasts_to_nonzero_rank(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr(train_cli, "get_rank", lambda: 1)
    monkeypatch.setattr(train_cli, "is_distributed", lambda: True)

    expected = (
        torch.arange(4, dtype=torch.long),
        torch.ones(8, dtype=torch.bool),
        4,
    )

    def broadcast(payload, src):
        assert src == 0
        payload[0] = expected

    broadcast_mock = Mock(side_effect=broadcast)
    monkeypatch.setattr(train_cli.dist, "broadcast_object_list", broadcast_mock)

    d2t, t2d, draft_vocab_size = train_cli.parse_vocab_mappings(_args(tmp_path, 4))

    assert torch.equal(d2t, expected[0])
    assert torch.equal(t2d, expected[1])
    assert draft_vocab_size == 4
    assert not (tmp_path / "d2t-4.npy").exists()
    assert not (tmp_path / "t2d-4.npy").exists()
    broadcast_mock.assert_called_once()
