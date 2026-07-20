"""Unit tests for MTP data preprocessing."""

import torch

from speculators.models.mtp.data import shift_batch_mtp


def test_shift_batch_mtp():
    seq_len, hidden_size = 32, 128
    batch = {
        "input_ids": torch.randint(0, 1000, (seq_len,)),
        "verifier_last_hidden_states": torch.randn(seq_len, hidden_size),
        "loss_mask": torch.ones(seq_len),
        "lengths": torch.tensor([seq_len]),
        "position_ids": torch.arange(seq_len),
    }

    result = shift_batch_mtp(batch)

    assert "verifier_last_hidden_states" not in result
    assert torch.equal(result["hidden_states"], batch["verifier_last_hidden_states"])
    assert result["hidden_states"].shape == (seq_len, hidden_size)

    for key in ("input_ids", "loss_mask", "lengths", "position_ids"):
        assert torch.equal(result[key], batch[key])


def test_shift_batch_mtp_forwards_verifier_kv():
    seq_len, hidden_size = 32, 128
    kv_dim = 64
    batch = {
        "input_ids": torch.randint(0, 1000, (seq_len,)),
        "verifier_last_hidden_states": torch.randn(seq_len, hidden_size),
        "loss_mask": torch.ones(seq_len),
        "lengths": torch.tensor([seq_len]),
        "position_ids": torch.arange(seq_len),
        "verifier_kv_last_local": torch.randn(seq_len, 2, kv_dim),
        "verifier_kv_last_global": torch.randn(seq_len, 2, kv_dim),
    }

    result = shift_batch_mtp(batch)

    assert torch.equal(result["verifier_kv_last_local"], batch["verifier_kv_last_local"])
    assert torch.equal(result["verifier_kv_last_global"], batch["verifier_kv_last_global"])


def test_shift_batch_mtp_without_kv():
    seq_len, hidden_size = 32, 128
    batch = {
        "input_ids": torch.randint(0, 1000, (seq_len,)),
        "verifier_last_hidden_states": torch.randn(seq_len, hidden_size),
        "loss_mask": torch.ones(seq_len),
        "lengths": torch.tensor([seq_len]),
        "position_ids": torch.arange(seq_len),
    }

    result = shift_batch_mtp(batch)

    assert "verifier_kv_last_local" not in result
    assert "verifier_kv_last_global" not in result
