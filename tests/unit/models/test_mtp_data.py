"""Unit tests for MTP data preprocessing."""

import torch

from speculators.models.mtp.data import shift_batch_mtp


def _make_batch(seq_len=16, hidden_size=64):
    return {
        "input_ids": torch.randint(0, 1000, (seq_len,)),
        "verifier_last_hidden_states": torch.randn(seq_len, hidden_size),
        "loss_mask": torch.ones(seq_len),
        "lengths": torch.tensor([seq_len]),
        "position_ids": torch.arange(seq_len),
    }


class TestShiftBatchMtp:
    def test_renames_verifier_hidden_states(self):
        batch = _make_batch()
        result = shift_batch_mtp(batch)
        assert "hidden_states" in result
        assert "verifier_last_hidden_states" not in result

    def test_hidden_states_match_verifier_input(self):
        batch = _make_batch()
        result = shift_batch_mtp(batch)
        assert torch.equal(
            result["hidden_states"], batch["verifier_last_hidden_states"]
        )

    def test_passes_through_required_fields(self):
        batch = _make_batch()
        result = shift_batch_mtp(batch)
        for key in ("input_ids", "loss_mask", "lengths", "position_ids"):
            assert key in result
            assert torch.equal(result[key], batch[key])

    def test_shapes_unchanged(self):
        seq_len, hidden_size = 32, 128
        batch = _make_batch(seq_len=seq_len, hidden_size=hidden_size)
        result = shift_batch_mtp(batch)
        assert result["input_ids"].shape == (seq_len,)
        assert result["hidden_states"].shape == (seq_len, hidden_size)
        assert result["loss_mask"].shape == (seq_len,)
        assert result["lengths"].shape == (1,)
        assert result["position_ids"].shape == (seq_len,)
