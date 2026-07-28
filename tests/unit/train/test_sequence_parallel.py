"""Unit tests for Ulysses sequence parallelism utilities."""

import pytest
import torch

from speculators.train.sequence_parallel import (
    maybe_replicate_kv_heads,
    split_batch_for_sp,
)


# ---------------------------------------------------------------------------
# split_batch_for_sp
# ---------------------------------------------------------------------------


class TestSplitBatchForSP:
    def _make_batch(self, seq_len: int = 16):
        return {
            "hidden_states": torch.randn(1, seq_len, 64),
            "verifier_last_hidden_states": torch.randn(1, seq_len, 32),
            "input_ids": torch.arange(seq_len).unsqueeze(0),
            "loss_mask": torch.ones(1, seq_len),
            "position_ids": torch.arange(seq_len).unsqueeze(0),
            "document_ids": torch.zeros(1, seq_len, dtype=torch.long),
        }

    def test_sp_size_1_passthrough(self):
        batch = self._make_batch()
        result = split_batch_for_sp(batch, sp_rank=0, sp_size=1)
        assert result is batch

    def test_split_keys_are_chunked(self):
        batch = self._make_batch(seq_len=16)
        result = split_batch_for_sp(batch, sp_rank=0, sp_size=2)

        for key in ("hidden_states", "verifier_last_hidden_states",
                     "input_ids", "loss_mask", "position_ids"):
            assert result[key].shape[1] == 8, f"{key} not split"

    def test_document_ids_kept_full(self):
        batch = self._make_batch(seq_len=16)
        result = split_batch_for_sp(batch, sp_rank=0, sp_size=2)
        assert result["document_ids"].shape[1] == 16

    def test_correct_chunk_per_rank(self):
        batch = self._make_batch(seq_len=16)
        r0 = split_batch_for_sp(batch, sp_rank=0, sp_size=2)
        r1 = split_batch_for_sp(batch, sp_rank=1, sp_size=2)

        torch.testing.assert_close(
            torch.cat([r0["input_ids"], r1["input_ids"]], dim=1),
            batch["input_ids"],
        )

    def test_full_seq_len_injected(self):
        batch = self._make_batch(seq_len=16)
        result = split_batch_for_sp(batch, sp_rank=0, sp_size=2)
        assert "full_seq_len" in result
        assert result["full_seq_len"].item() == 16

    def test_full_seq_len_not_injected_sp1(self):
        batch = self._make_batch(seq_len=16)
        result = split_batch_for_sp(batch, sp_rank=0, sp_size=1)
        assert "full_seq_len" not in result

    def test_four_way_split(self):
        batch = self._make_batch(seq_len=32)
        chunks = [split_batch_for_sp(batch, sp_rank=r, sp_size=4) for r in range(4)]
        for c in chunks:
            assert c["hidden_states"].shape[1] == 8
        reconstructed = torch.cat([c["input_ids"] for c in chunks], dim=1)
        torch.testing.assert_close(reconstructed, batch["input_ids"])

    def test_scalar_values_passed_through(self):
        batch = self._make_batch(seq_len=8)
        batch["some_scalar"] = 42
        result = split_batch_for_sp(batch, sp_rank=0, sp_size=2)
        assert result["some_scalar"] == 42

    def test_1d_tensor_not_split(self):
        batch = self._make_batch(seq_len=8)
        batch["some_1d"] = torch.tensor([1, 2, 3])
        result = split_batch_for_sp(batch, sp_rank=0, sp_size=2)
        torch.testing.assert_close(result["some_1d"], torch.tensor([1, 2, 3]))


# ---------------------------------------------------------------------------
# maybe_replicate_kv_heads
# ---------------------------------------------------------------------------


class TestMaybeReplicateKVHeads:
    def _make_kv(self, num_kv_heads: int, seq_len: int = 8, head_dim: int = 4):
        key = torch.randn(1, num_kv_heads, seq_len, head_dim)
        value = torch.randn(1, num_kv_heads, seq_len, head_dim)
        return key, value

    def test_no_replication_when_divisible(self):
        key, value = self._make_kv(num_kv_heads=8)
        k_out, v_out = maybe_replicate_kv_heads(key, value, sp_size=4)
        assert k_out.shape[1] == 8
        torch.testing.assert_close(k_out, key)
        torch.testing.assert_close(v_out, value)

    def test_no_replication_sp_size_1(self):
        key, value = self._make_kv(num_kv_heads=4)
        k_out, v_out = maybe_replicate_kv_heads(key, value, sp_size=1)
        assert k_out.shape[1] == 4
        torch.testing.assert_close(k_out, key)

    def test_replication_when_sp_greater_than_kv(self):
        key, value = self._make_kv(num_kv_heads=2)
        k_out, v_out = maybe_replicate_kv_heads(key, value, sp_size=8)
        assert k_out.shape[1] == 8
        # Each original head should be repeated 4 times
        for i in range(4):
            torch.testing.assert_close(k_out[:, i, :, :], key[:, 0, :, :])
        for i in range(4, 8):
            torch.testing.assert_close(k_out[:, i, :, :], key[:, 1, :, :])

    def test_error_kv_not_divisible_by_sp(self):
        key, value = self._make_kv(num_kv_heads=3)
        with pytest.raises(ValueError, match="must be divisible"):
            maybe_replicate_kv_heads(key, value, sp_size=2)

    def test_error_sp_not_divisible_by_kv(self):
        key, value = self._make_kv(num_kv_heads=3)
        with pytest.raises(ValueError, match="must be divisible"):
            maybe_replicate_kv_heads(key, value, sp_size=5)

    def test_sp_equals_kv_heads(self):
        key, value = self._make_kv(num_kv_heads=4)
        k_out, v_out = maybe_replicate_kv_heads(key, value, sp_size=4)
        assert k_out.shape[1] == 4
        torch.testing.assert_close(k_out, key)
