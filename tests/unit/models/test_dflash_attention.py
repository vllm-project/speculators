"""Unit tests for the DFlash anchor-block attention mask."""

import pytest
import torch
from torch.nn.attention.flex_attention import create_mask

from speculators.models.dflash.attention import create_anchor_block_mask_mod


def _reference_dense_from_mask_mod(
    lengths,
    total_seq_len,
    anchor_positions,
    block_size,
    sliding_window=None,
    sliding_window_non_causal=False,
):
    """Ground truth: evaluate the flex mask_mod element-wise over the q x kv grid."""
    mask_mod, q_len, kv_len = create_anchor_block_mask_mod(
        lengths=lengths,
        total_seq_len=total_seq_len,
        anchor_positions=anchor_positions,
        block_size=block_size,
        sliding_window=sliding_window,
        sliding_window_non_causal=sliding_window_non_causal,
    )
    zero = torch.zeros((), dtype=torch.long)
    ref = torch.zeros(q_len, kv_len, dtype=torch.bool)
    for q in range(q_len):
        for kv in range(kv_len):
            ref[q, kv] = bool(mask_mod(zero, zero, torch.tensor(q), torch.tensor(kv)))
    return ref


def _dense_from_create_mask(
    lengths,
    total_seq_len,
    anchor_positions,
    block_size,
    sliding_window=None,
    sliding_window_non_causal=False,
):
    mask_mod, q_len, kv_len = create_anchor_block_mask_mod(
        lengths=lengths,
        total_seq_len=total_seq_len,
        anchor_positions=anchor_positions,
        block_size=block_size,
        sliding_window=sliding_window,
        sliding_window_non_causal=sliding_window_non_causal,
    )
    return create_mask(
        mask_mod,
        B=None,
        H=None,
        Q_LEN=q_len,
        KV_LEN=kv_len,
        device=lengths.device,
    )


def test_create_mask_matches_mask_mod_full_attention():
    """Dense mask equals the mask_mod for the full-attention case."""
    device = torch.device("cpu")
    total_seq_len, block_size = 16, 4
    lengths = torch.tensor([10, 6])  # two packed documents summing to total_seq_len
    anchor_positions = torch.tensor([3, 8, 12])

    ref = _reference_dense_from_mask_mod(
        lengths, total_seq_len, anchor_positions, block_size
    )
    dense = _dense_from_create_mask(
        lengths.to(device), total_seq_len, anchor_positions, block_size
    )

    assert dense.shape == (1, 1, ref.shape[0], ref.shape[1])
    assert torch.equal(dense[0, 0].bool(), ref)


def test_create_mask_matches_mask_mod_sliding_window():
    """Dense mask equals the mask_mod when a sliding window is set."""
    device = torch.device("cpu")
    total_seq_len, block_size = 16, 4
    lengths = torch.tensor([16])  # single document
    anchor_positions = torch.tensor([5, 9, 14])
    sliding_window = 4

    ref = _reference_dense_from_mask_mod(
        lengths,
        total_seq_len,
        anchor_positions,
        block_size,
        sliding_window=sliding_window,
    )
    dense = _dense_from_create_mask(
        lengths.to(device),
        total_seq_len,
        anchor_positions,
        block_size,
        sliding_window=sliding_window,
    )

    assert torch.equal(dense[0, 0].bool(), ref)


def test_create_mask_each_query_sees_its_own_block():
    """Every query must attend to at least its own synthetic block."""
    device = torch.device("cpu")
    total_seq_len, block_size = 12, 4
    lengths = torch.tensor([12])
    anchor_positions = torch.tensor([2, 7, 10])

    dense = _dense_from_create_mask(
        lengths.to(device), total_seq_len, anchor_positions, block_size
    )

    assert bool(dense[0, 0].any(dim=-1).all())


def test_mask_mod_rejects_lengths_overflow():
    """sum(lengths) > total_seq_len raises."""
    with pytest.raises(ValueError, match="exceeds total_seq_len"):
        create_anchor_block_mask_mod(
            lengths=torch.tensor([10, 10]),  # sum = 20 > total_seq_len
            total_seq_len=16,
            anchor_positions=torch.tensor([3, 8]),
            block_size=4,
        )


def test_mask_mod_rejects_out_of_range_anchor():
    """anchor_positions outside [0, total_seq_len) raises."""
    with pytest.raises(ValueError, match="out of range"):
        create_anchor_block_mask_mod(
            lengths=torch.tensor([16]),
            total_seq_len=16,
            anchor_positions=torch.tensor([3, 20]),  # 20 >= total_seq_len
            block_size=4,
        )
