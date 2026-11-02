"""Unit tests for the DFlash anchor-block attention mask."""

import pytest
import torch
from torch.nn.attention.flex_attention import create_mask

from speculators.models.dflash.attention import create_anchor_block_mask_mod


def _lengths_to_document_ids(lengths, total_seq_len):
    document_ids = torch.full((total_seq_len,), -1, dtype=torch.long)
    document_ids[: lengths.sum()] = torch.repeat_interleave(
        torch.arange(lengths.shape[0], dtype=torch.long), lengths
    )
    return document_ids


def _reference_dense_from_mask_mod(
    document_ids,
    total_seq_len,
    anchor_positions,
    block_size,
    sliding_window=None,
    sliding_window_non_causal=False,
):
    """Ground truth: evaluate the flex mask_mod element-wise over the q x kv grid."""
    mask_mod, q_len, kv_len = create_anchor_block_mask_mod(
        document_ids=document_ids,
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
    document_ids,
    total_seq_len,
    anchor_positions,
    block_size,
    sliding_window=None,
    sliding_window_non_causal=False,
):
    mask_mod, q_len, kv_len = create_anchor_block_mask_mod(
        document_ids=document_ids,
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
        device=document_ids.device,
    )


def test_create_mask_matches_mask_mod_full_attention():
    """Dense mask equals the mask_mod for the full-attention case."""
    device = torch.device("cpu")
    total_seq_len, block_size = 16, 4
    lengths = torch.tensor([10, 6])  # two packed documents summing to total_seq_len
    document_ids = _lengths_to_document_ids(lengths, total_seq_len)
    anchor_positions = torch.tensor([3, 8, 12])

    ref = _reference_dense_from_mask_mod(
        document_ids, total_seq_len, anchor_positions, block_size
    )
    dense = _dense_from_create_mask(
        document_ids.to(device), total_seq_len, anchor_positions, block_size
    )

    assert dense.shape == (1, 1, ref.shape[0], ref.shape[1])
    assert torch.equal(dense[0, 0].bool(), ref)


def test_create_mask_matches_mask_mod_sliding_window():
    """Dense mask equals the mask_mod when a sliding window is set."""
    device = torch.device("cpu")
    total_seq_len, block_size = 16, 4
    lengths = torch.tensor([16])  # single document
    document_ids = _lengths_to_document_ids(lengths, total_seq_len)
    anchor_positions = torch.tensor([5, 9, 14])
    sliding_window = 4

    ref = _reference_dense_from_mask_mod(
        document_ids,
        total_seq_len,
        anchor_positions,
        block_size,
        sliding_window=sliding_window,
    )
    dense = _dense_from_create_mask(
        document_ids.to(device),
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
    document_ids = _lengths_to_document_ids(lengths, total_seq_len)
    anchor_positions = torch.tensor([2, 7, 10])

    dense = _dense_from_create_mask(
        document_ids.to(device), total_seq_len, anchor_positions, block_size
    )

    assert bool(dense[0, 0].any(dim=-1).all())


def test_block_causal_mask_hides_future_synthetic_positions():
    document_ids = torch.zeros(8, dtype=torch.long)
    anchors = torch.tensor([2])
    mask_mod, q_len, kv_len = create_anchor_block_mask_mod(
        document_ids,
        total_seq_len=8,
        anchor_positions=anchors,
        block_size=4,
        block_causal=True,
    )

    mask = torch.zeros(q_len, kv_len, dtype=torch.bool)
    index = torch.zeros((), dtype=torch.long)
    for query in range(q_len):
        for key in range(kv_len):
            mask[query, key] = mask_mod(
                index, index, torch.tensor(query), torch.tensor(key)
            )

    synthetic = mask[:, 8:]
    torch.testing.assert_close(synthetic, torch.ones(4, 4, dtype=torch.bool).tril())


@pytest.mark.parametrize("anchor", [2, 7, 12])
@pytest.mark.parametrize("window", [2, 4, 16])
def test_block_causal_sliding_window_matches_serving(anchor, window):
    """The prefix and synthetic keys share one per-query causal window."""
    total_seq_len, block_size = 20, 8
    document_ids = torch.zeros(total_seq_len, dtype=torch.long)
    mask_mod, q_len, kv_len = create_anchor_block_mask_mod(
        document_ids,
        total_seq_len=total_seq_len,
        anchor_positions=torch.tensor([anchor]),
        block_size=block_size,
        sliding_window=window,
        block_causal=True,
    )
    actual = create_mask(
        mask_mod, B=None, H=None, Q_LEN=q_len, KV_LEN=kv_len, device="cpu"
    )[0, 0]
    expected = torch.zeros_like(actual)
    for query in range(block_size):
        absolute_query = anchor + query
        for key in range(anchor):
            expected[query, key] = absolute_query - window < key <= absolute_query
        for key in range(block_size):
            expected[query, total_seq_len + key] = (
                absolute_query - window < anchor + key <= absolute_query
            )
    torch.testing.assert_close(actual, expected)
