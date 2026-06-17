"""Unit tests for the EAGLE3 dense mask extension."""

import torch
from torch.nn.attention.flex_attention import create_mask

from speculators.models.eagle3.attention import (
    create_combined_mask_mod,
    extend_dense_mask_for_draft_tokens,
)


def test_extend_dense_mask_matches_block_mask():
    """extend_dense_mask_for_draft_tokens preserves the original mask and
    appends a diagonal identity block at each step."""
    total_seq_len = 8
    lengths = torch.tensor([8])
    mask_mod = create_combined_mask_mod(lengths, total_seq_len)

    dense_mask = create_mask(
        mask_mod,
        B=None,
        H=None,
        Q_LEN=total_seq_len,
        KV_LEN=total_seq_len,
        device="cpu",
    )
    original = dense_mask.clone()

    for step in range(3):
        dense_mask = extend_dense_mask_for_draft_tokens(dense_mask, total_seq_len)
        expected_kv = total_seq_len * (step + 2)
        assert dense_mask.shape == (1, 1, total_seq_len, expected_kv)

        assert torch.equal(dense_mask[..., :total_seq_len], original)

        start = total_seq_len * (step + 1)
        end = total_seq_len * (step + 2)
        new_block = dense_mask[0, 0, :, start:end]
        expected_diag = torch.eye(total_seq_len, dtype=torch.bool)
        assert torch.equal(new_block.bool(), expected_diag), (
            f"New block at step {step} is not diagonal"
        )
