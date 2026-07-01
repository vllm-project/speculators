"""Unit tests for the EAGLE3 attention masks."""

import pytest
import torch
from torch.nn.attention.flex_attention import BlockMask, create_mask

from speculators.models.eagle3.attention import (
    create_combined_mask_mod,
    extend_dense_mask_for_draft_tokens,
    extend_mask_for_draft_tokens,
)


def test_create_combined_mask_mod():
    lengths = torch.tensor([1, 2, 3])
    document_ids = torch.repeat_interleave(
        torch.arange(lengths.shape[0], dtype=torch.long), lengths
    )
    mask_mod = create_combined_mask_mod(
        document_ids, total_seq_len=int(lengths.sum().item())
    )

    # Creates causal document mask mod that supports extended diagonals
    # lengths -> document ids [0, 1, 1, 2, 2, 2]
    # Expected mask mod values for q_idx (row), kv_idx (column):
    expected_mask_mod = [
        [1, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0],
        [0, 1, 1, 0, 0, 0],
        [0, 0, 0, 1, 0, 0],
        [0, 0, 0, 1, 1, 0],
        [0, 0, 0, 1, 1, 1],
    ]
    t0 = torch.tensor(0)

    for q_idx in range(len(expected_mask_mod)):
        for kv_idx in range(len(expected_mask_mod[q_idx])):
            assert mask_mod(t0, t0, q_idx, kv_idx) == expected_mask_mod[q_idx][kv_idx]


@pytest.mark.parametrize(
    "lengths", [torch.tensor([1, 2, 3]), torch.tensor([2, 2, 2]), torch.tensor([5])]
)
def test_diagonal_draft_tokens_mask_mod(lengths):
    # Causal  Diagonal
    # ⌄ ⌄ ⌄ | ⌄ ⌄ ⌄ ⌄ ⌄ ⌄
    # 1 0 0 | 1 0 0 1 0 0
    # 1 1 0 | 0 1 0 0 1 0
    # 1 1 1 | 0 0 1 0 0 1
    # If kv_idx > N (N = orig seq len = num query inds), only the diagonal tokens are
    # in the mask. Diagonal tokens are those where kv_idx % N == q_idx

    document_ids = torch.repeat_interleave(
        torch.arange(lengths.shape[0], dtype=torch.long), lengths
    )
    mask_mod = create_combined_mask_mod(
        document_ids, total_seq_len=lengths.sum().item()
    )

    N = lengths.sum().item()

    t0 = torch.tensor(0)
    for q_idx in range(N):
        for kv_idx in range(N, 3 * N):
            assert mask_mod(t0, t0, q_idx, kv_idx) == (kv_idx % N == q_idx)


@pytest.mark.parametrize(
    ("kv_num_blocks", "kv_indices", "expected_kv_indices"),
    [
        # Test 1: Dense matrix shown in comments in test code
        (
            torch.tensor([2, 2, 1]),
            torch.tensor([[0, 2, -1], [0, 1, -1], [1, -1, -1]]),
            torch.tensor([[0, 2, 3], [0, 1, 4], [1, 5, -1]]),
        ),
        # Test 2: Dense matrix below
        # 0 1 1 0
        # 1 0 1 1
        # 1 0 0 1
        # 1 1 1 1
        (
            torch.tensor([2, 3, 2, 4]),
            torch.tensor([[1, 2, -1, -1], [0, 2, 3, -1], [0, 3, -1, -1], [0, 1, 2, 3]]),
            torch.tensor(
                [
                    [1, 2, 4, -1, -1],
                    [0, 2, 3, 5, -1],
                    [0, 3, 6, -1, -1],
                    [0, 1, 2, 3, 7],
                ]
            ),
        ),
    ],
)
def test_extend_mask_for_draft_tokens(kv_num_blocks, kv_indices, expected_kv_indices):
    kv_num_blocks = kv_num_blocks.reshape(1, 1, *kv_num_blocks.shape)
    kv_indices = kv_indices.reshape(1, 1, *kv_indices.shape)
    expected_kv_indices = expected_kv_indices.reshape(1, 1, *expected_kv_indices.shape)

    def dummy_mask_mod(b, h, q_idx, kv_idx):
        return True

    block_mask = BlockMask.from_kv_blocks(
        kv_num_blocks=kv_num_blocks.clone(),
        kv_indices=kv_indices.clone(),
        mask_mod=dummy_mask_mod,
    )

    extended_mask = extend_mask_for_draft_tokens(block_mask)

    for q_idx in range(kv_num_blocks.shape[2]):
        num_defined_blocks_in_row = extended_mask.kv_num_blocks[0, 0, q_idx].item()
        assert torch.equal(
            extended_mask.kv_indices[0, 0, q_idx, :num_defined_blocks_in_row],
            expected_kv_indices[0, 0, q_idx, :num_defined_blocks_in_row],
        )

    assert extended_mask.mask_mod == block_mask.mask_mod


def test_extend_dense_mask_matches_block_mask():
    """extend_dense_mask_for_draft_tokens preserves the original mask and
    appends a diagonal identity block at each step."""
    total_seq_len = 8
    document_ids = torch.zeros(total_seq_len, dtype=torch.long)
    mask_mod = create_combined_mask_mod(document_ids, total_seq_len)

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
