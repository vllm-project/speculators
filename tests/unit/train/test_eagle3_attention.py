import pytest
from speculators.train.eagle3.attention import (
    create_combined_mask_mod,
    extend_mask_for_draft_tokens,
    flex_attention_forward,
)
from torch.nn.attention.flex_attention import BlockMask
import torch


def test_create_combined_mask_mod():
    lengths = torch.tensor([1, 2, 3])
    mask_mod = create_combined_mask_mod(lengths)

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
    # If kv_idx > N (N = original seq len = num query indices), only the diagonal tokens are in mask
    # Diagonal tokens are those where kv_idx % N == q_idx

    mask_mod = create_combined_mask_mod(lengths)

    N = lengths.sum().item()

    t0 = torch.tensor(0)
    for q_idx in range(N):
        for kv_idx in range(N, 3 * N):
            assert mask_mod(t0, t0, q_idx, kv_idx) == (kv_idx % N == q_idx)


@pytest.mark.parametrize(
    "kv_num_blocks, kv_indices, expected_kv_indices",
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
    # Block mask is stored in Block Compressed Sparse Row (BSRS) format
    # This means storing:
    # - kv_num_blocks (shape: [batch, head, q_blocks]): contains the number of blocks for each batch, head, and query block
    # - kv_indices (shape: [batch, head, q_blocks, kv_blocks]): contains the row indices of the blocks for each batch, head, and query block
    # Only the first kv_num_blocks of each row of kv_indices are defined
    # e.g. To store (ignoring batch and head dimensions):
    # 1 0 1
    # 1 1 0
    # 0 1 0
    # There are 2 blocks for the first query row (0, 2), 2 blocks for the second query row (0, 1), and 1 block for the third query row (1)
    # Therefore:
    # kv_num_blocks = [2, 2, 1]
    # kv_indices = [[[0, 2, U], [0, 1, U], [1, U, U]]] where U indicates the value is undefined
    # Note: for our masks currently batch and head indices aren't considered in the mask function, so we just treat them as 1 when storing the BlockMask

    # During ttt, we extend the mask to accomodate the new draft tokens. The tokens included will be those on the diagonal (see diagonal test above),
    # and therefore we need to include blocks on the newly added diagonal.

    # Therefore, we expect `kv_num_blocks` to increase by 1 for each query row because only the diagonal block will be added to each row.
    # We also expect `kv_indices` to include the new diagonal blocks for each query row.

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
        # Only the first num_defined_blocks_in_row of each row of kv_indices are defined, the rest can have any value
        # Check that the defined blocks are match expected values
        assert torch.equal(
            extended_mask.kv_indices[0, 0, q_idx, :num_defined_blocks_in_row],
            expected_kv_indices[0, 0, q_idx, :num_defined_blocks_in_row],
        )

    assert extended_mask.mask_mod == block_mask.mask_mod

