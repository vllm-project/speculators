"""Unit tests for get_base_indices_for_anchored_blocks."""

import pytest
import torch

from speculators.models.dflash.utils import get_base_indices_for_anchored_blocks


class TestGetBaseIndicesForAnchoredBlocks:
    def test_single_anchor(self):
        anchor_positions = torch.tensor([[3]])
        result = get_base_indices_for_anchored_blocks(anchor_positions, block_size=4)
        expected = torch.tensor([3, 4, 5, 6])
        assert torch.equal(result, expected)

    def test_multiple_anchors(self):
        anchor_positions = torch.tensor([[0, 5, 10]])
        result = get_base_indices_for_anchored_blocks(anchor_positions, block_size=3)
        expected = torch.tensor([0, 1, 2, 5, 6, 7, 10, 11, 12])
        assert torch.equal(result, expected)

    def test_block_size_one(self):
        anchor_positions = torch.tensor([[2, 7, 9]])
        result = get_base_indices_for_anchored_blocks(anchor_positions, block_size=1)
        expected = torch.tensor([2, 7, 9])
        assert torch.equal(result, expected)

    def test_1d_input(self):
        anchor_positions = torch.tensor([1, 4])
        result = get_base_indices_for_anchored_blocks(anchor_positions, block_size=2)
        expected = torch.tensor([1, 2, 4, 5])
        assert torch.equal(result, expected)

    def test_output_shape(self):
        num_anchors = 5
        block_size = 4
        anchor_positions = torch.tensor([[0, 3, 6, 9, 12]])
        result = get_base_indices_for_anchored_blocks(
            anchor_positions, block_size=block_size
        )
        assert result.shape == (num_anchors * block_size,)

    def test_output_dtype_is_long(self):
        anchor_positions = torch.tensor([[2.0, 5.0]])
        result = get_base_indices_for_anchored_blocks(anchor_positions, block_size=2)
        assert result.dtype == torch.long

    def test_total_seq_len_valid(self):
        anchor_positions = torch.tensor([[0, 3]])
        result = get_base_indices_for_anchored_blocks(
            anchor_positions, block_size=4, total_seq_len=7
        )
        expected = torch.tensor([0, 1, 2, 3, 3, 4, 5, 6])
        assert torch.equal(result, expected)

    def test_total_seq_len_out_of_range(self):
        anchor_positions = torch.tensor([[0, 3]])
        with pytest.raises(ValueError, match="out of range"):
            get_base_indices_for_anchored_blocks(
                anchor_positions, block_size=4, total_seq_len=6
            )

    def test_negative_anchor_raises(self):
        anchor_positions = torch.tensor([[-1, 3]])
        with pytest.raises(ValueError, match="out of range"):
            get_base_indices_for_anchored_blocks(anchor_positions, block_size=2)

    def test_no_total_seq_len_skips_upper_bound_check(self):
        anchor_positions = torch.tensor([[100]])
        result = get_base_indices_for_anchored_blocks(
            anchor_positions, block_size=3, total_seq_len=None
        )
        expected = torch.tensor([100, 101, 102])
        assert torch.equal(result, expected)
