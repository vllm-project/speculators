"""Unit tests for get_base_indices_for_anchored_blocks and select_anchors."""

import torch

from speculators.models.dflash.utils import (
    get_base_indices_for_anchored_blocks,
    select_anchors,
)


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


class TestSelectAnchors:
    def test_every_document_gets_num_anchors_slots(self):
        torch.manual_seed(0)
        loss_mask = torch.ones(1, 64)
        document_ids = torch.tensor([[0] * 32 + [1] * 32])
        anchors, valid = select_anchors(
            loss_mask, document_ids, num_anchors=4, block_size=4
        )

        assert anchors.shape == valid.shape == (8,)
        assert valid.all()
        assert (anchors[:4] < 32).all()
        assert (anchors[4:] >= 32).all()
        for doc in (anchors[:4], anchors[4:]):
            assert torch.equal(doc, torch.sort(doc).values)
        # The last block_size positions are excluded so blocks stay in bounds.
        assert anchors.max() < 60

    def test_short_document_is_fully_covered_and_padded(self):
        torch.manual_seed(0)
        loss_mask = torch.zeros(1, 64)
        loss_mask[0, [3, 7, 9]] = 1  # document 0: only 3 supervised tokens
        loss_mask[0, 32:60] = 1  # document 1: 28 supervised tokens
        document_ids = torch.tensor([[0] * 16 + [1] * 48])
        anchors, valid = select_anchors(
            loss_mask, document_ids, num_anchors=8, block_size=4
        )

        assert anchors.shape == (16,)
        assert torch.equal(anchors[:3], torch.tensor([3, 7, 9]))
        assert torch.equal(valid[:8], torch.tensor([True] * 3 + [False] * 5))
        assert valid[8:].all()

    def test_only_supervised_positions_are_anchors(self):
        loss_mask = torch.zeros(1, 64)
        loss_mask[0, 8:16] = 1
        document_ids = torch.tensor([[0] * 32 + [-1] * 32])
        anchors, valid = select_anchors(
            loss_mask, document_ids, num_anchors=100, block_size=4
        )

        assert torch.equal(anchors[valid], torch.arange(8, 16))
        assert valid.sum() == 8

    def test_no_supervised_positions_yields_masked_slots(self):
        loss_mask = torch.zeros(1, 64)
        document_ids = torch.tensor([[0] * 32 + [-1] * 32])
        anchors, valid = select_anchors(
            loss_mask, document_ids, num_anchors=8, block_size=4
        )

        assert anchors.shape == valid.shape == (8,)
        assert not valid.any()
