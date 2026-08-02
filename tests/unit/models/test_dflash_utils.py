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
    def test_sampled_anchors_are_sorted(self):
        # Anchors are returned sorted by position so the draft blocks form
        # contiguous flex-attention blocks (fast path) instead of scattered ones.
        torch.manual_seed(0)
        loss_mask = torch.ones(1, 64)
        anchors, anchor_valid = select_anchors(loss_mask, num_anchors=8, block_size=4)
        selected = anchors[anchor_valid]
        assert torch.equal(selected, torch.sort(selected).values)


class TestSelectAnchorsDocumentBoundaries:
    def test_blocks_never_cross_document_boundaries(self):
        torch.manual_seed(0)
        total, block_size = 32, 4
        loss_mask = torch.ones(1, total)
        document_ids = torch.zeros(1, total, dtype=torch.long)
        document_ids[:, 10:] = 1  # two packed documents: [0, 10) and [10, 32)

        anchors, anchor_valid = select_anchors(
            loss_mask,
            num_anchors=total,
            block_size=block_size,
            document_ids=document_ids,
        )
        selected = anchors[anchor_valid]
        assert selected.numel() > 0
        for pos in selected.tolist():
            block_docs = document_ids[0, pos : pos + block_size]
            assert torch.all(block_docs == block_docs[0]), (
                f"anchor {pos} spans documents {block_docs.tolist()}"
            )
        # Doc 0 allows anchors 0..6 (7..9 would spill into doc 1); doc 1 allows
        # 10..27 (the global last-block_size cut removes 28..31).
        assert anchor_valid.sum().item() == 7 + 18

    def test_document_ids_none_keeps_previous_behavior(self):
        torch.manual_seed(0)
        loss_mask = torch.ones(1, 16)
        anchors, anchor_valid = select_anchors(loss_mask, num_anchors=16, block_size=4)
        assert anchor_valid.sum().item() == 12  # only the tail cut applies
