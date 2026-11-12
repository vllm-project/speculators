"""Unit tests for DFlash anchor and position-ID utilities."""

import torch

from speculators.models.dflash.utils import (
    extend_position_ids_with_mask_tokens,
    select_anchors,
)


class TestExtendPositionIdsWithMaskTokens:
    def test_uses_anchor_position_ids(self):
        position_ids = torch.tensor([[10, 20, 30, 40]])
        anchor_positions = torch.tensor([1, 3])

        result = extend_position_ids_with_mask_tokens(
            position_ids, anchor_positions, block_size=2
        )

        expected = torch.tensor([[10, 20, 30, 40, 20, 21, 40, 41]])
        assert torch.equal(result, expected)

    def test_output_shape_and_dtype(self):
        position_ids = torch.arange(16, dtype=torch.long).unsqueeze(0)
        anchor_positions = torch.tensor([2, 7, 9])

        result = extend_position_ids_with_mask_tokens(
            position_ids, anchor_positions, block_size=3
        )

        assert result.shape == (1, 16 + 3 * 3)
        assert result.dtype == torch.long


class TestSelectAnchors:
    def test_sampled_anchors_are_sorted(self):
        # Anchors are returned sorted by position so the draft blocks form
        # contiguous flex-attention blocks (fast path) instead of scattered ones.
        torch.manual_seed(0)
        loss_mask = torch.ones(1, 64)

        anchors, block_positions, aligned_loss_mask = select_anchors(
            loss_mask, num_anchors=8, block_size=4, train_anchor=True
        )

        assert torch.equal(anchors, torch.sort(anchors).values)
        assert block_positions.shape == (8 * 4,)
        assert aligned_loss_mask.shape == (1, 8 * 4)

    def test_padded_anchor_blocks_are_masked(self):
        loss_mask = torch.ones(1, 8)

        _, block_positions, aligned_loss_mask = select_anchors(
            loss_mask, num_anchors=6, block_size=4, train_anchor=True
        )

        assert block_positions.shape == (6 * 4,)
        assert torch.all(aligned_loss_mask[:, : 4 * 4])
        assert not aligned_loss_mask[:, 4 * 4 :].any()

    def test_anchor_slot_is_masked_when_not_training_anchor(self):
        loss_mask = torch.ones(1, 32)

        _, _, aligned_loss_mask = select_anchors(
            loss_mask, num_anchors=4, block_size=4, train_anchor=False
        )

        assert not aligned_loss_mask[:, ::4].any()
