"""Unit tests for P-EAGLE COD sampling logic."""

import torch

from speculators.models.peagle.data import generate_cod_sample_indices


class TestMaxContextWindow:
    def test_applied_when_anchors_within_limit(self):
        """max_context_window caps the window even when anchors <= max_anchors."""
        seq_length = 256
        loss_mask = torch.zeros(1, seq_length)
        # Place 4 valid positions spread far apart (indices 10, 80, 160, 240)
        valid_positions = [10, 80, 160, 240]
        for pos in valid_positions:
            loss_mask[0, pos] = 1

        anchor_pos, depth = generate_cod_sample_indices(
            seq_length=seq_length,
            loss_mask=loss_mask,
            max_anchors=8,
            max_context_window=64,
        )

        depth_0_mask = depth == 0
        depth_0_positions = anchor_pos[depth_0_mask]
        window_size = depth_0_positions.shape[0]
        assert window_size <= 64

    def test_applied_when_anchors_exceed_limit(self):
        """max_context_window caps the window when anchors > max_anchors."""
        seq_length = 512
        loss_mask = torch.ones(1, seq_length)

        anchor_pos, depth = generate_cod_sample_indices(
            seq_length=seq_length,
            loss_mask=loss_mask,
            max_anchors=16,
            max_context_window=32,
        )

        depth_0_mask = depth == 0
        depth_0_positions = anchor_pos[depth_0_mask]
        window_size = depth_0_positions.shape[0]
        assert window_size <= 32

    def test_no_windowing_without_max_anchors(self):
        """Without max_anchors, full sequence is used."""
        seq_length = 128
        loss_mask = torch.ones(1, seq_length)

        anchor_pos, depth = generate_cod_sample_indices(
            seq_length=seq_length,
            loss_mask=loss_mask,
            max_anchors=None,
        )

        depth_0_mask = depth == 0
        assert anchor_pos[depth_0_mask].shape[0] == seq_length
