"""Unit tests for P-EAGLE COD sampling logic."""

import torch

from speculators.models.peagle.data import generate_cod_sample_indices


class TestGenerateCodSampleIndices:
    def test_depth0_is_full_sequence(self):
        seq_len = 64
        loss_mask = torch.ones(1, seq_len)
        anchor_pos, depth = generate_cod_sample_indices(
            seq_length=seq_len, loss_mask=loss_mask, num_depths=4
        )
        depth0_anchors = anchor_pos[depth == 0]
        assert depth0_anchors.shape[0] == seq_len
        assert torch.equal(depth0_anchors, torch.arange(seq_len))

    def test_max_anchors_caps_chains(self):
        seq_len = 128
        loss_mask = torch.ones(1, seq_len)
        max_anchors = 16
        anchor_pos, depth = generate_cod_sample_indices(
            seq_length=seq_len,
            loss_mask=loss_mask,
            num_depths=4,
            max_anchors=max_anchors,
        )
        # Depth 0 must still be the full sequence
        assert anchor_pos[depth == 0].shape[0] == seq_len

        # Depth 1 chains should be capped at max_anchors
        depth1_count = (depth == 1).sum().item()
        assert depth1_count <= max_anchors

    def test_max_anchors_preserves_full_depth0(self):
        seq_len = 256
        loss_mask = torch.ones(1, seq_len)
        anchor_pos, depth = generate_cod_sample_indices(
            seq_length=seq_len,
            loss_mask=loss_mask,
            num_depths=4,
            max_anchors=8,
        )
        depth0_anchors = anchor_pos[depth == 0]
        assert depth0_anchors.shape[0] == seq_len
        assert torch.equal(depth0_anchors, torch.arange(seq_len))

    def test_max_anchors_none_uses_all(self):
        seq_len = 64
        loss_mask = torch.ones(1, seq_len)
        _, depth_limited = generate_cod_sample_indices(
            seq_length=seq_len,
            loss_mask=loss_mask,
            num_depths=4,
            max_anchors=None,
        )
        _, depth_default = generate_cod_sample_indices(
            seq_length=seq_len,
            loss_mask=loss_mask,
            num_depths=4,
        )
        # Both should produce depth-0 with full sequence
        assert (depth_limited == 0).sum() == (depth_default == 0).sum() == seq_len

    def test_max_anchors_fewer_valid_than_cap(self):
        seq_len = 64
        loss_mask = torch.zeros(1, seq_len)
        loss_mask[0, 10:20] = 1  # only 10 valid positions
        anchor_pos, depth = generate_cod_sample_indices(
            seq_length=seq_len,
            loss_mask=loss_mask,
            num_depths=4,
            max_anchors=32,
        )
        # Depth 0 still full sequence
        assert anchor_pos[depth == 0].shape[0] == seq_len
        # Chains should use all 10 valid positions (< max_anchors)
        depth1_count = (depth == 1).sum().item()
        assert depth1_count <= 10

    def test_max_anchors_sorted_order(self):
        seq_len = 128
        loss_mask = torch.ones(1, seq_len)
        anchor_pos, depth = generate_cod_sample_indices(
            seq_length=seq_len,
            loss_mask=loss_mask,
            num_depths=4,
            max_anchors=16,
        )
        for d in range(1, 4):
            d_anchors = anchor_pos[depth == d]
            if d_anchors.shape[0] > 1:
                diffs = d_anchors[1:] - d_anchors[:-1]
                assert (diffs >= 0).all(), f"Depth {d} anchors not sorted"
