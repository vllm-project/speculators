"""Unit tests for P-EAGLE COD sampling with max_anchors."""

import torch

from speculators.models.peagle.data import generate_cod_sample_indices


def _loss_mask(seq_length: int) -> torch.Tensor:
    return torch.ones(1, seq_length, dtype=torch.float32)


class TestMaxAnchors:
    def test_depth0_is_full_sequence(self):
        """Depth 0 should always be the full sequence regardless of max_anchors."""
        seq_len = 32
        anchor_pos, depth = generate_cod_sample_indices(
            seq_length=seq_len,
            loss_mask=_loss_mask(seq_len),
            num_depths=4,
            max_anchors=4,
        )
        depth0_positions = anchor_pos[depth == 0]
        assert depth0_positions.shape[0] == seq_len
        assert torch.equal(depth0_positions, torch.arange(seq_len))

    def test_max_anchors_caps_chains(self):
        """With max_anchors, depth-1+ chains should not exceed max_anchors."""
        seq_len = 64
        max_anchors = 8
        anchor_pos, depth = generate_cod_sample_indices(
            seq_length=seq_len,
            loss_mask=_loss_mask(seq_len),
            num_depths=4,
            max_anchors=max_anchors,
        )
        for d in range(1, 4):
            assert (depth == d).sum().item() <= max_anchors

    def test_max_anchors_preserves_full_depth0(self):
        """Depth 0 count should equal seq_length even with small max_anchors."""
        seq_len = 128
        anchor_pos, depth = generate_cod_sample_indices(
            seq_length=seq_len,
            loss_mask=_loss_mask(seq_len),
            num_depths=8,
            max_anchors=4,
        )
        assert (depth == 0).sum().item() == seq_len

    def test_max_anchors_none_uses_all(self):
        """max_anchors=None should use all valid positions (default behavior)."""
        seq_len = 32
        torch.manual_seed(42)
        anchor_pos_none, depth_none = generate_cod_sample_indices(
            seq_length=seq_len,
            loss_mask=_loss_mask(seq_len),
            num_depths=4,
            max_anchors=None,
        )
        torch.manual_seed(42)
        anchor_pos_default, depth_default = generate_cod_sample_indices(
            seq_length=seq_len,
            loss_mask=_loss_mask(seq_len),
            num_depths=4,
        )
        assert torch.equal(anchor_pos_none, anchor_pos_default)
        assert torch.equal(depth_none, depth_default)

    def test_max_anchors_fewer_valid_than_cap(self):
        """When valid positions < max_anchors, all valid positions are used."""
        seq_len = 16
        loss_mask = torch.zeros(1, seq_len)
        loss_mask[0, :5] = 1
        torch.manual_seed(42)
        anchor_pos_capped, depth_capped = generate_cod_sample_indices(
            seq_length=seq_len,
            loss_mask=loss_mask,
            num_depths=4,
            max_anchors=100,
        )
        torch.manual_seed(42)
        anchor_pos_none, depth_none = generate_cod_sample_indices(
            seq_length=seq_len,
            loss_mask=loss_mask,
            num_depths=4,
            max_anchors=None,
        )
        assert (depth_capped == 0).sum().item() == seq_len
        assert torch.equal(anchor_pos_capped, anchor_pos_none)
        assert torch.equal(depth_capped, depth_none)

    def test_max_anchors_sorted_order(self):
        """Subsampled anchors should be in sorted order for causal masking."""
        seq_len = 64
        anchor_pos, depth = generate_cod_sample_indices(
            seq_length=seq_len,
            loss_mask=_loss_mask(seq_len),
            num_depths=4,
            max_anchors=8,
        )
        for d in range(1, 4):
            d_anchors = anchor_pos[depth == d]
            if d_anchors.shape[0] > 1:
                assert torch.all(d_anchors[1:] >= d_anchors[:-1])
