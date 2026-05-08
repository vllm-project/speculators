"""Flex attention mask functions for P-EAGLE parallel group prediction."""

import torch


def create_peagle_mask_mod(
    all_indices: torch.Tensor,
    seq_length: int,
    sample_ids: torch.Tensor,
):
    """
    Create a flex attention mask modifier for P-EAGLE parallel groups.

    P-EAGLE uses COD (Conditional-On-Distribution) sampling to create parallel
    prediction groups. Each group (depth) has progressively fewer positions.

    Args:
        all_indices: Encoded COD sample indices (depth * seq_length + pos)
            [total_sampled]
        seq_length: Original sequence length (before parallel expansion)
        sample_ids: Sample IDs to prevent cross-sample attention
            [batch_size, seq_length]

    Returns:
        A mask_mod function compatible with flex_attention create_block_mask
    """
    sample_ids_flat = sample_ids.squeeze(0)

    def peagle_mask_mod(_b, _h, q_idx, kv_idx):
        q_full = all_indices[q_idx]
        kv_full = all_indices[kv_idx]

        q_pos = q_full % seq_length
        kv_pos = kv_full % seq_length
        q_depth = q_full // seq_length
        kv_depth = kv_full // seq_length

        hierarchical = q_depth >= kv_depth
        same_sample = sample_ids_flat[q_pos] == sample_ids_flat[kv_pos]
        same_depth = q_depth == kv_depth

        # Same-depth: full causal at depth 0, self-only at depth > 0
        is_depth_0 = (q_depth == 0) & (kv_depth == 0)
        same_depth_causal_depth0 = (q_pos >= kv_pos) & is_depth_0
        same_depth_diagonal = (q_idx == kv_idx) & (~is_depth_0)
        same_depth_mask = same_depth_causal_depth0 | same_depth_diagonal

        # Cross-depth: causal with offset to depth 0, parent-only to intermediate
        kv_is_depth_0 = kv_depth == 0
        kv_is_intermediate = (kv_depth > 0) & (kv_depth < q_depth)
        cross_depth_to_depth0 = kv_is_depth_0 & (kv_pos <= (q_pos - q_depth))
        parent_pos = q_pos - (q_depth - kv_depth)
        cross_depth_to_intermediate = kv_is_intermediate & (kv_pos == parent_pos)
        cross_depth_mask = cross_depth_to_depth0 | cross_depth_to_intermediate

        attention_allowed = (same_depth & same_depth_mask) | (
            (~same_depth) & cross_depth_mask
        )

        return hierarchical & same_sample & attention_allowed

    return peagle_mask_mod
