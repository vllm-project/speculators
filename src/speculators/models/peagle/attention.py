"""Flex attention mask functions for P-EAGLE parallel group prediction."""

import torch


def create_peagle_mask_mod(
    all_indices: torch.Tensor,
    seq_length: int,
    num_depths: int,
    sample_ids: torch.Tensor | None = None,
):
    """
    Create a flex attention mask modifier for P-EAGLE parallel groups.

    P-EAGLE uses COD (Conditional-On-Distribution) sampling to create parallel
    prediction groups. Each group (depth) has progressively fewer positions.

    Args:
        all_indices: Flattened position indices for all groups [total_length]
        seq_length: Original sequence length (before parallel expansion)
        num_depths: Number of parallel groups (K)
        sample_ids: Optional sample IDs to prevent cross-sample attention
            [batch_size, seq_length]

    Returns:
        A mask_mod function compatible with flex_attention create_block_mask
    """
    device = all_indices.device
    total_length = num_depths * seq_length

    original_positions = torch.full(
        (total_length,), -1, dtype=torch.long, device=device
    )
    depth_assignments = torch.full((total_length,), -1, dtype=torch.long, device=device)

    for _concat_idx, full_idx in enumerate(all_indices):
        original_pos = full_idx % seq_length
        depth = full_idx // seq_length

        original_positions[full_idx] = original_pos
        depth_assignments[full_idx] = depth

    assert sample_ids is not None, "sample_ids must be provided"
    sample_ids_repeated = sample_ids.squeeze(0).repeat(num_depths)  # [total_length]

    def peagle_mask_mod(_b, _h, q_idx, kv_idx):
        """
        P-EAGLE attention mask matching p-eagle-train implementation.

        Note: q_idx and kv_idx are indices into the sampled sequence,
              not the full sequence. We map them to full indices using
              all_indices.
        """

        q_full_idx = all_indices[q_idx]
        kv_full_idx = all_indices[kv_idx]

        q_pos = original_positions[q_full_idx]
        kv_pos = original_positions[kv_full_idx]
        q_depth = depth_assignments[q_full_idx]
        kv_depth = depth_assignments[kv_full_idx]

        valid_query = q_depth != -1
        valid_key = kv_depth != -1

        hierarchical = q_depth >= kv_depth

        same_sample = (
            sample_ids_repeated[q_full_idx] == sample_ids_repeated[kv_full_idx]
        )

        same_depth = q_depth == kv_depth

        # Same-depth
        is_depth_0 = (q_depth == 0) & (kv_depth == 0)

        same_depth_causal_depth0 = (q_pos >= kv_pos) & is_depth_0

        same_depth_diagonal = (q_idx == kv_idx) & (~is_depth_0)

        same_depth_mask = same_depth_causal_depth0 | same_depth_diagonal

        kv_is_depth_0 = kv_depth == 0
        kv_is_intermediate = (kv_depth > 0) & (kv_depth < q_depth)

        cross_depth_to_depth0 = kv_is_depth_0 & (kv_pos <= (q_pos - q_depth))

        # Intermediate depths: attend ONLY to parent position
        parent_pos = q_pos - (q_depth - kv_depth)
        cross_depth_to_intermediate = kv_is_intermediate & (kv_pos == parent_pos)

        cross_depth_mask = cross_depth_to_depth0 | cross_depth_to_intermediate

        attention_allowed = (same_depth & same_depth_mask) | (
            (~same_depth) & cross_depth_mask
        )

        return valid_query & valid_key & hierarchical & same_sample & attention_allowed

    return peagle_mask_mod
