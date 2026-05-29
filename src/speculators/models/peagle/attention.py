"""Flex attention mask functions for P-EAGLE parallel group prediction."""

import torch


def create_peagle_mask_mod(
    anchor_pos: torch.Tensor,  # shape: [total_sampled]
    depth: torch.Tensor,  # shape: [total_sampled]
    lengths: torch.Tensor,  # shape: [batch_size]
    total_seq_len: int,
    sampled_anchors: torch.Tensor | None = None,  # shape: [num_anchors], indices into depth-0 positions
):
    """
    Create a flex attention mask modifier for P-EAGLE parallel groups.

    P-EAGLE uses COD (Conditional-On-Distribution) sampling to create parallel
    prediction groups. Each group (depth) has progressively fewer positions.

    This function creates a mask where each element can attend to only to previous
    elements in the same sampling chain/rollout and previous context in the base sample.

    Optional anchor-based attention: If sampled_anchors is provided, depth 0 attention
    is restricted to only the sampled anchor positions, reducing O(N²) to O(anchors × N).


    Args:
        anchor_pos: The starting position in the original sequence the current
            sampling chain started from.
        depth: Which COD sampling round each element belongs to
        lengths: The length of each document. Used to produce a document mask to prevent
            cross contamination
        total_seq_len: int, combined padded length of the original sequences
        sampled_anchors: Optional tensor of indices (into depth-0 positions) that were
            sampled as anchors. If provided, restricts depth-0 queries to these positions only.

    Args example:

    Given a sequnce of length 6.
    Original positions: [0,1,2,3,4,5]
    Apply COD sampling:
    Round 1: sample locations [0, 1, 3, 4]
    Round 2: sample locations [0, 3]
    Round 3: sample locations [0]
    anchor_pos: [0,1,2,3,4,5,0,1,3,4,0,3,0]
    depth: [0,0,0,0,0,0,1,1,1,1,2,2,3]
    reference positions (e.g. for target) = anchor_pos + depth
    [0,1,2,3,4,5,1,2,4,5,2,5,3]


    Returns:
        A mask_mod function compatible with flex_attention create_block_mask
    """

    # Generate sample_ids to prevent cross-sample attention
    document_ids = torch.repeat_interleave(
        torch.arange(lengths.shape[0], device=lengths.device, dtype=torch.long), lengths
    )
    # Pad ids with -1 to indicate padding
    document_ids = torch.cat(
        [
            document_ids,
            -1
            * torch.ones(
                total_seq_len - document_ids.shape[0],
                device=lengths.device,
                dtype=torch.long,
            ),
        ]
    ).contiguous()

    # Build anchor mask if using anchor-based attention for depth 0
    if sampled_anchors is not None:
        # Create a boolean mask indicating which positions are sampled anchors
        # is_anchor_position[i] = True if position i is a sampled anchor
        is_anchor_position = torch.zeros(anchor_pos.shape[0], dtype=torch.bool, device=anchor_pos.device)

        # Find all depth-0 positions
        depth0_mask = depth == 0
        depth0_positions = torch.where(depth0_mask)[0]  # Indices into anchor_pos/depth tensors

        # sampled_anchors contains indices into the set of depth-0 positions
        # Map these to actual positions in the full anchor_pos tensor
        if sampled_anchors.numel() > 0:
            sampled_depth0_indices = depth0_positions[sampled_anchors]
            is_anchor_position[sampled_depth0_indices] = True
    else:
        is_anchor_position = None

    def peagle_mask_mod(_b, _h, q_idx, kv_idx):
        q_anchor_pos = anchor_pos[q_idx]
        kv_anchor_pos = anchor_pos[kv_idx]
        q_depth = depth[q_idx]
        kv_depth = depth[kv_idx]

        same_document = document_ids[q_anchor_pos] == document_ids[kv_anchor_pos]
        is_not_padding = document_ids[q_anchor_pos] != -1
        same_rollout = q_anchor_pos == kv_anchor_pos
        kv_depth0 = kv_depth == 0
        in_depth_order = q_depth >= kv_depth
        is_anchor_causal = q_anchor_pos >= kv_anchor_pos

        # Depth 0 attention logic
        if is_anchor_position is not None:
            # With anchors: only sampled anchor positions can query depth 0
            q_is_anchor = is_anchor_position[q_idx]
            depth0_attention = kv_depth0 & is_anchor_causal & q_is_anchor
        else:
            # Without anchors: full causal attention for depth 0
            depth0_attention = kv_depth0 & is_anchor_causal

        return (
            is_not_padding
            & same_document
            & (depth0_attention | (same_rollout & in_depth_order))
        )

    return peagle_mask_mod
