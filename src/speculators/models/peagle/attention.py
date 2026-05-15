"""Flex attention mask functions for P-EAGLE parallel group prediction."""

import torch


def create_peagle_mask_mod(
    anchor_pos: torch.Tensor,  # shape: [total_sampled]
    depth: torch.Tensor,  # shape: [total_sampled]
    lengths: torch.Tensor,  # shape: [batch_size]
    total_seq_len: int,
):
    """
    Create a flex attention mask modifier for P-EAGLE parallel groups.

    P-EAGLE uses COD (Conditional-On-Distribution) sampling to create parallel
    prediction groups. Each group (depth) has progressively fewer positions.

    This function creates a mask where each element can attend to only to previous
    elements in the same sampling chain/rollout and previous context in the base sample.


    Args:
        anchor_pos: The starting position in the original sequence the current
            sampling chain started from.
        depth: Which COD sampling round each element belongs to
        lengths: The length of each document. Used to produce a document mask to prevent
            cross contamination
        total_seq_len: int, combined padded length of the original sequences

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

        return (
            is_not_padding
            & same_document
            & ((kv_depth0 & is_anchor_causal) | (same_rollout & in_depth_order))
        )

    return peagle_mask_mod
