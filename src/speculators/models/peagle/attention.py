"""Flex attention mask functions for P-EAGLE parallel group prediction."""

import torch


@torch.compiler.disable
def _compute_doc_start_positions(document_ids: torch.Tensor) -> torch.Tensor:
    """Compute the start position of each document in a packed sequence.

    Args:
        document_ids: [total_seq_len] maps each position to its doc index,
            -1 for padding.

    Returns:
        [total_seq_len] tensor where entry *i* is the first position of the
        document that position *i* belongs to.  Padding positions get -1.
    """
    valid = document_ids != -1
    result = torch.full_like(document_ids, -1)
    if not valid.any():
        return result

    positions = torch.arange(len(document_ids), device=document_ids.device)
    unique_docs, inverse = torch.unique(document_ids[valid], return_inverse=True)
    first_pos = torch.full(
        (unique_docs.shape[0],),
        len(document_ids),
        device=document_ids.device,
        dtype=torch.long,
    )
    first_pos.scatter_reduce_(0, inverse, positions[valid], reduce="amin")
    result[valid] = first_pos[inverse]
    return result


def create_peagle_mask_mod(
    anchor_pos: torch.Tensor,  # shape: [total_sampled]
    depth: torch.Tensor,  # shape: [total_sampled]
    document_ids: torch.Tensor,  # shape: [total_seq_len]
    sink_size: int | None = None,
    max_context_window: int | None = None,
):
    """
    Create a flex attention mask modifier for P-EAGLE parallel groups.

    P-EAGLE uses COD (Conditional-On-Distribution) sampling to create parallel
    prediction groups. Each group (depth) has progressively fewer positions.

    This function creates a mask where each element can attend to only to previous
    elements in the same sampling chain/rollout and previous context in the base sample.

    When ``sink_size`` and ``max_context_window`` are both set (StreamingLLM mode),
    depth-0 causal attention is restricted to:
    - the first ``sink_size`` positions of each document (attention sinks), and
    - the most recent ``max_context_window`` positions before the query.

    Args:
        anchor_pos: The starting position in the original sequence the current
            sampling chain started from.
        depth: Which COD sampling round each element belongs to
        document_ids: Maps each position to its document index, -1 for padding
        sink_size: Number of initial tokens per document retained as attention
            sinks. Must be set together with max_context_window.
        max_context_window: Size of the local sliding window for depth-0 KV
            attention. Must be set together with sink_size.

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

    use_streaming = sink_size is not None and max_context_window is not None
    if use_streaming:
        doc_start_positions = _compute_doc_start_positions(document_ids)

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

        depth0_attention = kv_depth0 & is_anchor_causal

        if use_streaming:
            kv_doc_start = doc_start_positions[kv_anchor_pos]
            is_sink = (kv_anchor_pos - kv_doc_start) < sink_size
            in_window = kv_anchor_pos >= (q_anchor_pos - max_context_window)
            depth0_attention = depth0_attention & (is_sink | in_window)

        return (
            is_not_padding
            & same_document
            & (depth0_attention | (same_rollout & in_depth_order))
        )

    return peagle_mask_mod
