import torch
from torch.nn.attention.flex_attention import (
    or_masks,
)


def create_anchor_block_mask_mod(
    document_ids: torch.Tensor,
    total_seq_len: int,
    anchor_positions: torch.Tensor,
    block_size: int,
    is_causal: bool,
    sliding_window: int | None = None,
):
    """
    Build a flex-attention mask mod where each query block corresponds to one anchor.

    Q side:
        n_anchors * block_size synthetic query tokens
        block j corresponds to anchor_positions[j]

    KV side:
        [ original packed sequence | synthetic anchor blocks ]

    For queries in block j:
        - may attend to base tokens in the same document with
          position < anchor_positions[j]
        - may attend to tokens in their own synthetic block j, subject to
          causal and sliding-window constraints
        - may not attend to other synthetic blocks or later base tokens

    Args:
        document_ids: [total_seq_len] maps each position to its doc index, pad -1
        total_seq_len: padded packed sequence width
        anchor_positions: [n_anchors] absolute positions into the packed base sequence
        block_size: number of query tokens per anchor block
        is_causal: Whether queries may attend only to current and prior positions
        sliding_window: integer size of sliding window or None for full attn

    Returns:
        mask_mod, q_len, kv_len
    """
    device = document_ids.device
    anchor_positions = anchor_positions.to(device=device, dtype=torch.long).contiguous()

    if anchor_positions.ndim != 1:
        raise ValueError(
            f"anchor_positions must be 1D, got shape {tuple(anchor_positions.shape)}"
        )

    n_anchors = anchor_positions.numel()
    q_len = n_anchors * block_size
    kv_len = total_seq_len + q_len

    # For each query position, which anchor does it belong to?
    # query q in [j*block_size, (j+1)*block_size) belongs to anchor_positions[j]
    query_anchor_positions = torch.repeat_interleave(anchor_positions, block_size)

    def base_prefix_mod(_b, _h, q_idx, kv_idx):
        """
        Queries may see base-sequence tokens in the same document before the anchor.
        """
        # absolute base position
        q_anchor = query_anchor_positions[q_idx]
        # doc id for this query block
        q_doc = document_ids[q_anchor]

        kv_is_base = kv_idx < total_seq_len
        kv_base_pos = torch.remainder(kv_idx, total_seq_len)  # safe indexing
        kv_doc = document_ids[kv_base_pos]

        same_doc = (q_doc == kv_doc) & (q_doc != -1)
        before_anchor = kv_base_pos < q_anchor

        # The window is relative to each query and includes the current token.
        query_position = q_anchor + q_idx % block_size
        in_window = (
            (kv_base_pos > query_position - sliding_window)
            if sliding_window is not None
            else True
        )

        return kv_is_base & same_doc & before_anchor & in_window

    def same_block_mod(_b, _h, q_idx, kv_idx):
        """
        Queries may attend within their own block, subject to causality and window.
        """
        q_block = q_idx // block_size
        kv_is_block = kv_idx >= total_seq_len
        kv_block = (kv_idx - total_seq_len) // block_size

        same = kv_is_block & (q_block == kv_block)
        if is_causal:
            same = same & (kv_idx <= q_idx + total_seq_len)
        return same

    return or_masks(base_prefix_mod, same_block_mod), q_len, kv_len
