import torch
from torch.nn.attention.flex_attention import (
    or_masks,
)


def create_causal_anchor_block_mask_mod(
    document_ids: torch.Tensor,
    total_seq_len: int,
    anchor_positions: torch.Tensor,
    block_size: int,
    sliding_window: int | None = None,
):
    """Build a flex-attention mask mod with CAUSAL blocks (JetSpec).

    Unlike DFlash's bidirectional block attention, JetSpec uses causal
    attention within each block: position i can attend to positions 0..i
    in its own block, plus the base prefix.

    Q side:
        n_anchors * block_size synthetic query tokens
        block j corresponds to anchor_positions[j]

    KV side:
        [ original packed sequence | synthetic anchor blocks ]

    For queries at position i in block j:
        - may attend to base tokens in the same document before anchor j
        - may attend to tokens at positions 0..i in their own block j (causal)
        - may not attend to other blocks or later base tokens
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

    query_anchor_positions = torch.repeat_interleave(anchor_positions, block_size)

    def base_prefix_mod(_b, _h, q_idx, kv_idx):
        """See base tokens in same document before anchor."""
        q_anchor = query_anchor_positions[q_idx]
        q_doc = document_ids[q_anchor]

        kv_is_base = kv_idx < total_seq_len
        kv_base_pos = torch.remainder(kv_idx, total_seq_len)
        kv_doc = document_ids[kv_base_pos]

        same_doc = (q_doc == kv_doc) & (q_doc != -1)
        before_anchor = kv_base_pos < q_anchor

        in_window = (
            (kv_base_pos >= q_anchor - sliding_window)
            if sliding_window is not None
            else True
        )

        return kv_is_base & same_doc & before_anchor & in_window

    def causal_block_mod(_b, _h, q_idx, kv_idx):
        """Queries attend causally to tokens in their own block."""
        q_block = q_idx // block_size
        kv_is_block = kv_idx >= total_seq_len
        kv_block = (kv_idx - total_seq_len) // block_size

        return kv_is_block & (q_block == kv_block) & (kv_idx <= q_idx + total_seq_len)

    return or_masks(base_prefix_mod, causal_block_mod), q_len, kv_len
