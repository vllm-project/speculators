import torch
from torch.nn.attention.flex_attention import (
    or_masks,
)


def create_anchor_block_mask_mod(
    document_ids: torch.Tensor,
    total_seq_len: int,
    anchor_positions: torch.Tensor,
    block_size: int,
    sliding_window: int | None = None,
    sliding_window_non_causal: bool = False,
    sliding_window_prefix: int = 0,
):
    non_causal = sliding_window is None or sliding_window_non_causal

    device = document_ids.device
    anchor_positions = anchor_positions.to(device=device, dtype=torch.long).contiguous()

    if anchor_positions.ndim != 1:
        raise ValueError(
            f"anchor_positions must be 1D, got shape {tuple(anchor_positions.shape)}"
        )

    n_anchors = anchor_positions.numel()
    q_len = n_anchors * block_size
    kv_len = total_seq_len + q_len

    document_ids = document_ids.to(device=device, dtype=torch.long).contiguous()
    if document_ids.numel() != total_seq_len:
        raise ValueError(
            f"document_ids has {document_ids.numel()} entries, expected "
            f"total_seq_len={total_seq_len}"
        )

    doc_starts = None
    if sliding_window_prefix > 0:
        raise NotImplementedError(
            "sliding_window_prefix > 0 is not supported by this mask builder"
        )

    if (oob := (anchor_positions < 0) | (anchor_positions >= total_seq_len)).any():
        raise ValueError(
            f"anchor_positions out of range: {anchor_positions[oob].tolist()}"
        )

    query_anchor_positions = torch.repeat_interleave(anchor_positions, block_size)

    def base_prefix_mod(_b, _h, q_idx, kv_idx):
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

        if doc_starts is not None:
            in_prefix = (kv_base_pos - doc_starts[kv_base_pos]) < sliding_window_prefix
            in_window = in_prefix | in_window

        return kv_is_base & same_doc & before_anchor & in_window

    def same_block_mod(_b, _h, q_idx, kv_idx):
        q_block = q_idx // block_size
        kv_is_block = kv_idx >= total_seq_len
        kv_block = (kv_idx - total_seq_len) // block_size

        same = kv_is_block & (q_block == kv_block)
        if not non_causal:
            same = same & (kv_idx <= q_idx + total_seq_len)
        return same

    return or_masks(base_prefix_mod, same_block_mod), q_len, kv_len


def create_block_only_mask_mod(
    anchor_positions: torch.Tensor,
    block_size: int,
    sliding_window: int | None = None,
    sliding_window_non_causal: bool = False,
):
    non_causal = sliding_window is None or sliding_window_non_causal

    if anchor_positions.ndim != 1:
        raise ValueError(
            f"anchor_positions must be 1D, got shape {tuple(anchor_positions.shape)}"
        )
    n_anchors = anchor_positions.numel()
    q_len = n_anchors * block_size
    kv_len = q_len

    def same_block_mod(_b, _h, q_idx, kv_idx):
        same = (q_idx // block_size) == (kv_idx // block_size)
        if not non_causal:
            same = same & (kv_idx <= q_idx)
        return same

    return same_block_mod, q_len, kv_len
