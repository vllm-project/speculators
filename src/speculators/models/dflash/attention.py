# ruff: noqa: ERA001
from typing import cast

import torch
from torch.nn.attention.flex_attention import (
    BlockMask,
    flex_attention,
    or_masks,
)
from transformers.modeling_utils import AttentionInterface


def create_anchor_block_mask_mod(
    lengths: torch.Tensor,
    total_seq_len: int,
    anchor_positions: torch.Tensor,
    block_size: int,
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
        - may attend to all tokens in their own synthetic block j
        - may not attend to other synthetic blocks or later base tokens

    Args:
        lengths: [num_docs] lengths of packed documents
        total_seq_len: padded packed sequence width
        anchor_positions: [n_anchors] absolute positions into the packed base sequence
        block_size: number of query tokens per anchor block

    Returns:
        mask_mod, q_len, kv_len
    """
    device = lengths.device
    anchor_positions = anchor_positions.to(device=device, dtype=torch.long).contiguous()

    if anchor_positions.ndim != 1:
        raise ValueError(
            f"anchor_positions must be 1D, got shape {tuple(anchor_positions.shape)}"
        )

    n_anchors = anchor_positions.numel()
    q_len = n_anchors * block_size
    kv_len = total_seq_len + q_len

    # Map each base-sequence position -> document id, padding -> -1
    document_ids = torch.repeat_interleave(
        torch.arange(lengths.shape[0], device=device, dtype=torch.long),
        lengths,
    )
    if document_ids.numel() > total_seq_len:
        raise ValueError(
            f"sum(lengths)={document_ids.numel()} exceeds total_seq_len={total_seq_len}"
        )
    if document_ids.numel() < total_seq_len:
        document_ids = torch.cat(
            [
                document_ids,
                -1
                * torch.ones(
                    total_seq_len - document_ids.numel(),
                    device=device,
                    dtype=torch.long,
                ),
            ]
        ).contiguous()

    if ((anchor_positions < 0) | (anchor_positions >= total_seq_len)).any():
        bad = anchor_positions[
            (anchor_positions < 0) | (anchor_positions >= total_seq_len)
        ]
        raise ValueError(f"anchor_positions out of range: {bad.tolist()}")

    anchor_docs = document_ids[anchor_positions]
    if (anchor_docs == -1).any():
        bad = anchor_positions[anchor_docs == -1]
        raise ValueError(f"anchor_positions include padding locations: {bad.tolist()}")

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

        return kv_is_base & same_doc & before_anchor

    def same_block_mod(_b, _h, q_idx, kv_idx):
        """
        Queries may attend bidirectionally to all tokens in their own synthetic block.
        """
        q_block = q_idx // block_size
        kv_is_block = kv_idx >= total_seq_len
        kv_block = (kv_idx - total_seq_len) // block_size

        return kv_is_block & (q_block == kv_block)

    return or_masks(base_prefix_mod, same_block_mod), q_len, kv_len


def block_mask_to_dense_attention_mask(
    block_mask: BlockMask, device: torch.device, dtype: torch.dtype
):
    attention_mask = torch.ones(block_mask.shape, device=device, dtype=dtype)

    for q_idx in range(attention_mask.shape[2]):
        attention_mask[0, 0, q_idx, :] = block_mask.mask_mod(
            torch.zeros(1, device=device, dtype=torch.long),
            torch.zeros(1, device=device, dtype=torch.long),
            torch.ones(1, device=device, dtype=torch.long) * q_idx,
            torch.arange(attention_mask.shape[3], device=device, dtype=torch.long),
        )
    return attention_mask


def flex_attention_forward(
    module: torch.nn.Module,  # noqa: ARG001
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask,
    scaling: float | None = None,
    **_kwargs,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    num_query_heads = query.shape[1]
    num_key_value_heads = key.shape[1]
    enable_gqa = num_query_heads != num_key_value_heads

    query = query.contiguous()
    key = key.contiguous()
    value = value.contiguous()

    flex_attention_output = flex_attention(
        query,
        key,
        value,
        score_mod=None,
        block_mask=attention_mask,
        enable_gqa=enable_gqa,
        scale=scaling,
    )
    attention_output: torch.Tensor = cast("torch.Tensor", flex_attention_output)
    attention_output = attention_output.transpose(1, 2).contiguous()
    return attention_output, None


ALL_ATTENTION_FUNCTIONS = AttentionInterface()  # Singleton class used for registry
ALL_ATTENTION_FUNCTIONS.register("simple_flex_attention", flex_attention_forward)
