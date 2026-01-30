# ruff: noqa: ERA001
from typing import cast

import torch
from torch.nn.attention.flex_attention import (
    BlockMask,
    and_masks,
    flex_attention,
    or_masks,
)
from transformers.modeling_utils import AttentionInterface


def create_combined_mask_mod(lengths: torch.Tensor, total_seq_len: int, block_size:int, padding:int):
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

    # print(f"DEBUG: document_ids: {document_ids}", flush=True)
    # print(f"DEBUG: lengths: {lengths}, total_seq_len: {total_seq_len}, block_size: {block_size}", flush=True)

    def causal_mask_mod(_b, _h, q_idx, kv_idx):
        causal = q_idx >= kv_idx  # bool
        return causal


    def document_mask_mod(_b, _h, q_idx, kv_idx):
        # Exclude padding tokens in attention mask
        return torch.logical_and(
            document_ids[q_idx] != -1,
            document_ids[q_idx] == document_ids[kv_idx % total_seq_len],
        )
    def right_mask_mod(_b, _h, q_idx, kv_idx):
        return kv_idx>total_seq_len 
    def diagonal_block_draft_mask_mod(_b, _h, q_idx, kv_idx):
        k = torch.remainder(kv_idx, total_seq_len)
        return (q_idx // block_size) == (k // block_size)
    def not_diagonal_block_draft_mask_mod(_b, _h, q_idx, kv_idx):
        k = torch.remainder(kv_idx, total_seq_len)
        return (q_idx // block_size) != (k // block_size)
    def right_doc_mod_q(_b, _h, q_idx, kv_idx):
        return q_idx<padding
    def right_doc_mod_kv(_b, _h, q_idx, kv_idx):
        return kv_idx<padding+total_seq_len

    right=and_masks(and_masks(right_doc_mod_kv,right_doc_mod_q),and_masks(and_masks(diagonal_block_draft_mask_mod, right_mask_mod), document_mask_mod))


    left=and_masks(document_mask_mod,and_masks(not_diagonal_block_draft_mask_mod, causal_mask_mod))
    
    return or_masks(right, left)







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
    # print(f"DEBUG: Query shape: {query.shape}, Key shape: {key.shape}, Value shape: {value.shape}", flush=True)
    # print(f"DEBUG: BlockMask shape: {attention_mask.shape}", flush=True)
    # print("attn mask", attention_mask, flush=True)
    # dense = block_mask_to_dense_attention_mask(attention_mask, query.device, torch.long)
    # print(f"DEBUG: Dense mask stats - min: {dense.min()}, max: {dense.max()}, has_nan: {torch.isnan(dense).any()}", flush=True)

    # Save mask visualization (disabled - requires matplotlib)
    # import matplotlib.pyplot as plt
    # img = dense[0, 0].float().cpu().nan_to_num()
    # print(img.shape)
    # plt.imsave("blockmask.png", img, cmap="gray")

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
