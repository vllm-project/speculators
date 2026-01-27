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


def create_combined_mask_mod(lengths: torch.Tensor, total_seq_len: int, block_size:int):
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

    def block_causal_mask_mod(_b, _h, q_idx, kv_idx):
        causal = q_idx >= kv_idx  # bool
        dif = q_idx - kv_idx

        in_range = (dif < block_size) & (dif > (-block_size))
        # causal * (1 - block)  ==>  causal & (~in_range)
        return causal & (~in_range)


    def document_mask_mod(_b, _h, q_idx, kv_idx):
        # Exclude padding tokens in attention mask
        return torch.logical_and(
            document_ids[q_idx] != -1,
            document_ids[q_idx] == document_ids[kv_idx % total_seq_len],
        )

    def diagonal_block_draft_mask_mod(_b, _h, q_idx, kv_idx):
        dif = torch.remainder(kv_idx, total_seq_len) - q_idx
        return ((-block_size) > dif) & (block_size < dif)

    return or_masks(
        and_masks(block_causal_mask_mod, document_mask_mod),
        diagonal_block_draft_mask_mod,
    )


def extend_mask_for_draft_tokens(block_mask):

    kv_num_blocks = block_mask.kv_num_blocks
    # shape: [B, H, Q_LEN // BLOCK_SIZE]

    kv_indices = block_mask.kv_indices
    # shape: [B, H, Q_LEN // BLOCK_SIZE, KV_LEN // BLOCK_SIZE]
    b, h, q_blocks, kv_blocks = kv_indices.shape

    # extend kv indices if needed
    kv_indices = torch.cat(
        [kv_indices, kv_indices.new_zeros((b, h, q_blocks, q_blocks))], dim=-1
    )
    new_block_indices = torch.arange(
        kv_blocks,
        kv_blocks + q_blocks,
        dtype=kv_indices.dtype,
        device=kv_indices.device,
    ).reshape(1, 1, q_blocks, 1)
    kv_indices.scatter_(
        dim=-1, index=kv_num_blocks.unsqueeze(-1), src=new_block_indices
    )

    kv_num_blocks = kv_num_blocks + 1
    if block_mask.full_kv_indices is not None:
        extended_full_kv_indices = torch.cat(
            [
                block_mask.full_kv_indices,
                block_mask.full_kv_indices.new_zeros((b, h, q_blocks, q_blocks)),
            ],
            dim=-1,
        )
    else:
        extended_full_kv_indices = None
    return BlockMask.from_kv_blocks(
        kv_num_blocks,
        kv_indices,
        block_mask.full_kv_num_blocks,
        extended_full_kv_indices,
        mask_mod=block_mask.mask_mod,
    )


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


import matplotlib.pyplot as plt
def flex_attention_forward(
    module: torch.nn.Module,  # noqa: ARG001
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask,
    scaling: float | None = None,
    **_kwargs,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    print("attn mask", attention_mask, flush=True)
    dense = block_mask_to_dense_attention_mask(attention_mask, query.device, torch.long)

    # strip batch + head dims
    img = dense[0, 0].float().cpu().nan_to_num()
    print(img.shape)
    plt.imsave("blockmask.png", img, cmap="gray")

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
