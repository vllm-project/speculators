import torch
from transformers.modeling_utils import AttentionInterface
from transformers.integrations.flex_attention import repeat_kv
from torch.nn.attention.flex_attention import flex_attention
from torch.nn.attention.flex_attention import or_masks, and_masks, BlockMask
from typing import Callable


def create_combined_mask_mod(lengths: torch.Tensor):
    total_seq_len = lengths.sum().item()
    document_ids = torch.repeat_interleave(
        torch.arange(lengths.shape[0], device=lengths.device, dtype=torch.long), lengths
    ).contiguous()
    N = document_ids.shape[0]

    def causal_mask_mod(b, h, q_idx, kv_idx):
        return q_idx >= kv_idx

    def document_mask_mod(b, h, q_idx, kv_idx):
        return document_ids[q_idx] == document_ids[kv_idx % N]

    def diagonal_draft_mask_mod(b, h, q_idx, kv_idx):
        return kv_idx % total_seq_len == q_idx

    return or_masks(
        and_masks(causal_mask_mod, document_mask_mod), diagonal_draft_mask_mod
    )


def extend_mask_for_draft_tokens(block_mask):
    """
    Extend the block mask to include new draft tokens. Concatenates a diagonal mask for the new draft tokens.

    Assumptions:
    - block_mask BLOCK_SIZE := KV_BLOCK_SIZE == Q_BLOCK_SIZE
    - The number of query values is the original total_seq_len (or equivalently the number of query blocks is the original total_seq_len // BLOCK_SIZE)

    i.e. if block_mask is:
    [
        [
            [1, 0, 0],
            [1, 1, 0],
            [0, 0, 1],
        ]
    ]
    the result will be:
    [
        [
            [1, 0, 0, 1, 0, 0],
            [1, 1, 0, 0, 1, 0],
            [0, 0, 1, 0, 0, 1],
        ]
    ]
    and then callinga again will give:
    [
        [
            [1, 0, 0, 1, 0, 0, 1, 0, 0],
            [1, 1, 0, 0, 1, 0, 0, 1, 0],
            [0, 0, 1, 0, 0, 1, 0, 0, 1],
        ]
    ]

    """
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

    return BlockMask.from_kv_blocks(
        kv_num_blocks,
        kv_indices,
        block_mask.full_kv_num_blocks,
        block_mask.full_kv_indices,
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


def flex_attention_forward(
    module: torch.nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask,
    scaling: float | None = None,
    softcap: float | None = None,
    head_mask: torch.Tensor | None = None,
    s_aux: torch.Tensor | None = None,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    block_mask = attention_mask
    enable_gqa = False

    num_local_query_heads = query.shape[1]
    # When running TP this helps:
    if (num_local_query_heads & (num_local_query_heads - 1)) != 0:
        key = repeat_kv(key, query.shape[1] // key.shape[1])
        value = repeat_kv(value, query.shape[1] // value.shape[1])

    return_lse = query.device.type != "cpu"

    query = query.contiguous()
    key = key.contiguous()
    value = value.contiguous()

    flex_attention_output = flex_attention(
        query,
        key,
        value,
        score_mod=None,
        block_mask=block_mask,
        enable_gqa=enable_gqa,
        scale=scaling,
        kernel_options=None,
        # Last time checked on PyTorch == 2.5.1: Flex Attention always computes the lse regardless.
        # For simplification, we thus always return it as no additional computations are introduced.
        return_lse=return_lse,
    )
    # lse is returned in float32
    if return_lse:
        attention_output, lse = flex_attention_output  # type: ignore[misc]
        lse = lse.to(value.dtype)
    else:
        attention_output = flex_attention_output  # type: ignore[assignment]
        lse = None

    attention_output = attention_output.transpose(1, 2).contiguous()
    return attention_output, lse


ALL_ATTENTION_FUNCTIONS = AttentionInterface()  # Singleton class used for registry
ALL_ATTENTION_FUNCTIONS.register("simple_flex_attention", flex_attention_forward)
