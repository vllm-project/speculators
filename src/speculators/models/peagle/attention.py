"""Flex attention mask functions for P-EAGLE parallel group prediction."""

from typing import cast

import torch
from torch.nn.attention.flex_attention import flex_attention
from transformers.modeling_utils import AttentionInterface


def create_peagle_mask_mod(
    all_indices: torch.Tensor,
    seq_length: int,
    para_depth: int,
    sample_ids: torch.Tensor | None = None,
):
    """
    Create a flex attention mask modifier for P-EAGLE parallel groups.

    P-EAGLE uses COD (Conditional-On-Distribution) sampling to create parallel
    prediction groups. Each group (depth) has progressively fewer positions.

    Args:
        all_indices: Flattened position indices for all groups [total_length]
        seq_length: Original sequence length (before parallel expansion)
        para_depth: Number of parallel groups (K)
        sample_ids: Optional sample IDs to prevent cross-sample attention
            [batch_size, seq_length]

    Returns:
        A mask_mod function compatible with flex_attention create_block_mask
    """
    device = all_indices.device
    total_length = para_depth * seq_length

    original_positions = torch.full(
        (total_length,), -1, dtype=torch.long, device=device
    )
    depth_assignments = torch.full((total_length,), -1, dtype=torch.long, device=device)

    for _concat_idx, full_idx in enumerate(all_indices):
        original_pos = full_idx % seq_length
        depth = full_idx // seq_length

        original_positions[full_idx] = original_pos
        depth_assignments[full_idx] = depth

    assert sample_ids is not None, "sample_ids must be provided"
    sample_ids_repeated = sample_ids.squeeze(0).repeat(para_depth)  # [total_length]

    def peagle_mask_mod(_b, _h, q_idx, kv_idx):
        """
        P-EAGLE attention mask matching p-eagle-train implementation.

        Note: q_idx and kv_idx are indices into the sampled sequence,
              not the full sequence. We map them to full indices using
              all_indices.
        """

        q_full_idx = all_indices[q_idx]
        kv_full_idx = all_indices[kv_idx]

        q_pos = original_positions[q_full_idx]
        kv_pos = original_positions[kv_full_idx]
        q_depth = depth_assignments[q_full_idx]
        kv_depth = depth_assignments[kv_full_idx]

        valid_query = q_depth != -1
        valid_key = kv_depth != -1

        hierarchical = q_depth >= kv_depth

        same_sample = (
            sample_ids_repeated[q_full_idx] == sample_ids_repeated[kv_full_idx]
        )

        same_depth = q_depth == kv_depth

        # Same-depth
        is_depth_0 = (q_depth == 0) & (kv_depth == 0)

        same_depth_causal_depth0 = (q_pos >= kv_pos) & is_depth_0

        same_depth_diagonal = (q_idx == kv_idx) & (~is_depth_0)

        same_depth_mask = same_depth_causal_depth0 | same_depth_diagonal

        kv_is_depth_0 = kv_depth == 0
        kv_is_intermediate = (kv_depth > 0) & (kv_depth < q_depth)

        cross_depth_to_depth0 = kv_is_depth_0 & (kv_pos <= (q_pos - q_depth))

        # Intermediate depths: attend ONLY to parent position
        parent_pos = q_pos - (q_depth - kv_depth)
        cross_depth_to_intermediate = kv_is_intermediate & (kv_pos == parent_pos)

        cross_depth_mask = cross_depth_to_depth0 | cross_depth_to_intermediate

        attention_allowed = (same_depth & same_depth_mask) | (
            (~same_depth) & cross_depth_mask
        )

        return valid_query & valid_key & hierarchical & same_sample & attention_allowed

    return peagle_mask_mod


def peagle_flex_attention_forward(
    module: torch.nn.Module,  # noqa: ARG001
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask,
    scaling: float | None = None,
    **_kwargs,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """
    Flex attention forward pass for P-EAGLE.

    Uses flex_attention with block-sparse masks for memory efficiency.

    Args:
        module: The attention module (unused, for interface compatibility)
        query: Query tensor [batch, num_heads, seq_len, head_dim]
        key: Key tensor [batch, num_kv_heads, seq_len, head_dim]
        value: Value tensor [batch, num_kv_heads, seq_len, head_dim]
        attention_mask: BlockMask object from create_block_mask()
        scaling: Optional attention scaling factor
        **_kwargs: Additional arguments (unused)

    Returns:
        Tuple of (attention_output, None)
    """
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


# Register P-EAGLE flex attention with transformers AttentionInterface
ALL_ATTENTION_FUNCTIONS = AttentionInterface()
ALL_ATTENTION_FUNCTIONS.register("peagle_flex_attention", peagle_flex_attention_forward)
