"""Utility functions for DSpark draft model.

Includes anchor sampling, attention mask creation, noise embedding,
position ID generation, and eval mask construction.
"""

import torch
from torch import nn


def build_anchor_candidate_mask(
    *,
    seq_len: int,
    loss_mask: torch.Tensor,
) -> torch.Tensor:
    """Build a boolean mask of valid anchor candidate positions.

    A position i is a valid anchor if both position i and i+1 are within
    the loss_mask (i.e., both are supervised tokens).
    """
    num_candidates = max(seq_len - 1, 0)
    if num_candidates == 0:
        return loss_mask[:, :0].bool()

    anchor_valid = loss_mask[:, :num_candidates] > 0.5
    first_target_valid = loss_mask[:, 1 : num_candidates + 1] > 0.5
    return anchor_valid & first_target_valid


def sample_anchor_positions(
    *,
    seq_len: int,
    loss_mask: torch.Tensor,
    num_anchors: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Randomly sample anchor positions from valid candidates.

    Args:
        seq_len: Length of the packed sequence.
        loss_mask: [B, T] binary mask of supervised positions.
        num_anchors: Maximum number of anchors to sample per sample.
        device: Target device.

    Returns:
        anchors: [B, num_anchors] sampled anchor positions (0 for invalid).
        keep_mask: [B, num_anchors] boolean mask of valid anchors.
    """
    valid = build_anchor_candidate_mask(seq_len=seq_len, loss_mask=loss_mask)
    valid_counts = valid.sum(dim=1)
    bsz = loss_mask.shape[0]
    num_candidates = valid.shape[1]
    max_n = int(num_anchors)

    if num_candidates == 0:
        anchors = torch.zeros(bsz, max_n, dtype=torch.long, device=device)
        keep_mask = torch.zeros(bsz, max_n, dtype=torch.bool, device=device)
        return anchors, keep_mask

    indices = torch.arange(num_candidates, device=device).unsqueeze(0).expand(bsz, -1)
    masked_indices = torch.where(
        valid,
        indices,
        torch.full_like(indices, seq_len + 1),
    )
    random_vals = torch.rand(bsz, num_candidates, device=device)
    random_vals = torch.where(valid, random_vals, torch.full_like(random_vals, 2.0))
    _, sorted_idx = random_vals.sort(dim=1)
    gathered = torch.gather(masked_indices, 1, sorted_idx)

    if num_candidates < max_n:
        pad = torch.full(
            (bsz, max_n - num_candidates),
            seq_len + 1,
            dtype=gathered.dtype,
            device=device,
        )
        gathered = torch.cat([gathered, pad], dim=1)

    anchors = gathered[:, :max_n].sort(dim=1).values
    keep_mask = torch.arange(max_n, device=device).unsqueeze(0) < (
        valid_counts.unsqueeze(1).clamp(max=max_n)
    )
    anchors = torch.where(keep_mask, anchors, torch.zeros_like(anchors))
    return anchors, keep_mask


def create_dspark_attention_mask(
    *,
    anchor_positions: torch.Tensor,
    block_keep_mask: torch.Tensor,
    seq_len: int,
    block_size: int,
    device: torch.device,
):
    """Create an attention mask for DSpark cross-attention.

    Draft queries in block j can attend to:
    - Target prefix tokens with position < anchor_positions[j]
    - Tokens in their own draft block j

    Returns a standard additive mask (0 for attend, -inf for masked) as a
    [B, 1, Q_LEN, KV_LEN] tensor, compatible with eager/sdpa attention.
    """
    bsz, num_blocks = anchor_positions.shape
    Q_LEN = num_blocks * block_size
    KV_LEN = seq_len + num_blocks * block_size

    # Create query and key position indices
    q_idx = torch.arange(Q_LEN, device=device).view(1, Q_LEN, 1)      # [1, Q, 1]
    kv_idx = torch.arange(KV_LEN, device=device).view(1, 1, KV_LEN)   # [1, 1, KV]

    # Compute block IDs
    q_block_id = q_idx // block_size  # [1, Q, 1]

    # Gather anchor positions for each query block: [B, num_blocks] -> [B, Q, 1]
    anchor_pos = anchor_positions[:, :, None].expand(-1, num_blocks, block_size)  # [B, num_blocks, block_size]
    anchor_pos = anchor_pos.reshape(bsz, Q_LEN).unsqueeze(-1)  # [B, Q, 1]

    # Context mask: kv_idx < seq_len AND kv_idx < anchor position
    is_context = (kv_idx < seq_len) & (kv_idx < anchor_pos)  # [1, 1, KV] & [B, Q, 1] -> [B, Q, KV]

    # Draft mask: kv_idx >= seq_len AND same block
    is_draft = kv_idx >= seq_len
    kv_block_id = (kv_idx - seq_len) // block_size  # [1, 1, KV]
    mask_draft = is_draft & (q_block_id == kv_block_id)  # [1, Q, 1] & [1, 1, KV] -> [1, Q, KV]

    # Block validity mask
    block_valid = block_keep_mask[:, :, None].expand(-1, num_blocks, block_size)
    block_valid = block_valid.reshape(bsz, Q_LEN).unsqueeze(-1)  # [B, Q, 1]

    # Combine: allowed = (context OR draft) AND valid_block
    allowed = (is_context | mask_draft) & block_valid  # [B, Q, KV]

    # Convert to additive mask: 0.0 for allowed, -inf for masked
    mask = torch.where(allowed, 0.0, float('-inf')).unsqueeze(1)  # [B, 1, Q, KV]
    return mask


def create_position_ids(
    anchor_positions: torch.Tensor,
    block_size: int,
) -> torch.Tensor:
    """Create position IDs for draft tokens based on anchor positions.

    Args:
        anchor_positions: [B, num_blocks] anchor positions.
        block_size: Number of draft tokens per anchor.

    Returns:
        [B, num_blocks * block_size] position IDs.
    """
    bsz, num_blocks = anchor_positions.shape
    device = anchor_positions.device
    offsets = torch.arange(block_size, device=device).view(1, 1, -1)
    return (anchor_positions.unsqueeze(-1) + offsets).view(bsz, num_blocks * block_size)


def create_noise_embed(
    embed_tokens: nn.Module,
    input_ids: torch.Tensor,
    anchor_positions: torch.Tensor,
    block_keep_mask: torch.Tensor,
    *,
    mask_token_id: int,
    block_size: int,
) -> torch.Tensor:
    """Create noise embedding for draft tokens.

    The first position of each block is set to the anchor token,
    remaining positions are filled with mask_token_id.

    Args:
        embed_tokens: Token embedding module.
        input_ids: [B, T] source token IDs.
        anchor_positions: [B, num_blocks] anchor positions.
        block_keep_mask: [B, num_blocks] valid block mask.
        mask_token_id: Token ID for masked positions.
        block_size: Number of draft tokens per anchor.

    Returns:
        [B, num_blocks * block_size, D] noise embeddings.
    """
    bsz = input_ids.shape[0]
    num_blocks = anchor_positions.shape[1]
    device = input_ids.device

    noise_ids = torch.full(
        (bsz, num_blocks * block_size),
        mask_token_id,
        dtype=torch.long,
        device=device,
    )
    block_starts = torch.arange(num_blocks, device=device) * block_size
    block_starts = block_starts.unsqueeze(0).expand(bsz, -1)
    anchor_tokens = torch.gather(input_ids, 1, anchor_positions)
    flat_batch_idx = torch.arange(bsz, device=device).unsqueeze(1).expand(bsz, num_blocks)
    noise_ids[flat_batch_idx, block_starts] = torch.where(
        block_keep_mask,
        anchor_tokens,
        torch.tensor(mask_token_id, dtype=torch.long, device=device),
    )
    return embed_tokens(noise_ids)


def build_eval_mask(
    *,
    seq_len: int,
    loss_mask: torch.Tensor,
    label_indices: torch.Tensor,
    safe_label_indices: torch.Tensor,
    block_keep_mask: torch.Tensor,
) -> torch.Tensor:
    """Build evaluation mask for draft token supervision.

    A draft position is supervised if:
    - Its label is within the sequence
    - The label is in the loss_mask
    - All previous positions in the same block are also supervised (contiguous prefix)

    Args:
        seq_len: Source sequence length.
        loss_mask: [B, T] supervised token mask.
        label_indices: [B, num_blocks, block_size] target token positions.
        safe_label_indices: [B, num_blocks, block_size] clamped label indices.
        block_keep_mask: [B, num_blocks] valid block mask.

    Returns:
        [B, num_blocks, block_size] boolean eval mask.
    """
    target_valid = label_indices < seq_len
    target_loss_mask = torch.gather(
        loss_mask.unsqueeze(1).expand(-1, label_indices.size(1), -1),
        2,
        safe_label_indices,
    )
    eval_mask = target_valid & (target_loss_mask > 0.5)
    eval_mask = eval_mask & block_keep_mask.unsqueeze(-1)
    return eval_mask.to(torch.int32).cumprod(dim=-1).bool()


def extract_context_feature(hidden_states, layer_ids):
    """Extract and concatenate hidden states from specified target layers.

    Args:
        hidden_states: Tuple of hidden states from target model layers.
        layer_ids: List of layer indices to extract (-1 for embedding output).

    Returns:
        Concatenated hidden states.
    """
    return torch.cat(
        [hidden_states[0 if layer_id == -1 else layer_id + 1] for layer_id in layer_ids],
        dim=-1,
    )


def validate_target_layer_ids(layer_ids, num_target_layers: int):
    """Validate target layer IDs are within range and strictly increasing."""
    layer_ids = [int(layer_id) for layer_id in layer_ids]
    if not layer_ids:
        raise ValueError("target_layer_ids must not be empty.")
    start = 0
    end = int(num_target_layers) - 1
    previous = None
    for layer_id in layer_ids:
        if not (layer_id == -1 or start <= layer_id <= end):
            raise ValueError(
                f"target_layer_id {layer_id} is out of range {{-1}} U [{start}, {end}] "
                f"for num_target_layers={num_target_layers}."
            )
        if previous is not None and layer_id <= previous:
            raise ValueError("target_layer_ids must be strictly increasing.")
        previous = layer_id
    return layer_ids


__all__ = [
    "build_anchor_candidate_mask",
    "sample_anchor_positions",
    "create_dspark_attention_mask",
    "create_position_ids",
    "create_noise_embed",
    "build_eval_mask",
    "extract_context_feature",
    "validate_target_layer_ids",
]
