"""Utility functions for DFlash draft model."""

import torch


def get_base_indices_for_anchored_blocks(
    anchor_positions: torch.Tensor,  # shape: [1, num_anchors]
    block_size: int,
) -> torch.Tensor:  # shape: [num_anchors*block_size]
    anchor_positions = anchor_positions.to(dtype=torch.long).view(-1)
    # dtype: long, shape: [num_anchors]

    offsets = torch.arange(block_size, device=anchor_positions.device, dtype=torch.long)
    idx = (
        anchor_positions[:, None] + offsets[None, :]
    )  # shape: [num_anchors, block_size]

    return idx.reshape(-1)


def select_anchors(
    loss_mask: torch.Tensor,  # shape: [1, total_seq_len]
    document_ids: torch.Tensor,  # shape: [1, total_seq_len]
    num_anchors: int,
    block_size: int,
) -> torch.Tensor:  # shape: [total_anchors]
    """Sample up to ``num_anchors`` anchor positions from every packed document.

    Anchors are supervised positions (``loss_mask == 1``) drawn uniformly within
    each document, so short documents are fully covered and long ones are capped.
    This matches the reference DFlash trainers, which sample per sample rather
    than per packed sequence. The last ``block_size`` positions are excluded so
    every anchor block stays in bounds.

    Returns the sorted anchor positions. A sequence without supervised positions
    yields one anchor at position 0; its block is zeroed by the loss mask, which
    keeps the downstream shapes non-empty.
    """
    loss_mask = loss_mask.squeeze(0)
    document_ids = document_ids.squeeze(0)
    valid = loss_mask.bool().clone()
    valid[-block_size:] = False

    anchors = []
    for doc in document_ids[valid].unique():
        candidates = torch.nonzero(valid & (document_ids == doc)).squeeze(-1)
        perm = torch.randperm(candidates.numel(), device=candidates.device)
        anchors.append(candidates[perm[:num_anchors]])
    if not anchors:
        return torch.zeros(1, dtype=torch.long, device=loss_mask.device)

    # Sorted anchors let flex attention use dense (fast) blocks instead of
    # scattered all-partial (slow) ones; the order never affects the loss.
    return torch.sort(torch.cat(anchors)).values
