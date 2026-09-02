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
) -> tuple[torch.Tensor, torch.Tensor]:  # shapes: [num_docs*num_anchors] x2
    """Sample up to ``num_anchors`` anchor positions from every packed document.

    Mirrors the per-sample contract of the reference DFlash trainers (SpecForge
    and TorchSpec sample ``[B, num_anchors]``): every document in the packed
    sequence gets exactly ``num_anchors`` slots, filled with a uniform sample of
    its supervised positions (``loss_mask == 1``) and padded with
    ``anchor_valid == False`` when it has fewer. Short documents are fully
    covered and long ones are capped. The slot count depends only on the number
    of documents, so the compiled forward sees one static shape per document
    count. The last ``block_size`` positions are excluded so every anchor block
    stays in bounds.

    Returns ``(anchors, anchor_valid)`` flattened to ``[num_docs * num_anchors]``
    with anchors sorted within each document. A sequence without supervised
    positions yields one document of fully masked slots.
    """
    loss_mask = loss_mask.squeeze(0)
    document_ids = document_ids.squeeze(0)
    valid = loss_mask.bool().clone()
    valid[-block_size:] = False

    docs = document_ids[valid].unique()
    num_docs = max(docs.numel(), 1)
    device = loss_mask.device
    anchors = torch.zeros(num_docs, num_anchors, dtype=torch.long, device=device)
    anchor_valid = torch.zeros(num_docs, num_anchors, dtype=torch.bool, device=device)
    for i, doc in enumerate(docs):
        candidates = torch.nonzero(valid & (document_ids == doc)).squeeze(-1)
        perm = torch.randperm(candidates.numel(), device=device)[:num_anchors]
        # Sorted anchors let flex attention use dense (fast) blocks instead of
        # scattered all-partial (slow) ones; the order never affects the loss.
        anchors[i, : perm.numel()] = torch.sort(candidates[perm]).values
        anchor_valid[i, : perm.numel()] = True

    return anchors.flatten(), anchor_valid.flatten()
