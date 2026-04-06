"""Utility functions for DFlash draft model."""

import torch


def build_kv_position_ids(
    base_position_ids: torch.Tensor,
    anchor_positions: torch.Tensor,
    block_size: int,
) -> torch.Tensor:
    """Construct position_ids for KV = [base | anchor_blocks].

    Appended block for anchor a gets positions:
        base_position_ids[..., a] + [0..block_size-1]

    Args:
        base_position_ids: Base position IDs [B, total_seq_len]
        anchor_positions: Anchor indices [B, n] or [n]
        block_size: Size of each anchor block

    Returns:
        Combined position IDs [B, total_seq_len + n*block_size]
    """
    B, T = base_position_ids.shape  # noqa: N806
    device = base_position_ids.device

    # Normalize anchor_positions to [B, n]
    if anchor_positions.ndim == 1:  # noqa: PLR2004
        anchor_positions = (
            anchor_positions.to(device=device, dtype=torch.long)
            .unsqueeze(0)
            .expand(B, -1)
        )
    elif anchor_positions.ndim == 2:  # noqa: PLR2004
        anchor_positions = anchor_positions.to(device=device, dtype=torch.long)
        if anchor_positions.shape[0] != B:
            raise ValueError(
                f"anchor_positions batch {anchor_positions.shape[0]} != {B}"
            )
    else:
        raise ValueError(
            f"anchor_positions must be [n] or [B, n], got {anchor_positions.shape}"
        )

    n = anchor_positions.shape[1]

    anchor_pos_ids = torch.gather(
        base_position_ids.to(torch.long), dim=1, index=anchor_positions
    )

    offsets = torch.arange(block_size, device=device, dtype=torch.long).view(
        1, 1, block_size
    )

    appended_pos_ids = (anchor_pos_ids.unsqueeze(-1) + offsets).reshape(
        B, n * block_size
    )

    return torch.cat([base_position_ids.to(torch.long), appended_pos_ids], dim=1)


def get_base_indices_for_anchored_blocks(
    anchor_positions: torch.Tensor,  # shape: [1, num_anchors]
    block_size: int,
    total_seq_len: int | None = None,
) -> torch.Tensor:  # shape: [num_anchors*block_size]
    anchor_positions = anchor_positions.to(dtype=torch.long).view(-1)
    # dtype: long, shape: [num_anchors]

    offsets = torch.arange(block_size, device=anchor_positions.device, dtype=torch.long)
    idx = (
        anchor_positions[:, None] + offsets[None, :]
    )  # shape: [num_anchors, block_size]

    if (idx < 0).any() or (total_seq_len and (idx >= total_seq_len).any()):
        raise ValueError(
            "Some anchor_positions + offsets are out of range for total_seq_len"
            f"={total_seq_len}. Max={idx.max().item()}, min={idx.min().item()}"
        )

    return idx.reshape(-1)


def build_target_layer_ids(num_target_layers: int, num_draft_layers: int):
    """Calculate which target model layers to use for DFlash draft model.

    Args:
        num_target_layers: Total number of layers in target model
        num_draft_layers: Number of layers in draft model

    Returns:
        List of target layer indices to extract hidden states from
    """
    if num_draft_layers == 1:
        return [(num_target_layers // 2)]
    start = 1
    end = num_target_layers - 3
    span = end - start
    return [
        int(round(start + (i * span) / (num_draft_layers - 1)))
        for i in range(num_draft_layers)
    ]


def _select_anchors(
    loss_mask: torch.Tensor,  # shape: [1, total_seq_len]
    num_anchors: int,
    block_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Randomly select anchor positions from valid tokens in sequence.

    Args:
        loss_mask: Binary mask indicating valid positions [1, total_seq_len]
        n: Number of anchors to select per batch item
        block_size: Block size (last block_size positions excluded)

    Returns:
        tuple: (anchors, anchor_valid)
            - anchors: Selected anchor indices [1, num_anchors]
            - anchor_valid: Boolean mask for valid anchors [1, num_anchors]
    """
    if loss_mask.ndim != 2:  # noqa: PLR2004
        raise ValueError(f"Expected [B, T], got {loss_mask.shape}")

    if block_size <= 0:
        raise ValueError(f"Expected block size >= 0, got {block_size}")

    valid_mask = loss_mask.bool().clone()
    valid_mask[:, -block_size + 1 :] = False

    valid_indices = torch.nonzero(valid_mask.squeeze(0), as_tuple=False).squeeze(
        -1
    )  # shape: [num_non_zero]

    device = loss_mask.device
    anchors = torch.zeros(num_anchors, dtype=torch.long, device=device)
    anchor_valid = torch.zeros(num_anchors, dtype=torch.bool, device=device)

    k = min(num_anchors, valid_indices.numel())
    if k > 0:
        perm = torch.randperm(valid_indices.numel(), device=loss_mask.device)
        anchors[:k] = valid_indices[perm[:k]]
        anchor_valid[:k] = True

    return anchors, anchor_valid
    # shape: [num_anchors], [num_anchors]
