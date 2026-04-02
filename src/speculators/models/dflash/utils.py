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


def gather_anchor_spans(
    input_ids: torch.Tensor,
    anchor_positions: torch.Tensor,
    block_size: int,
) -> torch.Tensor:
    """Gather spans of tokens starting at anchor positions.

    Args:
        input_ids: Input token IDs [T]
        anchor_positions: Anchor indices [n]
        block_size: Number of tokens per span

    Returns:
        Concatenated spans [n*block_size]
    """
    input_ids = input_ids.view(-1)
    anchor_positions = anchor_positions.to(dtype=torch.long).view(-1)

    offsets = torch.arange(block_size, device=input_ids.device, dtype=torch.long)
    idx = anchor_positions[:, None] + offsets[None, :]

    if (idx < 0).any() or (idx >= input_ids.numel()).any():
        raise ValueError(
            "Some anchor_positions + offsets are out of range for input_ids."
        )

    return input_ids[idx.reshape(-1)]


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
    loss_mask: torch.Tensor, n: int, block_size: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Randomly select anchor positions from valid tokens in sequence.

    Args:
        loss_mask: Binary mask indicating valid positions [B, T]
        n: Number of anchors to select per batch item
        block_size: Block size (last block_size positions excluded)

    Returns:
        tuple: (anchors, anchor_valid)
            - anchors: Selected anchor indices [B, n]
            - anchor_valid: Boolean mask for valid anchors [B, n]
    """
    if loss_mask.ndim != 2:  # noqa: PLR2004
        raise ValueError(f"Expected [B, T], got {loss_mask.shape}")

    B, T = loss_mask.shape  # noqa: N806
    valid_mask = loss_mask.bool().clone()

    if block_size > 0:
        valid_mask[:, T - block_size :] = False

    out = []
    out_valid = []
    for b in range(B):
        valid_indices = torch.nonzero(valid_mask[b], as_tuple=False).squeeze(1)

        anchors = torch.zeros(n, dtype=torch.long, device=loss_mask.device)
        anchor_valid = torch.zeros(n, dtype=torch.bool, device=loss_mask.device)

        k = min(n, valid_indices.numel())
        if k > 0:
            perm = torch.randperm(valid_indices.numel(), device=loss_mask.device)
            anchors[:k] = valid_indices[perm[:k]]
            anchor_valid[:k] = True

        out.append(anchors)
        out_valid.append(anchor_valid)

    return torch.stack(out, dim=0), torch.stack(out_valid, dim=0)
