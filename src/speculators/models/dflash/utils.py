"""Utility functions for DFlash draft model."""

import torch


def extend_position_ids_with_mask_tokens(
    position_ids: torch.Tensor,  # shape: [1, total_seq_len]
    anchor_positions: torch.Tensor,  # shape: [num_anchors]
    block_size: int,
) -> torch.Tensor:  # [1, total_seq_len + num_anchors * block_size]
    """Append each synthetic block's rotary positions to the source positions."""

    anchor_position_ids = (
        position_ids[0, anchor_positions].to(dtype=torch.long).view(-1)
    )
    # shape: [num_anchors]

    offsets = torch.arange(
        block_size, device=anchor_position_ids.device, dtype=torch.long
    )
    block_position_ids = (anchor_position_ids[:, None] + offsets[None, :]).reshape(
        1, -1
    )
    # shape: [1, num_anchors*block_size]

    return torch.cat([position_ids, block_position_ids], dim=1)
    # shape: [1, total_seq_len + num_anchors*block_size]


@torch.compiler.disable
def select_anchors(
    loss_mask: torch.Tensor,  # shape: [1, total_seq_len]
    num_anchors: int,
    block_size: int,
    train_anchor: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return anchors, safe source indices, and the aligned prediction mask.

    Inputs are a single packed sequence [1, T], with positive T and anchor
    count. The final block_size source positions cannot be sampled as anchors.
    Padded blocks have zero loss; their source indices remain in range even
    when T is shorter than block_size.
    """
    if loss_mask.ndim != 2 or loss_mask.shape[0] != 1:  # noqa: PLR2004
        raise ValueError(f"Expected a packed sequence [1, T], got {loss_mask.shape}")
    if loss_mask.shape[1] == 0:
        raise ValueError("Expected a nonempty packed sequence")
    if num_anchors <= 0:
        raise ValueError(f"Expected num_anchors > 0, got {num_anchors}")

    if block_size <= 0:
        raise ValueError(f"Expected block size > 0, got {block_size}")

    valid_mask = loss_mask.bool().clone()
    valid_mask[:, -block_size:] = False

    valid_indices = torch.nonzero(valid_mask.squeeze(0), as_tuple=False).squeeze(
        -1
    )  # shape: [num_non_zero]

    device = loss_mask.device
    anchors = torch.zeros(num_anchors, dtype=torch.long, device=device)

    k = min(num_anchors, valid_indices.numel())

    # Constrain value of k for torch dynamo
    torch._check(k <= valid_indices.numel())  # noqa: SLF001
    torch._check(k >= 0)  # noqa: SLF001

    perm = torch.randperm(valid_indices.numel(), device=loss_mask.device)
    # Contiguous anchors let flex attention use dense (fast) blocks instead of
    # scattered all-partial (slow) ones; the order never affects the loss.
    anchors[:k] = torch.sort(torch.gather(valid_indices, 0, perm[:k])).values

    offsets = torch.arange(block_size, device=anchors.device, dtype=torch.long)
    block_positions = (anchors[:, None] + offsets[None, :]).reshape(-1)
    block_positions = block_positions.clamp(max=loss_mask.shape[1] - 1)
    # shape: [num_anchors*block_size]

    aligned_loss_mask = loss_mask.clone()[:, block_positions]
    # shape: [1, num_anchors*block_size]

    # zero out any padded anchor blocks
    aligned_loss_mask[:, k * block_size :] = False

    # For sample_from_anchor=False, mask slot 0 (anchor) since it's not trained
    if not train_anchor:
        aligned_loss_mask[:, ::block_size] = False

    return (
        anchors,  # [num_anchors]
        block_positions,  # [num_anchors * block_size]
        aligned_loss_mask,  # [1, num_anchors * block_size]
    )
    # shape: [num_anchors], [num_anchors*block_size], [1, num_anchors*block_size]
