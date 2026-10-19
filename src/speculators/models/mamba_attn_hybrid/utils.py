import torch


def get_base_indices_for_anchored_blocks(
    anchor_positions: torch.Tensor,
    block_size: int,
    total_seq_len: int | None = None,
) -> torch.Tensor:
    anchor_positions = anchor_positions.to(dtype=torch.long).view(-1)

    offsets = torch.arange(block_size, device=anchor_positions.device, dtype=torch.long)
    idx = anchor_positions[:, None] + offsets[None, :]

    if (idx < 0).any() or (total_seq_len and (idx >= total_seq_len).any()):
        raise ValueError(
            "Some anchor_positions + offsets are out of range for total_seq_len"
            f"={total_seq_len}. Max={idx.max().item()}, min={idx.min().item()}"
        )

    return idx.reshape(-1)


def select_anchors(
    loss_mask: torch.Tensor,
    num_anchors: int,
    block_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if loss_mask.ndim != 2:  # noqa: PLR2004
        raise ValueError(f"Expected [B, T], got {loss_mask.shape}")

    if block_size <= 0:
        raise ValueError(f"Expected block size > 0, got {block_size}")

    valid_mask = loss_mask.bool().clone()
    valid_mask[:, -block_size:] = False

    valid_indices = torch.nonzero(valid_mask.squeeze(0), as_tuple=False).squeeze(-1)

    device = loss_mask.device
    anchors = torch.zeros(num_anchors, dtype=torch.long, device=device)
    anchor_valid = torch.zeros(num_anchors, dtype=torch.bool, device=device)

    k = min(num_anchors, valid_indices.numel())
    if k > 0:
        perm = torch.randperm(valid_indices.numel(), device=loss_mask.device)
        anchors[:k] = torch.sort(valid_indices[perm[:k]]).values
        anchor_valid[:k] = True

    return anchors, anchor_valid
