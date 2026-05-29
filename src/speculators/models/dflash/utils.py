"""Utility functions for DFlash draft model."""

import torch


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
