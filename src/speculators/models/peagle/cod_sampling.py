"""
Conditional-On-Distribution (COD) sampling for P-EAGLE.

This module implements the COD sampling algorithm from the P-EAGLE paper
(arXiv:2602.01469). COD reduces the number of positions at each prediction
depth through geometric decay, making parallel multi-token prediction
training tractable.

At depth 0, all valid positions (where loss_mask == 1) are retained.
At depth k, n_valid × r^k positions are randomly retained, where r is the
retention rate (down_sample_ratio). This reduces total positions from n×K
to n×(1 + r + r² + ... + r^(K-1)), significantly reducing attention
memory from O((nK)²) to O((nΣr^i)²).
"""

from __future__ import annotations

import torch


def cod_sample(
    loss_mask: torch.Tensor,
    down_sample_ratio: float,
    num_depths: int,
    down_sample_ratio_min: float = 0.0,
    generator: torch.Generator | None = None,
) -> list[torch.Tensor]:
    """Perform COD sampling for P-EAGLE parallel group generation.

    Generates index sets for each prediction depth using geometric decay-based
    position retention. Only positions where loss_mask is True are candidates
    for sampling. Positions are returned in sorted (causal) order.

    Args:
        loss_mask: Boolean tensor of shape [seq_len] indicating valid training
            positions. Only positions with True values are candidates for sampling.
        down_sample_ratio: Retention rate r ∈ (0, 1) for geometric decay.
            Depth k retains n_valid × r^k positions.
        num_depths: Number of parallel prediction depths K (including depth 0).
            Must be >= 1.
        down_sample_ratio_min: Minimum retention ratio to ensure at least
            floor(n_valid × down_sample_ratio_min) positions at each depth.
            Defaults to 0.0 (no minimum).
        generator: Optional random generator for reproducible sampling.

    Returns:
        List of K tensors, where depth_indices[k] contains sorted indices of
        retained positions for depth k. depth_indices[0] always contains all
        valid positions.

    Raises:
        ValueError: If down_sample_ratio is not in (0, 1), num_depths < 1,
            or down_sample_ratio_min is not in [0, 1].

    Example:
        >>> loss_mask = torch.tensor([True, False, True, True, True, False, True])
        >>> indices = cod_sample(loss_mask, down_sample_ratio=0.5, num_depths=3)
        >>> len(indices)
        3
        >>> indices[0]  # All valid positions
        tensor([0, 2, 3, 4, 6])
        >>> len(indices[1])  # ~50% of valid positions
        2
        >>> len(indices[2])  # ~25% of valid positions
        1
    """
    if not 0 < down_sample_ratio < 1:
        raise ValueError(
            f"down_sample_ratio must be in (0, 1), got {down_sample_ratio}"
        )
    if num_depths < 1:
        raise ValueError(f"num_depths must be >= 1, got {num_depths}")
    if not 0 <= down_sample_ratio_min <= 1:
        raise ValueError(
            f"down_sample_ratio_min must be in [0, 1], got {down_sample_ratio_min}"
        )

    # Get indices of valid positions
    valid_indices = torch.nonzero(loss_mask, as_tuple=False).squeeze(-1)
    n_valid = valid_indices.shape[0]

    depth_indices: list[torch.Tensor] = []

    for k in range(num_depths):
        if k == 0:
            # Depth 0 retains all valid positions
            depth_indices.append(valid_indices.clone())
        else:
            # Compute retention count with geometric decay
            retain_ratio = max(down_sample_ratio**k, down_sample_ratio_min)
            n_retain = max(int(n_valid * retain_ratio), 0)

            if n_retain == 0 or n_valid == 0:
                depth_indices.append(
                    torch.tensor([], dtype=valid_indices.dtype, device=loss_mask.device)
                )
            elif n_retain >= n_valid:
                # Retain all if computed count exceeds available
                depth_indices.append(valid_indices.clone())
            else:
                # Random sampling without replacement
                perm = torch.randperm(n_valid, generator=generator, device="cpu")
                sampled = perm[:n_retain]
                # Sort to maintain causal order
                sampled_sorted, _ = sampled.sort()
                sampled_indices = valid_indices[sampled_sorted.to(valid_indices.device)]
                depth_indices.append(sampled_indices)

    return depth_indices


def compute_cod_statistics(
    depth_indices: list[torch.Tensor],
    seq_len: int,
) -> dict[str, int | float | list[int]]:
    """Compute statistics for a COD sampling result.

    Args:
        depth_indices: Output from cod_sample(), list of index tensors per depth.
        seq_len: Original sequence length.

    Returns:
        Dictionary with statistics including total positions, per-depth counts,
        compression ratio, and effective sequence length.
    """
    per_depth_counts = [idx.shape[0] for idx in depth_indices]
    total_positions = sum(per_depth_counts)
    num_depths = len(depth_indices)

    return {
        "num_depths": num_depths,
        "seq_len": seq_len,
        "per_depth_counts": per_depth_counts,
        "total_positions": total_positions,
        "naive_total": per_depth_counts[0] * num_depths if num_depths > 0 else 0,
        "compression_ratio": (
            total_positions / (per_depth_counts[0] * num_depths)
            if num_depths > 0 and per_depth_counts[0] > 0
            else 0.0
        ),
    }


def build_depth_position_ids(
    depth_indices: list[torch.Tensor],
    seq_len: int,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Build position IDs that track original sequence positions across depths.

    Creates a concatenated position ID tensor where each element records the
    original sequence position of the corresponding token, maintaining
    positional information across COD-sampled parallel groups.

    Args:
        depth_indices: Output from cod_sample(), list of index tensors per depth.
        seq_len: Original sequence length (unused, reserved for validation).
        device: Target device for the output tensor.

    Returns:
        1D tensor of shape [total_positions] containing original position IDs
        for each retained position across all depths.
    """
    if not depth_indices:
        return torch.tensor([], dtype=torch.long, device=device)

    position_ids = []
    for indices in depth_indices:
        if indices.numel() > 0:
            position_ids.append(indices.to(device=device))

    if not position_ids:
        return torch.tensor([], dtype=torch.long, device=device)

    return torch.cat(position_ids, dim=0)
