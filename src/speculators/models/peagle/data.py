"""COD sampling logic for P-EAGLE parallel group generation."""

import torch


def generate_cod_sample_indices(
    seq_length: int,
    loss_mask: torch.Tensor,
    para_num: int = 8,
    down_sample_ratio: float = 0.7,
    down_sample_ratio_min: float = 0.2,
    filter_position_zero: bool = True,
) -> tuple[list[torch.Tensor], int]:
    """
    Generate sampling indices for parallel sequences using COD sampling.

    Conditional-On-Distribution (COD) sampling reduces memory by using geometric
    decay: depth 0 retains all n positions, depth 1 retains n*r positions,
    depth 2 retains n*r^2 positions, etc.

    Args:
        seq_length: Length of the sequence
        loss_mask: Binary mask indicating valid training positions [batch, seq_len]
        para_num: Number of parallel prediction groups (K)
        down_sample_ratio: Geometric decay ratio r ∈ (0,1)
        down_sample_ratio_min: Minimum retention ratio floor
        filter_position_zero: Whether to filter out position 0 from candidates

    Returns:
        Tuple of:
        - sample_indices: List of K tensors, each containing position
          indices for that depth
        - total_additional_length: Sum of lengths of depths 1 through K-1
    """
    loss_mask = loss_mask.squeeze()
    all_valid_indices = torch.where(loss_mask == 1)[0]

    # Depth 0: Always include ALL positions
    sample_indices = [torch.arange(seq_length, device=loss_mask.device)]
    prev_indices = all_valid_indices

    for depth in range(1, para_num):
        # Calculate retention ratio with geometric decay and minimum floor
        valid_length = max(0, len(all_valid_indices) - depth)
        ratio = max(down_sample_ratio**depth, down_sample_ratio_min)
        sample_size = int(valid_length * ratio)

        if sample_size <= 0:
            break

        # Random sampling for robustness
        if len(prev_indices) >= sample_size:
            # Randomly select sample_size positions from prev_indices
            random_selection = torch.randperm(
                len(prev_indices), device=loss_mask.device
            )[:sample_size]
            sampled_idx = prev_indices[random_selection]
            sampled_idx = torch.sort(sampled_idx)[0]
        else:
            sampled_idx = prev_indices

        next_candidates = (sampled_idx + 1) % seq_length
        if filter_position_zero:
            next_candidates = next_candidates[next_candidates != 0]
        mask = torch.isin(next_candidates, all_valid_indices)
        prev_indices = next_candidates[mask]

        sample_indices.append(sampled_idx)

    total_additional_length = sum(len(idx) for idx in sample_indices[1:])
    return sample_indices, total_additional_length
