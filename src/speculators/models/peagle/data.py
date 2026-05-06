"""COD sampling logic for P-EAGLE parallel group generation."""

import torch


def generate_cod_sample_indices(
    seq_length: int,
    loss_mask: torch.Tensor,
    num_depths: int = 8,
    down_sample_ratio: float = 0.7,
    down_sample_ratio_min: float = 0.2,
    filter_position_zero: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """
    Generate sampling indices for parallel sequences using COD sampling.

    Conditional-On-Distribution (COD) sampling reduces memory by using geometric
    decay: depth 0 retains all n positions, depth 1 retains n*r positions,
    depth 2 retains n*r^2 positions, etc.

    Args:
        seq_length: Length of the sequence
        loss_mask: Binary mask indicating valid training positions [batch, seq_len]
        num_depths: Number of parallel prediction groups (K)
        down_sample_ratio: Geometric decay ratio r in (0,1)
        down_sample_ratio_min: Minimum retention ratio floor
        filter_position_zero: Whether to filter out position 0 from candidates

    Returns:
        Tuple of:
        - all_indices: Flat tensor of encoded indices (depth * seq_length + pos)
          for all depths [total_sampled_length]
        - depth_ids: Per-element depth assignment [total_sampled_length]
        - num_depths_used: Actual number of depths produced
    """
    loss_mask = loss_mask.squeeze()
    device = loss_mask.device
    all_valid_indices = torch.where(loss_mask == 1)[0]

    sample_indices = [torch.arange(seq_length, device=device)]
    prev_indices = all_valid_indices

    for depth in range(1, num_depths):
        valid_length = max(0, all_valid_indices.shape[0] - depth)
        ratio = max(down_sample_ratio**depth, down_sample_ratio_min)
        sample_size = int(valid_length * ratio)

        if sample_size <= 0:
            break

        if prev_indices.shape[0] >= sample_size:
            random_selection = torch.randperm(prev_indices.shape[0], device=device)[
                :sample_size
            ]
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

    num_depths_used = len(sample_indices)

    all_indices = torch.cat(
        [depth * seq_length + idx for depth, idx in enumerate(sample_indices)]
    )
    depth_ids = torch.cat(
        [
            torch.full((idx.shape[0],), depth, device=device, dtype=torch.long)
            for depth, idx in enumerate(sample_indices)
        ]
    )

    return all_indices, depth_ids, num_depths_used
