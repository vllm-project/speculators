"""COD sampling logic for P-EAGLE parallel group generation."""

import torch


def generate_cod_sample_indices(
    seq_length: int,
    loss_mask: torch.Tensor,
    num_depths: int = 8,
    down_sample_ratio: float = 0.7,
    down_sample_ratio_min: float = 0.2,
    filter_position_zero: bool = True,
    max_anchors: int | None = None,
    max_context_window: int = 4096,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Generate sampling indices for parallel sequences using COD sampling.

    Conditional-On-Distribution (COD) sampling reduces memory by using geometric
    decay: depth 0 retains all n positions, depth 1 retains n*r positions,
    depth 2 retains n*r^2 positions, etc.

    When max_anchors is set, selects a contiguous window of the original sequence
    containing up to max_anchors valid (loss_mask=1) positions as COD starting
    points. The window preserves all intervening tokens (including prompts)
    to maintain context.

    Args:
        seq_length: Length of the sequence
        loss_mask: Binary mask indicating valid training positions [batch, seq_len]
        num_depths: Number of parallel prediction groups (K)
        down_sample_ratio: Geometric decay ratio r in (0,1)
        down_sample_ratio_min: Minimum retention ratio floor
        filter_position_zero: Whether to filter out position 0 from candidates
        max_anchors: Maximum number of COD chain starting points. None means
            use all positions.
        max_context_window: Hard cap on contiguous window size to prevent
            sparse loss masks from reinflating the window.

    Returns:
        Tuple of:
            anchor_pos: The starting position in the original sequence the current
                sampling chain started from.
            depth: Which COD sampling round each element belongs to
    """
    loss_mask = loss_mask.squeeze(0)
    device = loss_mask.device
    all_valid_indices = torch.where(loss_mask == 1)[0]

    if max_anchors is not None and all_valid_indices.shape[0] > 0:
        if all_valid_indices.shape[0] > max_anchors:
            n_valid = all_valid_indices.shape[0]
            start_idx = int(torch.randint(0, n_valid - max_anchors + 1, (1,)).item())
            selected_valid = all_valid_indices[start_idx : start_idx + max_anchors]
        else:
            selected_valid = all_valid_indices

        window_start = int(selected_valid[0].item())
        window_end = int(selected_valid[-1].item()) + 1

        if (window_end - window_start) > max_context_window:
            window_end = min(window_start + max_context_window, seq_length)
            selected_valid = selected_valid[
                (selected_valid >= window_start) & (selected_valid < window_end)
            ]
        all_valid_indices = selected_valid

        sample_indices = [torch.arange(window_start, window_end, device=device)]
        n_per_depth = [window_end - window_start]
    else:
        sample_indices = [torch.arange(seq_length, device=device)]
        n_per_depth = [seq_length]

    prev_indices = all_valid_indices

    for d in range(1, num_depths):
        valid_length = max(0, all_valid_indices.shape[0] - d)
        ratio = max(down_sample_ratio**d, down_sample_ratio_min)
        sample_size = int(valid_length * ratio)

        if sample_size <= 0:
            break

        # Subsample from candidate pool, or keep all if pool is too small
        if prev_indices.shape[0] >= sample_size:
            random_selection = torch.randperm(prev_indices.shape[0], device=device)[
                :sample_size
            ]
            sampled_idx = prev_indices[random_selection]
            sampled_idx = torch.sort(sampled_idx)[0]  # restore causal order
        else:
            sampled_idx = prev_indices

        # Build candidate pool for next depth: shift by +1 (next-token targets),
        next_candidates = (sampled_idx + 1) % seq_length
        if filter_position_zero:
            next_candidates = next_candidates[next_candidates != 0]
        mask = torch.isin(next_candidates, all_valid_indices)
        prev_indices = next_candidates[mask]

        sample_indices.append(sampled_idx - d)
        n_per_depth.append(sampled_idx.shape[0])

    anchor_pos = torch.cat(sample_indices)
    depth = torch.cat(
        [
            torch.full((n,), i, device=device, dtype=torch.long)
            for i, n in enumerate(n_per_depth)
        ]
    )

    return anchor_pos, depth
