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
    document_ids: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
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
        document_ids: Optional document ids, 1-D or [1, seq], -1 for padding.
            When set, a depth>0 sample is kept only if its anchor and target
            share a non-padding document, and the chain does not step into
            another document. When None, document identity is not checked.

    Returns:
        Tuple of:
            anchor_pos: The starting position in the original sequence the current
                sampling chain started from.
            depth: Which COD sampling round each element belongs to
    """
    loss_mask = loss_mask.squeeze(0)
    device = loss_mask.device
    all_valid_indices = torch.where(loss_mask == 1)[0]
    doc_ids = None
    if document_ids is not None:
        doc_ids = document_ids if document_ids.ndim == 1 else document_ids.squeeze(0)
        doc_ids = doc_ids.to(device)

    if max_anchors is not None and all_valid_indices.shape[0] > max_anchors:
        perm = torch.randperm(all_valid_indices.shape[0], device=device)[:max_anchors]
        seed_indices = all_valid_indices[perm].sort()[0]
    else:
        seed_indices = all_valid_indices

    sample_indices = [torch.arange(seq_length, device=device)]
    n_per_depth = [seq_length]
    prev_indices = seed_indices

    for d in range(1, num_depths):
        valid_length = max(0, seed_indices.shape[0] - d)
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

        # Anchor is sampled_idx - d. Drop targets that would start before 0,
        # and do not let them seed the next depth.
        sampled_idx = sampled_idx[sampled_idx >= d]
        if doc_ids is not None and sampled_idx.numel() > 0:
            anchor_docs = doc_ids[sampled_idx - d]
            target_docs = doc_ids[sampled_idx]
            same_document = (
                (anchor_docs == target_docs) & (anchor_docs != -1) & (target_docs != -1)
            )
            sampled_idx = sampled_idx[same_document]
        if sampled_idx.numel() == 0:
            break

        # Next-token targets stay inside the sequence; do not wrap to 0.
        next_candidates = sampled_idx + 1
        in_sequence = next_candidates < seq_length
        next_candidates = next_candidates[in_sequence]
        next_sources = sampled_idx[in_sequence]
        if filter_position_zero:
            keep = next_candidates != 0
            next_candidates = next_candidates[keep]
            next_sources = next_sources[keep]
        if doc_ids is not None and next_candidates.numel() > 0:
            same_document = doc_ids[next_candidates] == doc_ids[next_sources]
            next_candidates = next_candidates[same_document]
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
