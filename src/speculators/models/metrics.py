"""Loss-independent prefix agreement with stored continuation tokens.

Adapters pass already-computed draft IDs and their reference positions. No
teacher projection or probability distribution is needed by this metric.
"""

import torch


@torch.no_grad()
def compute_reference_prefix_metrics(
    pred_ids: torch.Tensor,  # [batch, starts, horizon], in draft vocabulary
    input_ids: torch.Tensor,  # [batch, sequence], in verifier vocabulary
    first_target: torch.Tensor,  # [starts] or [batch, starts]
    loss_mask: torch.Tensor | None,
    document_ids: torch.Tensor | None,
    *,
    d2t: torch.Tensor | None = None,
    observed: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Count complete matching prefixes at every configured draft position.

    Zero-based position i uses starts with i+1 observed, supervised references
    in the anchor's document. Eligibility is independent of prediction matches.
    ``d2t`` stores offsets: verifier_id = draft_id + d2t[draft_id].
    """
    horizon = pred_ids.shape[-1]
    seq_len = input_ids.shape[1]
    if seq_len == 0 or horizon == 0:
        valid = matches = torch.zeros_like(pred_ids, dtype=torch.bool)
    else:
        positions = first_target.unsqueeze(-1) + torch.arange(
            horizon, device=pred_ids.device
        )
        positions = positions.expand_as(pred_ids)
        safe_positions = positions.clamp(0, seq_len - 1).flatten(1)
        references = input_ids.gather(1, safe_positions).reshape_as(pred_ids)
        valid = (first_target.unsqueeze(-1) > 0) & (positions < seq_len)
        if loss_mask is not None:
            valid &= loss_mask.gather(1, safe_positions).reshape_as(pred_ids).bool()
        if document_ids is not None:
            docs = document_ids.gather(1, safe_positions).reshape_as(pred_ids)
            anchor_positions = (positions[..., 0] - 1).clamp(0, seq_len - 1)
            anchor_docs = document_ids.gather(1, anchor_positions).unsqueeze(-1)
            valid &= (docs == anchor_docs) & (anchor_docs >= 0)
        if observed is not None:
            valid &= observed
        if d2t is not None:
            pred_ids = pred_ids + d2t[pred_ids]
        matches = pred_ids.eq(references)

    prefix_valid = valid.cumprod(dim=-1).bool()
    prefix_correct = (matches & valid).float().cumprod(dim=-1)
    correct = prefix_correct.sum(dim=(0, 1))
    total = prefix_valid.float().sum(dim=(0, 1))
    return {
        f"reference_acc_at_pos_{i}_{kind}": values[i]
        for i in range(horizon)
        for kind, values in (("sum", correct), ("total", total))
    }


@torch.no_grad()
def compute_block_reference_metrics(
    pred_ids: torch.Tensor,
    input_ids: torch.Tensor,
    block_indices: torch.Tensor,
    block_mask: torch.Tensor,
    loss_mask: torch.Tensor,
    document_ids: torch.Tensor,
    block_size: int,
    sample_from_anchor: bool,
    d2t: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Align DFlash-family proposals to the token after each input anchor."""
    start = 0 if sample_from_anchor else 1
    horizon = block_size - start
    predictions = pred_ids.reshape(input_ids.shape[0], -1, block_size)
    block_valid = block_mask.reshape_as(predictions).bool().any(dim=-1, keepdim=True)
    return compute_reference_prefix_metrics(
        predictions[..., start : start + horizon],
        input_ids,
        block_indices.reshape(-1, block_size)[:, 0] + 1,
        loss_mask,
        document_ids,
        d2t=d2t,
        observed=block_valid,
    )


@torch.no_grad()
def compute_sampled_reference_metrics(
    pred_ids: torch.Tensor,
    input_ids: torch.Tensor,
    anchor_pos: torch.Tensor,
    depth: torch.Tensor,
    num_depths: int,
    loss_mask: torch.Tensor,
    document_ids: torch.Tensor,
    d2t: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Align P-EAGLE samples by start and depth, marking missing predictions."""
    horizon = num_depths
    if horizon == 0:
        return {}
    batch_size, seq_len = input_ids.shape
    # The extra slot absorbs samples whose anchor is outside the sequence. Real
    # (anchor, depth) pairs are unique in COD sampling, even when shuffled.
    eligible = (anchor_pos >= 0) & (anchor_pos < seq_len) & (depth < horizon)
    indices = torch.where(eligible, anchor_pos * horizon + depth, seq_len * horizon)
    indices = indices.unsqueeze(0).expand_as(pred_ids)
    predictions = pred_ids.new_zeros(batch_size, seq_len * horizon + 1)
    observed = torch.zeros_like(predictions, dtype=torch.bool)
    predictions.scatter_(1, indices, pred_ids)
    observed.scatter_(1, indices, eligible.unsqueeze(0).expand_as(pred_ids))
    return compute_reference_prefix_metrics(
        predictions[:, :-1].reshape(batch_size, seq_len, horizon),
        input_ids,
        torch.arange(seq_len, device=input_ids.device) + 1,
        loss_mask,
        document_ids,
        d2t=d2t,
        observed=observed[:, :-1].reshape(batch_size, seq_len, horizon),
    )
