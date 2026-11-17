"""Teacher forcing and iterative consistency training for the XPress refiner."""

import torch

from speculators.losses import LossConfig, resolve_loss_config
from speculators.models.xpress.metrics import compute_metrics


def draft_to_verifier_ids(model, draft_ids: torch.Tensor) -> torch.Tensor:
    if model.d2t is not None:
        return draft_ids + model.d2t[draft_ids]
    return draft_ids


def xpress_objective(
    model,
    hidden_states: torch.Tensor,  # [1, total_seq_len, num_target_layers * hidden_size]
    input_ids: torch.Tensor,  # [1, total_seq_len]
    loss_mask: torch.Tensor,  # [1, total_seq_len]
    verifier_last_hidden_states: torch.Tensor,  # [1, total_seq_len, hidden_size]
    document_ids: torch.Tensor,  # [1, total_seq_len]
    position_ids: torch.Tensor | None = None,  # [1, total_seq_len]
    loss_config: LossConfig | None = None,
    gamma: float = 4.0,
    max_anchors: int = 3072,
    per_position_loss_weight: str = "fixed-exp-decay",
    dpace_alpha: float = 0.5,
    consistency_weight: float = 0.3,
    consistency_passes: int = 3,
    base_anchor_weight: float | torch.Tensor = 0.6,
    base_anchor_full_weight: bool = False,
    ce_from_data: bool = False,
    **kwargs,
):
    features = model.draft_features(
        hidden_states,
        input_ids,
        loss_mask,
        document_ids,
        position_ids,
        max_anchors=max_anchors,
        **kwargs,
    )
    hidden, logits, layout = features
    aligned_loss_mask = layout.loss_mask
    anchored_block_indices = layout.block_positions
    targets = model._build_targets(  # noqa: SLF001
        verifier_last_hidden_states, anchored_block_indices
    )

    num_blocks = max_anchors
    block = model.block_size
    mask_tokens_size = num_blocks * block
    block_tokens = input_ids[0, anchored_block_indices].view(num_blocks, block)
    anchor_tokens = block_tokens[:, :1]
    anchor_pos = anchored_block_indices.view(num_blocks, block)[:, 0]
    am1_pos = (anchor_pos - 1).clamp(min=0)
    am1_ok = (anchor_pos > 0) & (
        document_ids[0, am1_pos] == document_ids[0, anchor_pos]
    )
    am1_tokens = torch.where(
        am1_ok, input_ids[0, am1_pos], anchor_tokens[:, 0]
    ).unsqueeze(1)

    data_labels = None
    if model.d2t is None:
        if model.config.sample_from_anchor:
            nxt = (anchored_block_indices + 1).clamp(max=input_ids.shape[1] - 1)
            data_labels = input_ids[0, nxt].view(1, mask_tokens_size)
        else:
            data_labels = block_tokens.reshape(1, mask_tokens_size)

    # ce_from_data: CE labels = the DATA tokens, not the teacher
    # argmax. Requires the full verifier vocab.
    ce_data_labels = None
    if ce_from_data:
        if data_labels is None:
            raise ValueError(
                "ce_from_data requires draft_vocab == verifier_vocab "
                "(XPress recipes are full-vocab)"
            )
        ce_data_labels = data_labels
    if model.config.sample_from_anchor:
        # Slot k predicts token p+k+1; teacher prev for slot k = token p+k.
        prev_tf = block_tokens
    else:
        # Fill-in: slot k predicts token p+k (slot 0 = anchor); teacher prev
        # for slot k = token p+k-1 (shifted).
        prev_tf = torch.cat([am1_tokens, block_tokens[:, :-1]], dim=1)

    hidden_blocks = hidden.view(num_blocks, block, -1)
    base_block_logits = logits.view(num_blocks, block, -1)
    hcache = model.correction.hidden_cache(
        hidden_blocks.to(model.correction.down_h.weight.dtype)
    )

    # Teacher-forced refine pass.
    bias_tf = model.correction.block_bias(prev_tf, hcache=hcache)
    logits_tf = (base_block_logits + bias_tf).view(1, mask_tokens_size, -1)

    round_logits: list[torch.Tensor] = []
    if consistency_weight > 0.0 and consistency_passes > 0:
        pred = draft_to_verifier_ids(model, base_block_logits.detach().argmax(dim=-1))
        for _ in range(int(consistency_passes)):
            if model.config.sample_from_anchor:
                prev_j = torch.cat([anchor_tokens, pred[:, :-1]], dim=1)
            else:
                pred[:, :1] = anchor_tokens  # slot 0 is the anchor, not predicted
                prev_j = torch.cat([am1_tokens, pred[:, :-1]], dim=1)
            bias_j = model.correction.block_bias(prev_j, hcache=hcache)
            refined_j = base_block_logits + bias_j
            round_logits.append(refined_j.view(1, mask_tokens_size, -1))
            pred = draft_to_verifier_ids(model, refined_j.detach().argmax(dim=-1))

    # The trainer's annealing schedule passes a 0-dim tensor so its value stays
    # out of Dynamo's guards (a per-step float would recompile this frame until it
    # hit recompile_limit and fell back to eager). Comparing a tensor here would
    # make the branch data-dependent, so a tensor simply means "enabled" -- which
    # holds by construction, and if the schedule's floor were 0 the anchor term
    # would contribute exactly 0 anyway.
    base_anchor_on = (
        True
        if isinstance(base_anchor_weight, torch.Tensor)
        else base_anchor_weight > 0.0
    )

    loss, metrics = compute_metrics(
        logits_tf,
        round_logits,
        logits if base_anchor_on else None,
        targets,
        aligned_loss_mask,
        model.block_size,
        loss_config=loss_config or resolve_loss_config('{"ce": 0.1, "tv": 0.9}'),
        gamma=gamma,
        consistency_weight=consistency_weight,
        base_anchor_weight=base_anchor_weight,
        base_anchor_full_weight=base_anchor_full_weight,
        per_position_loss_weight=per_position_loss_weight,
        dpace_alpha=dpace_alpha,
        sample_from_anchor=model.config.sample_from_anchor,
        ce_data_labels=ce_data_labels,
        data_labels=data_labels,
    )
    return None, loss, metrics  # loss: []
