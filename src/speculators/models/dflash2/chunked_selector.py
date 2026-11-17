"""Chunked candidate selection for DFlash2's selector.

``CandidateSelector`` needs three things from the full-vocab projections
that the rest of the DFlash family has already learned to avoid
materializing (see ``losses.fused_linear``), none of which ``fused_linear``
itself provides (it returns only the top-1 argmax, and only for the scalar
losses it already computes):

- the draft side's real top-``k`` ids (``unary_logits.topk(top_k).indices``)
  -- selecting them needs no gradient, exactly as in the materialized path,
  where only ``.indices`` is ever kept;
- the draft side's logit *values* at a given set of candidate ids
  (``unary_logits.gather(-1, training_candidate_ids)``) -- differentiable.
  This is a **different** id set from the one above: ``training_candidate_ids``
  may have its last slot replaced with an injected target id when the
  top-k miss policy is ``"replace"``, so the gradient-bearing gather must
  happen at that (possibly modified) id set, not at the plain top-k;
- the target logit values at those same candidate ids, plus the full-vocab
  log-normalizer (``targets.gather(-1, candidate_ids)`` and
  ``logsumexp(targets, dim=-1)``) -- both ``no_grad``, used only for the
  ``candidate_mass`` metric.

All three are computed one row chunk at a time, exactly like
``fused_linear.fused_linear_compound``, so neither the draft nor the target
``[N, V]`` tile is ever held for more than one chunk.

The gradient-bearing gather's backward is a gather/scatter against the
selected weight rows -- the same shape as an embedding backward, not a GEMM
-- so, like ``fused_linear``'s deferred mode, backward here does no extra
projection work.
"""

from __future__ import annotations

import torch

__all__ = ["chunked_gather_values", "chunked_target_gather", "chunked_topk_ids"]

# Deliberately not importing fused_linear.chunk_rows_for: fused_linear.py
# pulls in fused.py, which requires Triton at import time even for this kind
# of plain-Python sizing helper, which would block CPU-only testing of a
# module that itself has no Triton dependency. Same budget convention,
# reimplemented standalone; nothing here holds a dz/jac tile, so there is no
# recompute_targets/dz_tiles term to carry over.
_BUDGET_MIB = 4096
_MIN_CHUNK_ROWS = 64
_MAX_CHUNK_ROWS = 2048


def _default_chunk_rows(vocab: int, dtype: torch.dtype, rows: int) -> int:
    """Rows per chunk: the largest power of two whose ``[chunk_rows, V]``
    scratch tile fits ``_BUDGET_MIB``, capped at ``_MAX_CHUNK_ROWS``."""
    budget = _BUDGET_MIB * 1024**2
    per_row = vocab * torch.empty((), dtype=dtype).element_size()
    fit = max(budget // per_row, 1)
    chunk = max(1 << (fit.bit_length() - 1), _MIN_CHUNK_ROWS)
    return min(chunk, _MAX_CHUNK_ROWS, max(rows, 1))


@torch.no_grad()
def chunked_topk_ids(
    hidden: torch.Tensor,  # [N, H]
    weight: torch.Tensor,  # [V, H]
    top_k: int,
    chunk_rows: int | None = None,
) -> torch.Tensor:
    """Top-``k`` ids of ``hidden @ weight.T``, chunked, no gradient.

    Mirrors the materialized path's ``unary_logits.topk(top_k).indices`` --
    only the indices are ever used there too (``.values`` is discarded), so
    this needs no backward.
    """
    rows, vocab = hidden.shape[0], weight.shape[0]
    if chunk_rows is None:
        chunk_rows = _default_chunk_rows(vocab, hidden.dtype, rows)
    ids_chunks = []
    for start in range(0, rows, chunk_rows):
        end = min(start + chunk_rows, rows)
        z_chunk = hidden[start:end] @ weight.T  # [chunk_rows, V], transient
        ids_chunks.append(z_chunk.topk(min(top_k, vocab), dim=-1).indices)
    return torch.cat(ids_chunks, dim=0)


class _ChunkedGatherValues(torch.autograd.Function):
    """``(hidden @ weight.T).gather(-1, ids)``, chunked over rows.

    Forward holds one chunk's ``[chunk_rows, V]`` tile at a time. Backward
    never recomputes it: the gradient for a gathered value is exactly the
    weight row it came from, so only ``(ids, hidden, weight)`` need to
    survive from forward.
    """

    @staticmethod
    def forward(ctx, hidden, weight, ids, chunk_rows):
        rows = hidden.shape[0]
        out_chunks = []
        for start in range(0, rows, chunk_rows):
            end = min(start + chunk_rows, rows)
            z_chunk = hidden[start:end] @ weight.T  # [chunk_rows, V], transient
            out_chunks.append(z_chunk.gather(-1, ids[start:end]))
        ctx.save_for_backward(hidden, weight, ids)
        ctx.chunk_rows = chunk_rows
        return torch.cat(out_chunks, dim=0)

    @staticmethod
    def backward(ctx, grad_out):
        hidden, weight, ids = ctx.saved_tensors
        rows, h = hidden.shape
        chunk_rows = ctx.chunk_rows

        d_hidden = torch.zeros_like(hidden)
        d_weight = torch.zeros_like(weight)
        for start in range(0, rows, chunk_rows):
            end = min(start + chunk_rows, rows)
            ids_c = ids[start:end]  # [c, k]
            grad_c = grad_out[start:end]  # [c, k]
            hidden_c = hidden[start:end]  # [c, h]

            w_sel = weight[ids_c]  # [c, k, h] -- gather, no [c, V] materialized
            d_hidden[start:end] = torch.einsum("ck,ckh->ch", grad_c, w_sel)
            contrib = grad_c.unsqueeze(-1) * hidden_c.unsqueeze(1)  # [c, k, h]
            d_weight.index_add_(0, ids_c.reshape(-1), contrib.reshape(-1, h))
        return d_hidden, d_weight, None, None


def chunked_gather_values(
    hidden: torch.Tensor,  # [N, H]
    weight: torch.Tensor,  # [V, H]
    ids: torch.Tensor,  # [N, k]
    chunk_rows: int | None = None,
) -> torch.Tensor:
    """Differentiable ``(hidden @ weight.T).gather(-1, ids)``, chunked so the
    full ``[N, V]`` projection is never materialized.

    Args:
        hidden: ``[N, H]`` final (post-norm) draft hidden states.
        weight: ``[V, H]`` draft ``lm_head`` weight.
        ids: ``[N, k]`` ids to gather at -- e.g. ``training_candidate_ids``,
            which may differ from whatever selected ``ids`` (see
            ``chunked_topk_ids``) at positions where a missing target was
            injected.
        chunk_rows: rows per chunk; ``None`` sizes chunks the same way
            ``fused_linear_compound`` does.

    Returns:
        ``[N, k]``, carrying a gradient back to ``hidden`` and ``weight``.
    """
    if chunk_rows is None:
        chunk_rows = _default_chunk_rows(weight.shape[0], hidden.dtype, hidden.shape[0])
    return _ChunkedGatherValues.apply(hidden, weight, ids, chunk_rows)


@torch.no_grad()
def chunked_target_gather(
    targets: torch.Tensor,  # [N, V] materialized, or [N, H_t] if target_weight given
    target_weight: torch.Tensor | None,  # [V, H_t], or None if targets are [N, V]
    candidate_ids: torch.Tensor,  # [N, top_k]
    chunk_rows: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Target logit values at ``candidate_ids``, and the full-vocab
    log-normalizer, chunked, no gradient.

    For ``candidate_mass``
    (``exp(logsumexp(targets.gather(-1, candidate_ids)) - logsumexp(targets))``),
    which needs no gradient since the verifier head is frozen. With
    ``target_weight`` (the verifier ``lm_head``), ``targets`` holds the
    post-norm verifier hidden states and each chunk's target logits are
    projected on the fly; otherwise ``targets`` is already ``[N, V]`` and
    this only chunks the gather/logsumexp themselves (matching
    ``fused_linear_targets="materialize"``) so neither ever makes a second
    full-vocab fp32 copy. Both outputs come from the same per-chunk tile.

    Returns:
        ``(gathered, log_normalizer)``: ``gathered`` is ``[N, top_k]``,
        ``targets.gather(-1, candidate_ids)``; ``log_normalizer`` is ``[N]``,
        ``logsumexp(targets, dim=-1)`` in fp32.
    """
    rows = targets.shape[0]
    vocab = target_weight.shape[0] if target_weight is not None else targets.shape[1]
    if chunk_rows is None:
        chunk_rows = _default_chunk_rows(vocab, targets.dtype, rows)
    gathered_chunks, lognorm_chunks = [], []
    for start in range(0, rows, chunk_rows):
        end = min(start + chunk_rows, rows)
        chunk = targets[start:end]
        if target_weight is not None:
            chunk = chunk @ target_weight.T  # transient target-logit tile
        gathered_chunks.append(chunk.gather(-1, candidate_ids[start:end]))
        lognorm_chunks.append(torch.logsumexp(chunk.float(), dim=-1))
    return torch.cat(gathered_chunks, dim=0), torch.cat(lognorm_chunks, dim=0)
