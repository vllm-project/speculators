"""Fused linear losses: drafter losses straight from hidden states.

The draft logits ``hidden @ weight.T`` are computed one row chunk at a time
into an fp32 scratch tile, reduced by the multi-term vocab kernel from
``losses.fused``, and dropped; backward recomputes each chunk's logits and
writes ``d hidden = dz @ weight`` chunk by chunk. Neither the ``[N, V]``
draft logits nor their ``[N, V]`` gradient ever exist, and the logits reach
the reduction in fp32 straight from the GEMM accumulator (``out_dtype``)
instead of rounded to bf16.

The per-row loss contract is unchanged: the op returns unreduced per-term
losses and takes a per-row ``grad_output`` per term, so masking, decay,
D-PACE and term weights stay in ``losses.utils``.

Two ``torch.library.custom_op``s (forward and backward) keep the chunk loop
opaque to torch.compile: traced, it would be unrolled per chunk count and
recompile whenever the row count changes. Per-row statistics are returned
as an output and saved in ``setup_context``.

Targets come in either as materialized ``[N, V]`` logits, or as the
(post-norm) verifier hidden states ``[N, H_t]`` plus the frozen verifier
``lm_head`` weight, in which case the target logits are recomputed per chunk
the same way and no ``[N, V]`` tensor exists at all. Targets receive no
gradient. The draft weight must be frozen; trainable draft weights are not
supported yet.
"""

import os
from collections.abc import Sequence

import torch

from speculators.losses import fused

__all__ = ["chunk_rows_for", "fused_linear_compound"]

# Peak scratch the chunk loop may use, in MiB. Measured on MI355X: chunks of
# <= 256 rows cost 1.5-4.6x a full GEMM at V = 152k and 1024 rows 1.2-1.45x,
# 2048 rows about 1.18x (PR 0 gemm-sweep). 4 GiB gives 2048-row chunks with
# recomputed targets (10 B per row and vocab entry at bf16): 16k x 110k
# loss head 165 -> 140 ms vs 2 GiB, still ~0.65x one [N, V] tensor.
_BUDGET_ENV = "SPECULATORS_FUSED_LINEAR_BUDGET_MIB"
_DEFAULT_BUDGET_MIB = 4096
_MIN_CHUNK_ROWS = 64
_LOG2 = 0.6931471805599453


def chunk_rows_for(
    vocab: int, dtype: torch.dtype, rows: int, recompute_targets: bool = False
) -> int:
    """Rows per chunk: the largest power of two whose scratch fits the budget.

    Per row the loop holds an fp32 draft-logits row, an fp32 target-logits
    row when targets are recomputed, and, in backward, an input-dtype ``dz``
    row.
    """
    budget = int(os.environ.get(_BUDGET_ENV, _DEFAULT_BUDGET_MIB)) * 1024**2
    target_bytes = 4 if recompute_targets else 0
    per_row = vocab * (4 + target_bytes + torch.empty((), dtype=dtype).element_size())
    fit = max(budget // per_row, 1)
    chunk = max(1 << (fit.bit_length() - 1), _MIN_CHUNK_ROWS)
    return min(chunk, max(rows, 1))


def _out_dtype(hidden: torch.Tensor) -> torch.dtype:
    """Loss / stats dtype: fp32, or fp64 for fp64 inputs (CPU gradcheck)."""
    return torch.promote_types(torch.float32, hidden.dtype)


def _project(hidden: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """``hidden @ weight.T`` with an fp32 (or wider) result from the accumulator.

    Autocast is disabled: forward and backward must compute bit-identical
    logits from the operands as given (the wrapper casts them for autocast),
    or the saved softmax statistics would not match the recomputed logits.
    """
    with torch.autocast(hidden.device.type, enabled=False):
        if hidden.dtype in (torch.float16, torch.bfloat16):
            return torch.mm(hidden, weight.t(), out_dtype=torch.float32)
        return hidden @ weight.t()


def _chunks(rows: int, chunk_rows: int):
    for start in range(0, rows, chunk_rows):
        yield start, min(start + chunk_rows, rows)


# ---------------------------------------------------------------------------
# Torch reference of one chunk (CPU / fp64): same math and stats layout as the
# Triton kernels in losses.fused.
# ---------------------------------------------------------------------------


def _ref_softmax_stats(z: torch.Tensor):
    m = z.max(dim=-1).values
    s = torch.exp(z - m[:, None]).sum(dim=-1)
    lse = m + torch.log(s)
    return m, s, z - lse[:, None]


def _ref_forward_chunk(z, targets, terms, losses, argmax, stats):
    t = targets.to(z.dtype)
    m_d, s_d, ldp = _ref_softmax_stats(z)
    dp = torch.exp(ldp)
    stats[0], stats[1] = m_d, s_d
    argmax[0] = z.argmax(dim=-1)
    argmax[1] = t.argmax(dim=-1)
    if terms & fused.TERM_BITS["ce"]:
        idx = argmax[1]
        losses[3] = m_d + torch.log(s_d) - z.gather(-1, idx[:, None])[:, 0]
        stats[6] = idx.to(stats.dtype)
    if not terms & fused.DIST_MASK:
        return
    m_t, s_t, ltp = _ref_softmax_stats(t)
    tp = torch.exp(ltp)
    stats[2], stats[3] = m_t, s_t
    if terms & fused.TERM_BITS["kl_div"]:
        losses[0] = (tp * (ltp - ldp)).sum(dim=-1)
    if terms & fused.TERM_BITS["rkl"]:
        losses[1] = stats[4] = (dp * (ldp - ltp)).sum(dim=-1)
    if terms & fused.TERM_BITS["jsd"]:
        lmp = torch.logaddexp(ldp, ltp) - _LOG2
        jsd_draft = (dp * (ldp - lmp)).sum(dim=-1)
        losses[2] = 0.5 * (jsd_draft + (tp * (ltp - lmp)).sum(dim=-1))
        stats[5] = jsd_draft
    if terms & fused.TERM_BITS["tv"]:
        losses[4] = 1.0 - torch.minimum(dp, tp).sum(dim=-1)
        stats[7] = torch.where(dp <= tp, dp, 0.0).sum(dim=-1)


def _ref_backward_chunk(z, targets, grad_out, stats, terms) -> torch.Tensor:
    ldp = z - (stats[0] + torch.log(stats[1]))[:, None]
    dp = torch.exp(ldp)
    grad = torch.zeros_like(z)
    if terms & fused.TERM_BITS["ce"]:
        onehot = torch.zeros_like(z).scatter_(-1, stats[6].long()[:, None], 1.0)
        grad += grad_out[3][:, None] * (dp - onehot)
    if terms & fused.DIST_MASK:
        ltp = targets.to(z.dtype) - (stats[2] + torch.log(stats[3]))[:, None]
        tp = torch.exp(ltp)
        if terms & fused.TERM_BITS["kl_div"]:
            grad += grad_out[0][:, None] * (dp - tp)
        if terms & fused.TERM_BITS["rkl"]:
            grad += grad_out[1][:, None] * (dp * (ldp - ltp - stats[4][:, None]))
        if terms & fused.TERM_BITS["jsd"]:
            lmp = torch.logaddexp(ldp, ltp) - _LOG2
            grad += grad_out[2][:, None] * (0.5 * dp * (ldp - lmp - stats[5][:, None]))
        if terms & fused.TERM_BITS["tv"]:
            grad += grad_out[4][:, None] * (
                -dp * ((dp <= tp).to(z.dtype) - stats[7][:, None])
            )
    return grad


# ---------------------------------------------------------------------------
# Custom ops
# ---------------------------------------------------------------------------


def _target_chunk(
    targets: torch.Tensor,
    target_weight: torch.Tensor | None,
    start: int,
    end: int,
) -> torch.Tensor:
    """Target logits for rows ``start:end``: a slice of the materialized
    logits, or recomputed from the verifier hidden states (fp32 tile)."""
    rows = targets[start:end]
    return rows if target_weight is None else _project(rows, target_weight)


@torch.library.custom_op("speculators::fused_linear_loss_fwd", mutates_args=())
def _fused_linear_fwd(
    hidden: torch.Tensor,
    weight: torch.Tensor,
    targets: torch.Tensor,
    target_weight: torch.Tensor | None,
    terms: int,
    grad_terms: int,
    chunk_rows: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Per-term losses ``[_N_TERMS, N]``, argmax ``[2, N]`` and stats ``[_N_STATS, N]``.

    ``targets`` is ``[N, V]`` logits, or ``[N, H_t]`` verifier hidden states
    when ``target_weight`` (``[V, H_t]``) is given. ``grad_terms`` (a subset
    of ``terms``) is only consumed by the autograd registration; slots of
    terms outside ``terms`` are zero.
    """
    del grad_terms
    rows, vocab = hidden.shape[0], weight.shape[0]
    dtype = _out_dtype(hidden)
    losses = hidden.new_zeros((fused.N_TERMS, rows), dtype=dtype)
    argmax = hidden.new_zeros((2, rows), dtype=torch.int64)
    stats = hidden.new_zeros((fused.N_STATS, rows), dtype=dtype)
    for start, end in _chunks(rows, chunk_rows):
        z = _project(hidden[start:end], weight)
        chunk_targets = _target_chunk(targets, target_weight, start, end)
        if hidden.device.type == "cpu":
            _ref_forward_chunk(
                z,
                chunk_targets,
                terms,
                losses[:, start:end],
                argmax[:, start:end],
                stats[:, start:end],
            )
        else:
            fused.loss_forward_kernel[(end - start,)](
                z,
                chunk_targets,
                losses[:, start:],
                losses.stride(0),
                argmax[:, start:],
                argmax.stride(0),
                stats[:, start:],
                stats.stride(0),
                vocab,
                TERMS=terms,
            )
        # Release the tiles before the next chunk allocates its own.
        del z, chunk_targets
    return losses, argmax, stats


@_fused_linear_fwd.register_fake
def _(hidden, weight, targets, target_weight, terms, grad_terms, chunk_rows):  # noqa: ARG001
    rows = hidden.shape[0]
    dtype = _out_dtype(hidden)
    return (
        hidden.new_empty((fused.N_TERMS, rows), dtype=dtype),
        hidden.new_empty((2, rows), dtype=torch.int64),
        hidden.new_empty((fused.N_STATS, rows), dtype=dtype),
    )


@torch.library.custom_op("speculators::fused_linear_loss_bwd", mutates_args=())
def _fused_linear_bwd(
    grad_losses: torch.Tensor,
    hidden: torch.Tensor,
    weight: torch.Tensor,
    targets: torch.Tensor,
    target_weight: torch.Tensor | None,
    stats: torch.Tensor,
    grad_terms: int,
    chunk_rows: int,
) -> torch.Tensor:
    """``d hidden``: per chunk, recompute the logits, form ``dz``, ``dz @ weight``.

    Target logits are only needed (sliced or recomputed) for distribution
    terms; CE reads its cached target argmax from ``stats``.
    """
    rows, vocab = hidden.shape[0], weight.shape[0]
    needs_targets = bool(grad_terms & fused.DIST_MASK)
    d_hidden = torch.empty_like(hidden)
    for start, end in _chunks(rows, chunk_rows):
        z = _project(hidden[start:end], weight)
        chunk_targets = (
            _target_chunk(targets, target_weight, start, end)
            if needs_targets
            else targets[:0]
        )
        if hidden.device.type == "cpu":
            dz = _ref_backward_chunk(
                z,
                chunk_targets,
                grad_losses[:, start:end],
                stats[:, start:end],
                grad_terms,
            ).to(hidden.dtype)
        else:
            dz = torch.empty((end - start, vocab), device=z.device, dtype=hidden.dtype)
            fused.loss_backward_kernel[(end - start,)](
                z,
                chunk_targets,
                dz,
                grad_losses[:, start:],
                grad_losses.stride(0),
                stats[:, start:],
                stats.stride(0),
                vocab,
                TERMS=grad_terms,
            )
        del z, chunk_targets
        with torch.autocast(hidden.device.type, enabled=False):
            torch.mm(dz, weight, out=d_hidden[start:end])
        del dz
    return d_hidden


@_fused_linear_bwd.register_fake
def _(grad_losses, hidden, weight, targets, target_weight, stats, *_):
    del grad_losses, weight, targets, target_weight, stats
    return torch.empty_like(hidden)


def _setup_context(ctx, inputs, output):
    hidden, weight, targets, target_weight, _terms, grad_terms, chunk_rows = inputs
    _, argmax, stats = output
    ctx.mark_non_differentiable(argmax, stats)
    # Backward reads the targets only for distribution terms; a CE-only loss
    # releases them after forward (CE caches the target argmax in stats).
    if not grad_terms & fused.DIST_MASK:
        targets = targets.new_empty((0, targets.shape[-1]))
        target_weight = None
    ctx.save_for_backward(hidden, weight, targets, target_weight, stats)
    ctx.grad_terms = grad_terms
    ctx.chunk_rows = chunk_rows


def _backward(ctx, grad_losses, _grad_argmax, _grad_stats):
    hidden, weight, targets, target_weight, stats = ctx.saved_tensors
    d_hidden = None
    if ctx.grad_terms and ctx.needs_input_grad[0]:
        d_hidden = _fused_linear_bwd(
            grad_losses.contiguous(),
            hidden,
            weight,
            targets,
            target_weight,
            stats,
            ctx.grad_terms,
            ctx.chunk_rows,
        )
    return d_hidden, None, None, None, None, None, None


_fused_linear_fwd.register_autograd(_backward, setup_context=_setup_context)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def fused_linear_compound(
    hidden: torch.Tensor,
    weight: torch.Tensor,
    targets: torch.Tensor,
    names: Sequence[str],
    metric_terms: Sequence[str] = (),
    chunk_rows: int | None = None,
    target_weight: torch.Tensor | None = None,
) -> "fused.FusedOutputs":
    """``fused.fused_compound`` without materializing ``hidden @ weight.T``.

    Args:
        hidden: ``[..., H]`` final (post-norm) draft hidden states.
        weight: ``[V, H]`` frozen draft ``lm_head`` weight.
        targets: rows aligned with ``hidden``: ``[..., V]`` target logits, or
            ``[..., H_t]`` post-norm verifier hidden states when
            ``target_weight`` is given.
        names: named losses (keys of ``fused.LOSS_TERMS``) to differentiate.
        metric_terms: primitive terms computed in the same pass, detached.
        chunk_rows: rows per chunk; ``None`` sizes chunks from the budget
            (``SPECULATORS_FUSED_LINEAR_BUDGET_MIB``, default 4096).
        target_weight: ``[V, H_t]`` frozen verifier ``lm_head`` weight; the
            target logits are then recomputed per chunk, never materialized.

    Returns:
        ``FusedOutputs`` whose tensors have ``hidden``'s leading shape.
    """
    if weight.requires_grad:
        raise ValueError(
            "fused_linear_compound supports frozen draft weights only; "
            "use the 'fused' loss implementation for a trainable lm_head"
        )
    if target_weight is not None and target_weight.requires_grad:
        raise ValueError("target_weight (verifier lm_head) must be frozen")
    lead = hidden.shape[:-1]
    # The kernels index targets as contiguous rows; both are no-ops when
    # the inputs already are.
    flat_hidden = hidden.reshape(-1, hidden.shape[-1]).contiguous()
    flat_targets = targets.reshape(-1, targets.shape[-1]).contiguous()
    device_type = flat_hidden.device.type
    if torch.is_autocast_enabled(device_type):
        # The operands autocast would give lm_head (the same bf16 weight copy);
        # the ops run with autocast off, so forward and the backward recompute
        # see identical logits.
        compute_dtype = torch.get_autocast_dtype(device_type)
        flat_hidden = flat_hidden.to(compute_dtype)
        weight = weight.to(compute_dtype)
        if target_weight is not None:
            flat_targets = flat_targets.to(compute_dtype)
            target_weight = target_weight.to(compute_dtype)
    elif weight.dtype != flat_hidden.dtype:
        raise ValueError(
            f"hidden ({flat_hidden.dtype}) and weight ({weight.dtype}) dtypes "
            "differ; cast them or run under torch.autocast"
        )
    elif target_weight is not None and target_weight.dtype != flat_targets.dtype:
        raise ValueError(
            f"verifier hidden ({flat_targets.dtype}) and target_weight "
            f"({target_weight.dtype}) dtypes differ; cast them or run under "
            "torch.autocast"
        )
    rows, vocab = flat_hidden.shape[0], weight.shape[0]
    if target_weight is not None and (
        target_weight.shape[0] != vocab
        or flat_targets.shape != (rows, target_weight.shape[1])
    ):
        raise ValueError(
            f"verifier hidden {tuple(targets.shape)} and target_weight "
            f"{tuple(target_weight.shape)} do not match hidden rows {rows} and "
            f"vocab {vocab}"
        )
    if target_weight is None and flat_targets.shape != (rows, vocab):
        raise ValueError(
            f"targets {tuple(targets.shape)} do not match hidden rows {rows} "
            f"and vocab {vocab}"
        )
    if chunk_rows is None:
        chunk_rows = chunk_rows_for(
            vocab, flat_hidden.dtype, rows, recompute_targets=target_weight is not None
        )

    loss_terms = sorted(
        {t for name in names for t in fused.LOSS_TERMS[name]},
        key=lambda t: fused.TERM_BITS[t],
    )
    grad_mask = fused.terms_mask(loss_terms)
    losses, argmax, _ = _fused_linear_fwd(
        flat_hidden,
        weight,
        flat_targets,
        target_weight,
        grad_mask | fused.terms_mask(metric_terms),
        grad_mask,
        chunk_rows,
    )
    terms = {t: losses[fused.TERM_SLOTS[t]].view(lead) for t in loss_terms}
    return fused.FusedOutputs(
        losses={name: fused.compose_loss(name, terms) for name in names},
        metrics={
            t: losses[fused.TERM_SLOTS[t]].view(lead).detach() for t in metric_terms
        },
        pred_ids=argmax[0].view(lead),
        target_ids=argmax[1].view(lead),
    )
