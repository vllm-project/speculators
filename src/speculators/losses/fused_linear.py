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

Targets come in as materialized ``[N, V]`` logits (no gradient). The draft
weight must be frozen; trainable draft weights are not supported yet.
"""

import os
from collections.abc import Sequence

import torch

from speculators.losses import fused

__all__ = ["chunk_rows_for", "fused_linear_compound"]

# Peak scratch the chunk loop may use, in MiB. Measured on MI355X (PR 0
# gemm-sweep): chunks of <= 256 rows cost 1.5-4.6x a full GEMM at V = 152k,
# 1024 rows about 1.2x, so the budget is sized for ~1-2k-row chunks.
_BUDGET_ENV = "SPECULATORS_FUSED_LINEAR_BUDGET_MIB"
_DEFAULT_BUDGET_MIB = 2048
_MIN_CHUNK_ROWS = 64


def chunk_rows_for(vocab: int, dtype: torch.dtype, rows: int) -> int:
    """Rows per chunk: the largest power of two whose scratch fits the budget.

    Per row the loop holds an fp32 logits row (forward and backward) and, in
    backward, an input-dtype ``dz`` row.
    """
    budget = int(os.environ.get(_BUDGET_ENV, _DEFAULT_BUDGET_MIB)) * 1024**2
    per_row = vocab * (4 + torch.empty((), dtype=dtype).element_size())
    fit = max(budget // per_row, 1)
    chunk = max(1 << (fit.bit_length() - 1), _MIN_CHUNK_ROWS)
    return min(chunk, max(rows, 1))


def _project(hidden: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """``hidden @ weight.T`` with an fp32 result from the accumulator.

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
# Custom ops
# ---------------------------------------------------------------------------


@torch.library.custom_op("speculators::fused_linear_loss_fwd", mutates_args=())
def _fused_linear_fwd(
    hidden: torch.Tensor,
    weight: torch.Tensor,
    targets: torch.Tensor,
    terms: int,
    grad_terms: int,
    chunk_rows: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Per-term losses ``[N_TERMS, N]``, argmax ``[2, N]`` and stats ``[N_STATS, N]``.

    ``grad_terms`` (a subset of ``terms``) is only consumed by the autograd
    registration; slots of terms outside ``terms`` are zero.
    """
    del grad_terms
    rows, vocab = hidden.shape[0], weight.shape[0]
    dtype = torch.float32
    losses = hidden.new_zeros((fused.N_TERMS, rows), dtype=dtype)
    argmax = hidden.new_zeros((2, rows), dtype=torch.int64)
    stats = hidden.new_zeros((fused.N_STATS, rows), dtype=dtype)
    for start, end in _chunks(rows, chunk_rows):
        z = _project(hidden[start:end], weight)
        chunk_targets = targets[start:end]
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
        # Release the tile before the next chunk allocates its own.
        del z
    return losses, argmax, stats


@_fused_linear_fwd.register_fake
def _(hidden, weight, targets, terms, grad_terms, chunk_rows):  # noqa: ARG001
    rows = hidden.shape[0]
    dtype = torch.float32
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
    stats: torch.Tensor,
    grad_terms: int,
    chunk_rows: int,
) -> torch.Tensor:
    """``d hidden``: per chunk, recompute the logits, form ``dz``, ``dz @ weight``."""
    rows, vocab = hidden.shape[0], weight.shape[0]
    d_hidden = torch.empty_like(hidden)
    for start, end in _chunks(rows, chunk_rows):
        z = _project(hidden[start:end], weight)
        chunk_targets = targets[start:end]
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
        with torch.autocast(hidden.device.type, enabled=False):
            torch.mm(dz, weight, out=d_hidden[start:end])
        del z, dz
    return d_hidden


@_fused_linear_bwd.register_fake
def _(grad_losses, hidden, weight, targets, stats, grad_terms, chunk_rows):  # noqa: ARG001
    return torch.empty_like(hidden)


def _setup_context(ctx, inputs, output):
    hidden, weight, targets, _terms, grad_terms, chunk_rows = inputs
    _, argmax, stats = output
    ctx.mark_non_differentiable(argmax, stats)
    # Backward reads the targets only for distribution terms; a CE-only loss
    # releases them after forward (CE caches the target argmax in stats).
    if not grad_terms & fused.DIST_MASK:
        targets = targets.new_empty((0, targets.shape[-1]))
    ctx.save_for_backward(hidden, weight, targets, stats)
    ctx.grad_terms = grad_terms
    ctx.chunk_rows = chunk_rows


def _backward(ctx, grad_losses, _grad_argmax, _grad_stats):
    hidden, weight, targets, stats = ctx.saved_tensors
    d_hidden = None
    if ctx.grad_terms and ctx.needs_input_grad[0]:
        d_hidden = _fused_linear_bwd(
            grad_losses.contiguous(),
            hidden,
            weight,
            targets,
            stats,
            ctx.grad_terms,
            ctx.chunk_rows,
        )
    return d_hidden, None, None, None, None, None


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
) -> "fused.FusedOutputs":
    """``fused.fused_compound`` without materializing ``hidden @ weight.T``.

    Args:
        hidden: ``[..., H]`` final (post-norm) draft hidden states.
        weight: ``[V, H]`` frozen draft ``lm_head`` weight.
        targets: ``[..., V]`` target logits, rows aligned with ``hidden``.
        names: named losses (keys of ``fused.LOSS_TERMS``) to differentiate.
        metric_terms: primitive terms computed in the same pass, detached.
        chunk_rows: rows per chunk; ``None`` sizes chunks from the budget
            (``SPECULATORS_FUSED_LINEAR_BUDGET_MIB``, default 2048).

    Returns:
        ``FusedOutputs`` whose tensors have ``hidden``'s leading shape.
    """
    if weight.requires_grad:
        raise ValueError(
            "fused_linear_compound supports frozen draft weights only; "
            "use the 'fused' loss implementation for a trainable lm_head"
        )
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
    elif weight.dtype != flat_hidden.dtype:
        raise ValueError(
            f"hidden ({flat_hidden.dtype}) and weight ({weight.dtype}) dtypes "
            "differ; cast them or run under torch.autocast"
        )
    rows, vocab = flat_hidden.shape[0], weight.shape[0]
    if flat_targets.shape != (rows, vocab):
        raise ValueError(
            f"targets {tuple(targets.shape)} do not match hidden rows {rows} "
            f"and vocab {vocab}"
        )
    if chunk_rows is None:
        chunk_rows = chunk_rows_for(vocab, flat_hidden.dtype, rows)
    elif chunk_rows < 1:
        raise ValueError(f"chunk_rows must be positive, got {chunk_rows}")
    if flat_hidden.device.type == "cpu":
        raise ValueError("fused_linear runs Triton kernels and needs a GPU")

    loss_terms = sorted(
        {t for name in names for t in fused.LOSS_TERMS[name]},
        key=lambda t: fused.TERM_BITS[t],
    )
    grad_mask = fused.terms_mask(loss_terms)
    losses, argmax, _ = _fused_linear_fwd(
        flat_hidden,
        weight,
        flat_targets,
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
