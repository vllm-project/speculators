"""Fused linear losses: drafter losses straight from hidden states.

The draft logits ``hidden @ weight.T`` are computed one row chunk at a time
into a scratch tile, reduced by the vocab kernels from ``losses.fused``, and
dropped, so the ``[N, V]`` draft logits and their gradient never exist.
Targets come in either as materialized ``[N, V]`` logits, or as the post-norm
verifier hidden states ``[N, H_t]`` plus the frozen verifier ``lm_head``
weight, in which case each chunk's target logits are computed the same way
and no ``[N, V]`` tensor exists at all. Targets receive no gradient; the
draft weight must be frozen.

Gradients, two modes:

- **deferred** (default for at most two differentiated terms): forward also
  writes each term's unit-weight logits gradient for the chunk and contracts
  it with the weight right away, ``jac_t = (d loss_t / d z) @ W``. Backward is
  ``d hidden = sum_t grad_output_t[:, None] * jac_t``: no GEMM, nothing
  recomputed, and only ``jac`` (``[terms, N, H]``) is kept for backward. This
  is exact because every loss weight -- mask, decay, D-PACE, term weight --
  is a per-row scalar applied after the loss. Same GEMM count as
  materialized logits (draft, target, ``dz @ W``).
- **recompute**: forward computes losses and per-row statistics only;
  backward recomputes each chunk's logits and writes ``dz @ W``.

Logit tiles are in the input dtype by default (bf16 in training, like
materialized logits); ``logits_dtype="float32"`` takes them in fp32 straight
from the GEMM accumulator -- more accurate, ~20% slower GEMMs.

The per-row loss contract is unchanged: unreduced per-term losses, a per-row
``grad_output`` per term, so masking, decay, D-PACE and term weights stay in
``losses.utils``. The chunk loops are ``torch.library.custom_op``s, opaque to
torch.compile (traced, they would unroll per chunk count and recompile when
the row count changes).
"""

import os
from collections.abc import Sequence

import torch

from speculators.losses import fused

__all__ = ["chunk_rows_for", "fused_linear_compound"]

# Peak scratch the chunk loop may use, in MiB, and the chunk-size cap.
# Measured on MI355X (hipBLASLt, H = 4096, V = 152k, ms per 1k rows): the
# projection runs at full-GEMM speed from ~768 rows up; the dz @ W
# contraction is shape-sensitive -- 1.80-1.91 at 1024, 1792, 2048, 4096,
# 6144 and 8192 rows but 2.4-3.1 at 1536, 3072, 3584, 3840 and 5120. 2048-row
# chunks are the fastest for both GEMMs and halve the scratch of 4096, so the
# budget only lowers the chunk below that (e.g. fp32 tiles at large V).
_BUDGET_ENV = "SPECULATORS_FUSED_LINEAR_BUDGET_MIB"
_DEFAULT_BUDGET_MIB = 4096
_MIN_CHUNK_ROWS = 64
_MAX_CHUNK_ROWS = 2048
_LOG2 = 0.6931471805599453
GRAD_MODES = ("auto", "deferred", "recompute")
LOGITS_DTYPES = ("input", "float32")
# auto picks deferred up to this many differentiated terms: jac costs one
# [N, H] tensor and one dz @ W GEMM per term in forward.
_MAX_DEFERRED_TERMS = 2


def chunk_rows_for(
    vocab: int,
    dtype: torch.dtype,
    rows: int,
    recompute_targets: bool = False,
    fp32_logits: bool = False,
    dz_tiles: int = 1,
) -> int:
    """Rows per chunk: the largest power of two whose scratch fits the
    budget, capped at ``_MAX_CHUNK_ROWS``; the last chunk takes the remainder.

    Per row the loop holds a draft-logits row, a target-logits row when
    targets are recomputed (both fp32 or input dtype), and ``dz_tiles``
    input-dtype gradient rows (one per term in deferred mode).
    """
    budget = int(os.environ.get(_BUDGET_ENV, _DEFAULT_BUDGET_MIB)) * 1024**2
    elem = torch.empty((), dtype=dtype).element_size()
    tile = 4 if fp32_logits else elem
    per_row = vocab * (tile * (2 if recompute_targets else 1) + elem * dz_tiles)
    fit = max(budget // per_row, 1)
    chunk = max(1 << (fit.bit_length() - 1), _MIN_CHUNK_ROWS)
    return min(chunk, _MAX_CHUNK_ROWS, max(rows, 1))


def _out_dtype(hidden: torch.Tensor) -> torch.dtype:
    """Loss / stats dtype: fp32, or fp64 for fp64 inputs (CPU gradcheck)."""
    return torch.promote_types(torch.float32, hidden.dtype)


def _project(hidden: torch.Tensor, weight: torch.Tensor, fp32: bool) -> torch.Tensor:
    """``hidden @ weight.T``; fp32 straight from the accumulator if ``fp32``.

    Autocast is disabled: forward and a backward recompute must produce
    bit-identical logits from the operands as given (the wrapper casts them
    for autocast), or saved statistics would not match recomputed logits.
    """
    with torch.autocast(hidden.device.type, enabled=False):
        if fp32 and hidden.dtype in (torch.float16, torch.bfloat16):
            return torch.mm(hidden, weight.t(), out_dtype=torch.float32)
        return hidden @ weight.t()


def _contract(dz: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """``dz @ weight`` with an fp32 (or wider) result: ``jac`` stays unrounded
    until backward scales and sums it (the output is small, so this costs no
    GEMM time)."""
    with torch.autocast(dz.device.type, enabled=False):
        if dz.dtype in (torch.float16, torch.bfloat16):
            return torch.mm(dz, weight, out_dtype=torch.float32)
        return dz @ weight


def _chunks(rows: int, chunk_rows: int):
    for start in range(0, rows, chunk_rows):
        yield start, min(start + chunk_rows, rows)


def _grad_term_names(grad_terms: int) -> list[str]:
    """Differentiated primitive terms in slot order (the jac / dz order)."""
    return [name for name, bit in fused.TERM_BITS.items() if grad_terms & bit]


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


def _target_chunk(targets, target_weight, start, end, fp32):
    """Target logits for rows ``start:end``: a slice of the materialized
    logits, or computed from the verifier hidden states."""
    rows = targets[start:end]
    return rows if target_weight is None else _project(rows, target_weight, fp32)


def _forward_deferred_chunk(
    z, chunk_targets, weight, terms, grad_terms, losses, argmax, jac, start, end
):
    """Losses, argmaxes and ``jac[:, start:end]`` for one chunk."""
    names = _grad_term_names(grad_terms)
    rows, vocab = end - start, z.shape[1]
    if z.device.type == "cpu":
        stats = z.new_zeros((fused.N_STATS, rows))
        _ref_forward_chunk(
            z, chunk_targets, terms, losses[:, start:end], argmax[:, start:end], stats
        )
        for k, name in enumerate(names):
            unit = z.new_zeros((fused.N_TERMS, rows))
            unit[fused.TERM_SLOTS[name]] = 1.0
            dz = _ref_backward_chunk(
                z, chunk_targets, unit, stats, fused.TERM_BITS[name]
            ).to(weight.dtype)
            jac[k, start:end] = _contract(dz, weight)
        return
    # dz in the weight (input) dtype, like the materialized grad_in; jac is
    # accumulated in fp32 so the gradient is rounded to the input dtype once.
    dz = torch.empty((len(names), rows, vocab), device=z.device, dtype=weight.dtype)
    slot = {name: k for k, name in enumerate(names)}
    fused.loss_grad_kernel[(rows,)](
        z,
        chunk_targets,
        losses[:, start:],
        losses.stride(0),
        argmax[:, start:],
        argmax.stride(0),
        dz,
        rows * vocab,
        vocab,
        TERMS=terms,
        GRAD_TERMS=grad_terms,
        DZ_KL=slot.get("kl_div", 0),
        DZ_RKL=slot.get("rkl", 0),
        DZ_JSD=slot.get("jsd", 0),
        DZ_CE=slot.get("ce", 0),
        DZ_TV=slot.get("tv", 0),
    )
    for k in range(len(names)):
        jac[k, start:end] = _contract(dz[k], weight)


@torch.library.custom_op("speculators::fused_linear_loss_fwd", mutates_args=())
def _fused_linear_fwd(
    hidden: torch.Tensor,
    weight: torch.Tensor,
    targets: torch.Tensor,
    target_weight: torch.Tensor | None,
    terms: int,
    grad_terms: int,
    chunk_rows: int,
    deferred: bool,
    fp32_logits: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Losses ``[_N_TERMS, N]``, argmax ``[2, N]``, stats ``[_N_STATS, N]``, jac.

    ``targets`` is ``[N, V]`` logits, or ``[N, H_t]`` verifier hidden states
    when ``target_weight`` (``[V, H_t]``) is given. With ``deferred``, ``jac``
    is ``[n_grad_terms, N, H]`` (one unit-weight ``d loss / d hidden`` per
    differentiated term, slot order) and stats are unused; otherwise ``jac``
    is empty and stats feed the recompute backward. Slots of terms outside
    ``terms`` are zero.
    """
    rows, vocab = hidden.shape[0], weight.shape[0]
    dtype = _out_dtype(hidden)
    losses = hidden.new_zeros((fused.N_TERMS, rows), dtype=dtype)
    argmax = hidden.new_zeros((2, rows), dtype=torch.int64)
    stats = hidden.new_zeros((fused.N_STATS, rows), dtype=dtype)
    n_jac = len(_grad_term_names(grad_terms)) if deferred else 0
    jac = hidden.new_empty((n_jac, rows if n_jac else 0, hidden.shape[1]), dtype=dtype)
    for start, end in _chunks(rows, chunk_rows):
        z = _project(hidden[start:end], weight, fp32_logits)
        chunk_targets = _target_chunk(targets, target_weight, start, end, fp32_logits)
        if deferred:
            _forward_deferred_chunk(
                z,
                chunk_targets,
                weight,
                terms,
                grad_terms,
                losses,
                argmax,
                jac,
                start,
                end,
            )
        elif hidden.device.type == "cpu":
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
    return losses, argmax, stats, jac


@_fused_linear_fwd.register_fake
def _(
    hidden,
    weight,
    targets,
    target_weight,
    terms,
    grad_terms,
    chunk_rows,
    deferred,
    fp32_logits,
):
    del weight, targets, target_weight, terms, chunk_rows, fp32_logits
    rows = hidden.shape[0]
    dtype = _out_dtype(hidden)
    n_jac = len(_grad_term_names(grad_terms)) if deferred else 0
    return (
        hidden.new_empty((fused.N_TERMS, rows), dtype=dtype),
        hidden.new_empty((2, rows), dtype=torch.int64),
        hidden.new_empty((fused.N_STATS, rows), dtype=dtype),
        hidden.new_empty((n_jac, rows if n_jac else 0, hidden.shape[1]), dtype=dtype),
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
    fp32_logits: bool,
) -> torch.Tensor:
    """Recompute backward: per chunk, recompute the logits, form ``dz``,
    write ``dz @ weight``. Target logits are only needed for distribution
    terms; CE reads its cached target argmax from ``stats``."""
    rows, vocab = hidden.shape[0], weight.shape[0]
    needs_targets = bool(grad_terms & fused.DIST_MASK)
    d_hidden = torch.empty_like(hidden)
    for start, end in _chunks(rows, chunk_rows):
        z = _project(hidden[start:end], weight, fp32_logits)
        chunk_targets = (
            _target_chunk(targets, target_weight, start, end, fp32_logits)
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


@torch.library.custom_op("speculators::fused_linear_deferred_bwd", mutates_args=())
def _fused_linear_deferred_bwd(
    grad_losses: torch.Tensor,
    jac: torch.Tensor,
    slots: list[int],
    out_dtype: torch.dtype,
) -> torch.Tensor:
    """Deferred backward: ``d hidden = sum_t grad_losses[slot_t][:, None] * jac_t``,
    accumulated in ``jac``'s dtype (fp32) and rounded once to ``out_dtype``."""
    acc = torch.zeros(jac.shape[1:], device=jac.device, dtype=jac.dtype)
    for k, slot in enumerate(slots):
        acc.addcmul_(grad_losses[slot].to(jac.dtype)[:, None], jac[k])
    return acc.to(out_dtype)


@_fused_linear_deferred_bwd.register_fake
def _(grad_losses, jac, slots, out_dtype):
    del grad_losses, slots
    return jac.new_empty(jac.shape[1:], dtype=out_dtype)


def _setup_context(ctx, inputs, output):
    (
        hidden,
        weight,
        targets,
        target_weight,
        _terms,
        grad_terms,
        chunk_rows,
        deferred,
        fp32_logits,
    ) = inputs
    _, argmax, stats, jac = output
    ctx.mark_non_differentiable(argmax, stats, jac)
    ctx.grad_terms = grad_terms
    ctx.chunk_rows = chunk_rows
    ctx.deferred = deferred
    ctx.fp32_logits = fp32_logits
    if deferred:
        # Everything backward needs is already contracted into jac.
        ctx.grad_slots = [fused.TERM_SLOTS[n] for n in _grad_term_names(grad_terms)]
        ctx.hidden_dtype = hidden.dtype
        ctx.save_for_backward(jac)
        return
    # Backward reads the targets only for distribution terms; a CE-only loss
    # releases them after forward (CE caches the target argmax in stats).
    if not grad_terms & fused.DIST_MASK:
        targets = targets.new_empty((0, targets.shape[-1]))
        target_weight = None
    ctx.save_for_backward(hidden, weight, targets, target_weight, stats)


def _backward(ctx, grad_losses, _grad_argmax, _grad_stats, _grad_jac):
    d_hidden = None
    if ctx.deferred and ctx.needs_input_grad[0]:
        (jac,) = ctx.saved_tensors
        d_hidden = _fused_linear_deferred_bwd(
            grad_losses.contiguous(), jac, ctx.grad_slots, ctx.hidden_dtype
        )
    elif ctx.grad_terms and ctx.needs_input_grad[0]:
        hidden, weight, targets, target_weight, stats = ctx.saved_tensors
        d_hidden = _fused_linear_bwd(
            grad_losses.contiguous(),
            hidden,
            weight,
            targets,
            target_weight,
            stats,
            ctx.grad_terms,
            ctx.chunk_rows,
            ctx.fp32_logits,
        )
    return d_hidden, None, None, None, None, None, None, None, None


_fused_linear_fwd.register_autograd(_backward, setup_context=_setup_context)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def _prepare_inputs(hidden, weight, targets, target_weight):
    """Validate and flatten the inputs; cast them for autocast.

    Returns ``(flat_hidden, weight, flat_targets, target_weight)``.
    """
    if weight.requires_grad:
        raise ValueError(
            "fused_linear_compound supports frozen draft weights only; "
            "use the 'fused' loss implementation for a trainable lm_head"
        )
    if target_weight is not None and target_weight.requires_grad:
        raise ValueError("target_weight (verifier lm_head) must be frozen")
    # The kernels index targets as contiguous rows; both are no-ops when
    # the inputs already are.
    flat_hidden = hidden.reshape(-1, hidden.shape[-1]).contiguous()
    flat_targets = targets.reshape(-1, targets.shape[-1]).contiguous()
    device_type = flat_hidden.device.type
    if torch.is_autocast_enabled(device_type):
        # The operands autocast would give lm_head (the same bf16 weight copy);
        # the ops run with autocast off, so forward and a backward recompute
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
    total_rows, vocab = flat_hidden.shape[0], weight.shape[0]
    if target_weight is not None and (
        target_weight.shape[0] != vocab
        or flat_targets.shape != (total_rows, target_weight.shape[1])
    ):
        raise ValueError(
            f"verifier hidden {tuple(targets.shape)} and target_weight "
            f"{tuple(target_weight.shape)} do not match hidden rows {total_rows} "
            f"and vocab {vocab}"
        )
    if target_weight is None and flat_targets.shape != (total_rows, vocab):
        raise ValueError(
            f"targets {tuple(targets.shape)} do not match hidden rows "
            f"{total_rows} and vocab {vocab}"
        )
    return flat_hidden, weight, flat_targets, target_weight


def fused_linear_compound(
    hidden: torch.Tensor,
    weight: torch.Tensor,
    targets: torch.Tensor,
    names: Sequence[str],
    metric_terms: Sequence[str] = (),
    chunk_rows: int | None = None,
    target_weight: torch.Tensor | None = None,
    grad_mode: str = "auto",
    logits_dtype: str = "input",
    rows: torch.Tensor | None = None,
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
            target logits are then computed per chunk, never materialized.
        grad_mode: ``"deferred"``, ``"recompute"`` or ``"auto"`` (deferred
            for at most two differentiated terms). Without a backward pass
            (no grad) only the losses are computed.
        logits_dtype: ``"input"`` (logit tiles in the input dtype, as
            materialized logits) or ``"float32"`` (fp32 from the GEMM
            accumulator: more accurate, slower GEMMs).
        rows: optional 1-D index of the rows to compute; every other row is
            treated as fully masked (zero loss, zero gradient, ids 0). Use it
            to skip rows that are masked by construction.

    Returns:
        ``FusedOutputs`` whose tensors have ``hidden``'s leading shape.
    """
    if grad_mode not in GRAD_MODES:
        raise ValueError(f"grad_mode must be one of {GRAD_MODES}, got {grad_mode!r}")
    if logits_dtype not in LOGITS_DTYPES:
        raise ValueError(
            f"logits_dtype must be one of {LOGITS_DTYPES}, got {logits_dtype!r}"
        )
    flat_hidden, weight, flat_targets, target_weight = _prepare_inputs(
        hidden, weight, targets, target_weight
    )
    lead = hidden.shape[:-1]
    total_rows, vocab = flat_hidden.shape[0], weight.shape[0]
    if rows is not None:
        flat_hidden = flat_hidden.index_select(0, rows)
        flat_targets = flat_targets.index_select(0, rows)

    loss_terms = sorted(
        {t for name in names for t in fused.LOSS_TERMS[name]},
        key=lambda t: fused.TERM_BITS[t],
    )
    grad_mask = fused.terms_mask(loss_terms)
    needs_grad = torch.is_grad_enabled() and flat_hidden.requires_grad
    deferred = needs_grad and (
        grad_mode == "deferred"
        or (grad_mode == "auto" and len(loss_terms) <= _MAX_DEFERRED_TERMS)
    )
    fp32_logits = logits_dtype == "float32"
    if chunk_rows is None:
        chunk_rows = chunk_rows_for(
            vocab,
            flat_hidden.dtype,
            flat_hidden.shape[0],
            recompute_targets=target_weight is not None,
            fp32_logits=fp32_logits,
            dz_tiles=len(loss_terms) if deferred else 1,
        )
    losses, argmax, _, _ = _fused_linear_fwd(
        flat_hidden,
        weight,
        flat_targets,
        target_weight,
        grad_mask | fused.terms_mask(metric_terms),
        grad_mask if needs_grad else 0,
        chunk_rows,
        deferred,
        fp32_logits,
    )
    if rows is not None:
        losses = losses.new_zeros((losses.shape[0], total_rows)).index_copy(
            1, rows, losses
        )
        argmax = argmax.new_zeros((2, total_rows)).index_copy(1, rows, argmax)
    terms = {t: losses[fused.TERM_SLOTS[t]].view(lead) for t in loss_terms}
    return fused.FusedOutputs(
        losses={name: fused.compose_loss(name, terms) for name in names},
        metrics={
            t: losses[fused.TERM_SLOTS[t]].view(lead).detach() for t in metric_terms
        },
        pred_ids=argmax[0].view(lead),
        target_ids=argmax[1].view(lead),
    )
