# ruff: noqa: SLF001 -- built from fused_linear's chunk helpers
"""Fused linear loss with the reduction inside, for a trainable extra linear.

DSpark's draft logits are ``hidden @ W.T + u @ W2.T``: the frozen ``lm_head``
plus the Markov bias ``markov_w2(u)``, whose weight ``W2`` trains. ``W2``'s
gradient sums rows with their loss weights, which ``fused_linear_compound``
only receives in backward, so this op reduces the loss itself (mask, fixed
decay or D-PACE, term weights, as ``losses.utils._weighted_sum``) and forms
every gradient in forward; backward scales them by the scalar loss gradient.
Exact as long as the loss is used linearly, as a training loss is. Tiles,
the shared head and chunking are those of ``fused_linear``.

Each call builds ``[W | W2]`` once, so every chunk forms both linears in one
GEMM over ``[hidden | u]``; a separate ``K = r`` GEMM runs far below peak.
The copy lives only for the call; no state is kept between calls.
"""

from collections.abc import Sequence

import torch

from speculators.losses import fused
from speculators.losses import fused_linear as fl
from speculators.losses.utils import _LOSS_REDUCTION_EPS, dpace_loss_decay

__all__ = ["fused_linear_reduced"]

_SPLIT_K = (16, 8, 4, 2)


def _term_derivatives(name, terms):
    """``d compose_loss(name) / d term`` per row (lk_hybrid's weight is
    detached there, so it is a constant here)."""
    if name == "nla":
        alive = 1.0 - terms["tv"]
        return {"tv": torch.where(alive > fused._NLA_EPS, 1.0 / alive, 0.0)}
    if name == "lk_hybrid":
        weight = torch.exp(-3.0 * (1.0 - terms["tv"]))
        return {"kl_div": weight, "tv": 1.0 - weight}
    return {name: torch.ones_like(terms[name])}


def _chunk_reduction(losses, names, weights, mask, decay, denom, alpha, block_size):
    """The chunk's share of the scalar loss and its derivative per primitive
    term and row, reduced as ``_weighted_sum`` (D-PACE weights are constants)."""
    terms = {t: losses[slot] for t, slot in fused.TERM_SLOTS.items()}
    total = losses.new_zeros(())
    mult = {t: torch.zeros_like(mask) for t in terms}
    for name, w in zip(names, weights, strict=True):
        el = fused.compose_loss(name, terms) * mask
        if alpha is not None:
            # pos_idx is unused by D-PACE
            decay = dpace_loss_decay(
                mask[None], mask[None], block_size, alpha, elementwise_loss=el[None]
            )[0]
        scale = decay * mask * (w / denom)
        total = total + (el * scale).sum()
        for t, d in _term_derivatives(name, terms).items():
            mult[t] = mult[t] + scale * d
    return total, mult


def _concat_weight(weight, extra_weight):
    """``[weight | extra_weight]``."""
    hid = weight.shape[1]
    cat = weight.new_empty((weight.shape[0], hid + extra_weight.shape[1]))
    cat[:, :hid] = weight
    cat[:, hid:] = extra_weight
    return cat


def _contract_extra(dz: torch.Tensor, extra_weight: torch.Tensor) -> torch.Tensor:
    """``dz @ extra_weight`` ([rows, V] @ [V, r]) in fp32, split over the vocab
    as a batched GEMM: the output alone is too few tiles to fill the GPU."""
    rows, vocab = dz.shape
    split = next((s for s in _SPLIT_K if vocab % s == 0), 1)
    if split == 1 or not extra_weight.is_contiguous():
        return fl._contract(dz, extra_weight)
    with torch.autocast(dz.device.type, enabled=False):
        parts = torch.bmm(
            dz.view(rows, split, vocab // split).transpose(0, 1),
            extra_weight.view(split, vocab // split, -1),
            out_dtype=torch.float32,
        )
    return parts.sum(0)


def _launch(z, t, losses, argmax, start, terms, grad_terms, delta, dz_dtype):
    """Losses and argmaxes of rows ``start:`` (written in place) and, with
    ``grad_terms``, each term's unit-weight ``d loss / d z``, ``[k, n, V]``."""
    n, vocab = z.shape
    out = (losses[:, start:], losses.stride(0), argmax[:, start:], argmax.stride(0))
    if not grad_terms:
        stats = z.new_empty((fused.N_STATS, n), dtype=torch.float32)
        fused.loss_forward_kernel[(n,)](
            z, t, *out, stats, stats.stride(0), vocab, TERMS=terms, DELTA=delta
        )
        return None
    names = fl._grad_term_names(grad_terms)
    slot = {name: k for k, name in enumerate(names)}
    dz = torch.empty((len(names), n, vocab), device=z.device, dtype=dz_dtype)
    fused.loss_grad_kernel[(n,)](
        z, t, *out, dz, n * vocab, vocab, TERMS=terms, GRAD_TERMS=grad_terms,
        DZ_KL=slot.get("kl_div", 0), DZ_RKL=slot.get("rkl", 0),
        DZ_JSD=slot.get("jsd", 0), DZ_CE=slot.get("ce", 0),
        DZ_TV=slot.get("tv", 0), DELTA=delta,
    )  # fmt: skip
    return dz


def _fold(dz, mult, names):
    """``(d, g)`` with ``sum_k mult[k] * dz[k] = g * d``: one tile to
    contract; a single term is scaled after the contraction instead."""
    if len(names) == 1:
        return dz[0], mult[names[0]][:, None]
    acc = torch.zeros(dz.shape[1:], device=dz.device, dtype=torch.float32)
    for k, name in enumerate(names):
        acc.addcmul_(dz[k], mult[name][:, None])
    return acc.to(dz.dtype), None


@torch.library.custom_op("speculators::fused_linear_reduced_fwd", mutates_args=())
def _reduced_fwd(
    hidden: torch.Tensor,
    weight: torch.Tensor,
    extra: torch.Tensor,
    extra_weight: torch.Tensor,
    targets: torch.Tensor,
    target_weight: torch.Tensor,
    mask: torch.Tensor,
    decay: torch.Tensor,
    names: str,
    name_weights: list[float],
    dpace_alpha: float,
    block_size: int,
    terms: int,
    grad_terms: int,
    chunk_rows: int,
    fp32_logits: bool,
    delta: bool,
) -> tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
]:
    """``(loss, losses [N_TERMS, N], argmax [2, N], G_hidden, G_extra,
    G_extra_weight)``, the ``G`` fp32 ``d loss / d input`` (empty without
    grad terms). ``dpace_alpha < 0`` selects the fixed ``decay``."""
    rows, hid = hidden.shape
    f32 = torch.float32
    losses = hidden.new_zeros((fused.N_TERMS, rows), dtype=f32)
    argmax = hidden.new_zeros((2, rows), dtype=torch.int64)
    total = hidden.new_zeros((), dtype=f32)
    grad_names = fl._grad_term_names(grad_terms)

    def grad_buffer(t):
        return t.new_zeros(t.shape if grad_names else (0,), dtype=f32)

    g_hidden, g_extra, g_w2 = map(grad_buffer, (hidden, extra, extra_weight))
    alpha = None if dpace_alpha < 0 else dpace_alpha
    denom = mask.sum().to(f32) + _LOSS_REDUCTION_EPS
    cat = _concat_weight(weight, extra_weight)
    # a shared head projects the targets through the same W columns
    tile_tw = cat[:, :hid] if target_weight is weight else target_weight
    for start, end in fl._chunks(rows, chunk_rows):
        x = hidden[start:end]
        if delta:
            x = x.to(f32) - targets[start:end].to(f32)
        xu = torch.cat([x.to(cat.dtype), extra[start:end].to(cat.dtype)], 1)
        z = fl._project(xu, cat, fp32_logits)
        del xu
        t = fl._project(targets[start:end], tile_tw, fp32_logits)
        dz = _launch(
            z, t, losses, argmax, start, terms, grad_terms, delta, weight.dtype
        )
        del z, t
        part, mult = _chunk_reduction(
            losses[:, start:end],
            names.split(","),
            name_weights,
            mask[start:end].to(f32),
            decay[start:end].to(f32),
            denom,
            alpha,
            block_size,
        )
        total += part
        if dz is None:
            continue
        d, g = _fold(dz, mult, grad_names)
        del dz
        part_h = fl._contract(d, weight)
        part_u = _contract_extra(d, extra_weight)
        u = extra[start:end]
        if g is not None:
            part_h, part_u = g * part_h, g * part_u
            u = (g * u.to(f32)).to(d.dtype)
        g_hidden[start:end] = part_h
        g_extra[start:end] = part_u
        with torch.autocast(d.device.type, enabled=False):
            torch.addmm(g_w2, d.t(), u.to(d.dtype), out_dtype=f32, out=g_w2)
    return total, losses, argmax, g_hidden, g_extra, g_w2


@_reduced_fwd.register_fake
def _(hidden, weight, extra, extra_weight, *args):
    del weight
    grad_terms = args[-4]
    rows = hidden.shape[0]
    f32 = torch.float32
    grad = bool(grad_terms)
    return (
        hidden.new_empty((), dtype=f32),
        hidden.new_empty((fused.N_TERMS, rows), dtype=f32),
        hidden.new_empty((2, rows), dtype=torch.int64),
        hidden.new_empty(hidden.shape if grad else (0,), dtype=f32),
        hidden.new_empty(extra.shape if grad else (0,), dtype=f32),
        hidden.new_empty(extra_weight.shape if grad else (0,), dtype=f32),
    )


_GRAD_INPUTS = (0, 2, 3)  # hidden, extra, extra_weight


def _setup_context(ctx, inputs, output):
    _, losses, argmax, *grads = output
    ctx.mark_non_differentiable(losses, argmax, *grads)
    ctx.dtypes = [inputs[i].dtype for i in _GRAD_INPUTS]
    ctx.save_for_backward(*grads)


def _backward(ctx, grad_loss, *_):
    out: list[torch.Tensor | None] = [None] * 17
    for i, g, dtype in zip(_GRAD_INPUTS, ctx.saved_tensors, ctx.dtypes, strict=True):
        if ctx.needs_input_grad[i] and g.numel():
            out[i] = (g * grad_loss).to(dtype)
    return tuple(out)


_reduced_fwd.register_autograd(_backward, setup_context=_setup_context)


def fused_linear_reduced(
    hidden: torch.Tensor,
    weight: torch.Tensor,
    extra: torch.Tensor,
    extra_weight: torch.Tensor,
    targets: torch.Tensor,
    target_weight: torch.Tensor,
    names: Sequence[str],
    name_weights: Sequence[float],
    loss_mask: torch.Tensor,
    decay: torch.Tensor | None = None,
    dpace_alpha: float | None = None,
    block_size: int = 1,
    logits_dtype: str = "input",
    metric_terms: Sequence[str] = (),
    chunk_rows: int | None = None,
) -> tuple[torch.Tensor, "fused.FusedOutputs"]:
    """Scalar compound loss of ``hidden @ weight.T + extra @ extra_weight.T``.

    Args:
        hidden: ``[1, N, H]`` final draft hidden states.
        weight: ``[V, H]`` frozen draft ``lm_head`` weight.
        extra / extra_weight: ``[1, N, r]`` and ``[V, r]`` (trainable).
        targets / target_weight: ``[1, N, H_t]`` post-norm verifier hidden
            states and the frozen verifier ``lm_head`` ``[V, H_t]``; the
            target logits are projected per chunk.
        logits_dtype / metric_terms / chunk_rows: as ``fused_linear_compound``.
        names / name_weights: named losses and their weights.
        loss_mask: ``[1, N]``.
        decay: ``[1, N]`` fixed per-row decay (``None``: none); ignored with
            ``dpace_alpha``, which applies D-PACE per block of ``block_size``.

    Returns:
        ``(loss, outputs)``: the scalar loss, equal to ``_weighted_sum`` of
        the same per-row losses, and detached ``FusedOutputs``.
    """
    if logits_dtype not in fl.LOGITS_DTYPES:
        raise ValueError(
            f"logits_dtype must be one of {fl.LOGITS_DTYPES}, got {logits_dtype!r}"
        )
    if hidden.dim() != 3 or hidden.shape[0] != 1 or loss_mask.shape != hidden.shape[:2]:  # noqa: PLR2004
        raise ValueError("expected hidden [1, N, H] and loss_mask [1, N]")
    flat_hidden, weight, flat_targets, target_weight = fl._prepare_inputs(
        hidden, weight, targets, target_weight
    )
    rows, vocab = flat_hidden.shape[0], weight.shape[0]
    flat_extra = extra.reshape(rows, -1)
    if torch.is_autocast_enabled(flat_hidden.device.type):
        flat_extra = flat_extra.to(flat_hidden.dtype)
        extra_weight = extra_weight.to(flat_hidden.dtype)
    if extra_weight.shape != (vocab, flat_extra.shape[1]):
        raise ValueError(
            f"extra {tuple(extra.shape)} and extra_weight "
            f"{tuple(extra_weight.shape)} do not match vocab {vocab}"
        )
    if dpace_alpha is not None and rows % block_size:
        raise ValueError(f"D-PACE needs whole blocks: {rows} rows, block {block_size}")

    loss_terms = sorted(
        {t for name in names for t in fused.LOSS_TERMS[name]},
        key=lambda t: fused.TERM_BITS[t],
    )
    grad_mask = fused.terms_mask(loss_terms)
    all_terms = grad_mask | fused.terms_mask(metric_terms)
    needs_grad = torch.is_grad_enabled() and any(
        t.requires_grad for t in (flat_hidden, flat_extra, extra_weight)
    )
    delta = target_weight is weight and bool(all_terms & fused.DIST_MASK)
    if chunk_rows is None:
        chunk_rows = fl.chunk_rows_for(
            vocab,
            flat_hidden.dtype,
            rows,
            recompute_targets=True,
            fp32_logits=logits_dtype == "float32",
            dz_tiles=len(loss_terms),
        )
    if dpace_alpha is not None:
        chunk_rows = max(chunk_rows // block_size, 1) * block_size
    mask = loss_mask.reshape(-1)
    decay_rows = (
        torch.ones_like(mask, dtype=torch.float32)
        if decay is None
        else decay.reshape(-1)
    )
    loss, losses, argmax, *_ = _reduced_fwd(
        flat_hidden,
        weight,
        flat_extra,
        extra_weight,
        flat_targets,
        target_weight,
        mask,
        decay_rows,
        ",".join(names),
        [float(w) for w in name_weights],
        -1.0 if dpace_alpha is None else float(dpace_alpha),
        block_size,
        all_terms,
        grad_mask if needs_grad else 0,
        chunk_rows,
        logits_dtype == "float32",
        delta,
    )
    lead = hidden.shape[:-1]
    terms = {t: losses[fused.TERM_SLOTS[t]].view(lead) for t in loss_terms}
    return loss, fused.FusedOutputs(
        losses={name: fused.compose_loss(name, terms) for name in names},
        metrics={t: losses[fused.TERM_SLOTS[t]].view(lead) for t in metric_terms},
        pred_ids=argmax[0].view(lead),
        target_ids=argmax[1].view(lead),
    )
