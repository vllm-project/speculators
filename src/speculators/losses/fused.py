"""Fused Triton kernels for the dense draft/target logit losses.

One kernel pair serves every fused loss configured in ``losses.utils``, and
every combination of them. Forward computes the draft and target
online-softmax statistics once per row and then streams all requested terms in
a single pass; backward recomputes probabilities from the saved per-row
scalars and writes one gradient, the per-row ``grad_output``-weighted sum of
each term's closed form. No ``[T, V]`` intermediate is materialized or saved
-- the memory win over the eager losses -- and a multi-term loss allocates one
``grad_in`` instead of one per term. ``TERMS`` is a bitmask of primitive terms
baked in at Triton specialization time; ``nla`` and ``lk_hybrid`` are
``[1, T]`` torch compositions of the primitives.

Per-row gradients (dp = softmax(logits), draft; tp = softmax(targets),
treated as constant), L the row loss:

    kl_div  dp - tp
    rkl     dp * (log dp - log tp - L)
    jsd     0.5 * dp * (log(dp / mix) - KL(dp || mix)),  mix = (dp + tp) / 2
    ce      dp - onehot(argmax tp)
    tv      -dp * (1[dp <= tp] - s_s),  s_s = sum of dp where dp <= tp

Derived from SpecForge specforge/core/loss.py (Apache-2.0, Unsloth/Liger
lineage); TV gradient sign convention matches Liger ops/tvd.py.
"""

# Triton kernels idiomatically use uppercase constexpr/dim names (B, T, V,
# BLOCK_SIZE); exempt this file from the pep8-naming lints rather than fight
# that convention.
# ruff: noqa: N803, N806

from collections.abc import Sequence
from dataclasses import dataclass

import torch
import triton
import triton.language as tl

from speculators.utils.util import is_npu_available

# Ascend NPU's Unified Buffer (~192 KB) cannot fit the double-row load these
# kernels perform (logits + targets per block) beyond 4096 elements per block;
# triton-ascend raises "ub overflow" at BLOCK_SIZE >= 8192. CE-only launches
# share the same cap so every term set uses one path.
MAX_FUSED_SIZE_NPU = 4096

# Search space for the row-per-program tile, autotuned per (vocab, TERMS) on
# the first launch. Each program keeps several fp32 vectors of BLOCK_SIZE /
# threads elements live at once, so what bounds the tile is elements *per
# thread*, not BLOCK_SIZE: register use grows at roughly two per element,
# exhausting the 255-register budget around 128, after which Triton spills to
# local memory. Measured registers/spill-bytes per thread, forward kernel
# (single KL term), V=151936, A100:
#
#     BLOCK_SIZE     warps=4      warps=8     warps=16     warps=32
#          4096     54/0         32/0         31/0         27/0
#          8192     96/6         48/0         32/0         31/0
#         16384    253/0        127/0         64/0         32/0
#         32768    255/126      254/0         97/0         53/0
#         65536     32/1222     255/152      128/38        64/38
#        131072     32/3748      32/1162      32/532       32/234
#
# Spilling is not disqualifying on its own: 8192 at 4 warps sits inside the
# bound, spills 6 B/thread and still runs within 8% of the winner. The cutoff
# is drawn where spills stop being incidental. Every tile above it spills at
# least 38 B/thread and lands between 1.3x and 22x the winner (kl, V=151936,
# 2048 rows, A100) -- among them the fixed BLOCK_SIZE=131072 this file used to
# select for any vocab > 65536, which at 234 B/thread ran the forward at
# 98 GB/s, 6% of A100 HBM peak. Bounding the space up front beats rediscovering
# that at every startup; test_autotuner_never_picks_a_spilling_tile times the
# unbounded space and checks that nothing above the cutoff wins.
# Which tile *within* that region wins is machine-dependent, which is what the
# autotuner is for; on an A100 it lands on 8192/8 warps at a 151936 vocab and
# 16384/16 at 32000.
_MAX_ELEMS_PER_THREAD = 128
_MIN_ELEMS_PER_THREAD = 4
_BLOCK_SIZES = (512, 1024, 2048, 4096, 8192, 16384, 32768)
_THREADS_PER_PROGRAM = (128, 256, 512, 1024)

# tl.constexpr instances: Triton kernels may only read globals wrapped this way.
_LOG2 = tl.constexpr(0.6931471805599453)

# Primitive terms and their output slot. A term set crosses the autograd
# boundary as a plain int bitmask (bit = 1 << slot) -- Dynamo cannot represent
# a constexpr object as an autograd.Function argument and would graph-break;
# the ``TERMS: tl.constexpr`` parameter re-wraps it in the kernel.
TERM_SLOTS = {"kl_div": 0, "rkl": 1, "jsd": 2, "ce": 3, "tv": 4}
_N_TERMS = len(TERM_SLOTS)
_BITS = {name: 1 << slot for name, slot in TERM_SLOTS.items()}
_KL = tl.constexpr(_BITS["kl_div"])
_RKL = tl.constexpr(_BITS["rkl"])
_JSD = tl.constexpr(_BITS["jsd"])
_CE = tl.constexpr(_BITS["ce"])
_TV = tl.constexpr(_BITS["tv"])
# Terms that need the target softmax; CE needs only the target argmax.
_DIST_MASK = _BITS["kl_div"] | _BITS["rkl"] | _BITS["jsd"] | _BITS["tv"]
_DIST = tl.constexpr(_DIST_MASK)

# stats [_N_STATS, n_rows], fp32: [0]/[1] draft row max / sum-exp, [2]/[3]
# the same for targets (distribution terms only), [4] rkl row loss L, [5] jsd
# KL(dp || mix), [6] ce target argmax as float32 (exact: vocab < 2**24),
# [7] tv s_s. Rows of terms not in TERMS are left unwritten.
_N_STATS = 8

# Shared with losses.fused_linear, which drives the same kernels on chunks.
TERM_BITS = _BITS
DIST_MASK = _DIST_MASK
N_TERMS = _N_TERMS
N_STATS = _N_STATS

# Named losses and the primitive terms each is composed from.
LOSS_TERMS: dict[str, tuple[str, ...]] = {
    "kl_div": ("kl_div",),
    "rkl": ("rkl",),
    "jsd": ("jsd",),
    "ce": ("ce",),
    "tv": ("tv",),
    "nla": ("tv",),
    "lk_hybrid": ("kl_div", "tv"),
}


def _tile_configs():
    """Register-resident (BLOCK_SIZE, num_warps) candidates for the autotuner."""
    if is_npu_available():
        # triton-ascend is Unified-Buffer bound rather than register bound:
        # exactly one tile fits, so there is nothing to search.
        return [triton.Config({"BLOCK_SIZE": MAX_FUSED_SIZE_NPU}, num_warps=4)]

    # AMD wavefronts are 64-wide, so a thread count is half as many warps.
    warp_size = 64 if getattr(torch.version, "hip", None) is not None else 32
    return [
        triton.Config({"BLOCK_SIZE": block}, num_warps=threads // warp_size)
        for threads in _THREADS_PER_PROGRAM
        for block in _BLOCK_SIZES
        if _MIN_ELEMS_PER_THREAD <= block // threads <= _MAX_ELEMS_PER_THREAD
    ]


def _prune_oversized_tiles(configs, nargs, **_):
    """Drop tiles wider than the vocabulary; those only add masked-off lanes.

    Dynamo traces this callback when the launch is inside a compiled region, so
    it has to stay to simple list operations -- a ``min(..., key=...)`` here
    fails to trace.
    """
    cap = triton.next_power_of_2(nargs["n_cols"])
    keep = [config for config in configs if config.kwargs["BLOCK_SIZE"] <= cap]
    if keep:
        return keep
    # Every tile is oversized for a very narrow vocab. They all still mask
    # correctly, but only the narrowest are worth compiling.
    floor = min(config.kwargs["BLOCK_SIZE"] for config in configs)
    return [config for config in configs if config.kwargs["BLOCK_SIZE"] == floor]


def _autotune_tile(kernel):
    """Autotune ``kernel`` over the register-resident tiles.

    Keyed on TERMS as well as the vocab: CE, KL and RKL stream the row once
    where JSD and TV stream it twice, and each extra term adds live
    registers, so term sets do not share an optimum. Each kernel gets its
    own config list: Dynamo tracks the autotuner's ``configs`` as a mutable
    list, and two kernels sharing one list fail to trace ("already tracked for
    mutation").
    """
    return triton.autotune(
        configs=_tile_configs(),
        key=["n_cols", "TERMS"],
        prune_configs_by={"early_config_prune": _prune_oversized_tiles},
    )(kernel)


@triton.jit
def _log_mix(ldp, ltp):
    """log((exp(ldp) + exp(ltp)) / 2): the JSD mixture, stable in log space."""
    mx = tl.maximum(ldp, ltp)
    return mx + tl.log(tl.exp(ldp - mx) + tl.exp(ltp - mx)) - _LOG2


@_autotune_tile
@triton.jit
def loss_forward_kernel(  # noqa: C901 -- constexpr TERMS branches
    logits_ptr,
    targets_ptr,
    loss_ptr,
    loss_row,
    argmax_ptr,
    argmax_row,
    stats_ptr,
    stats_row,
    n_cols,
    TERMS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """One row per program: shared softmax stats, then every TERMS reduction.

    Masked tail lanes can hold inf/nan before ``tl.where`` selects; they
    never enter an accumulator.
    """
    pid = tl.program_id(0).to(tl.int64)
    logits_ptr += pid * n_cols
    targets_ptr += pid * n_cols

    # Pass 1: online max (m) and sum-exp (z) of the draft row, the target
    # row's (distribution terms), and the first-index argmax of both rows, in
    # one stream of each. KL and RKL finish here too: with
    # tp = e^(y - m_t) / z_t,
    #
    #   KL(tp || dp) = sum tp (y - x) + lse_d - lse_t = s_kl / z_t + lse_d - lse_t,
    #   s_kl = sum e^(y - m_t) (y - x),
    #
    # and RKL likewise with the roles swapped. s rescales by exp(m_old - m_new)
    # under a running max exactly as z does, so it accumulates alongside the
    # online softmaxes; only JSD and TV need a second stream.
    m_d = float("-inf")
    z_d = 0.0
    m_t = float("-inf")
    z_t = 0.0
    s_kl = 0.0
    s_rkl = 0.0
    best_val_d = float("-inf")
    best_idx_d = 0
    best_val_t = float("-inf")
    best_idx_t = 0
    for i in range(0, n_cols, BLOCK_SIZE):
        offsets = i + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_cols
        # Masked lanes load as -inf, so they never win a max or an argmax.
        x = tl.load(logits_ptr + offsets, mask=mask, other=float("-inf")).cast(
            tl.float32
        )
        block_max_d = tl.max(tl.where(mask, x, float("-inf")))
        m_new = tl.maximum(m_d, block_max_d)
        scale_d = tl.exp(m_d - m_new)
        ex = tl.where(mask, tl.exp(x - m_new), 0.0)
        z_d = z_d * scale_d + tl.sum(ex)
        m_d = m_new
        # tl.argmax keeps the first maximum within a block and strict > keeps
        # the earliest block: torch.argmax's first-index convention.
        block_arg_d = tl.argmax(x, axis=0)
        update_d = block_max_d > best_val_d
        best_idx_d = tl.where(update_d, i + block_arg_d, best_idx_d)
        best_val_d = tl.where(update_d, block_max_d, best_val_d)
        y = tl.load(targets_ptr + offsets, mask=mask, other=float("-inf")).cast(
            tl.float32
        )
        if TERMS & _DIST:
            m_new_t = tl.maximum(m_t, tl.max(tl.where(mask, y, float("-inf"))))
            scale_t = tl.exp(m_t - m_new_t)
            ey = tl.where(mask, tl.exp(y - m_new_t), 0.0)
            z_t = z_t * scale_t + tl.sum(ey)
            m_t = m_new_t
            # Masked lanes: ey / ex are 0 but (y - x) is nan; where drops them.
            if TERMS & _KL:
                s_kl = s_kl * scale_t + tl.sum(tl.where(mask, ey * (y - x), 0.0))
            if TERMS & _RKL:
                s_rkl = s_rkl * scale_d + tl.sum(tl.where(mask, ex * (x - y), 0.0))
        block_max_t = tl.max(y)
        block_arg_t = tl.argmax(y, axis=0)
        update_t = block_max_t > best_val_t
        best_idx_t = tl.where(update_t, i + block_arg_t, best_idx_t)
        best_val_t = tl.where(update_t, block_max_t, best_val_t)

    lse_d = m_d + tl.log(z_d)
    tl.store(stats_ptr + 0 * stats_row + pid, m_d)
    tl.store(stats_ptr + 1 * stats_row + pid, z_d)
    # mypy reads traced Triton code as Python and types the indices as int
    tl.store(argmax_ptr + 0 * argmax_row + pid, best_idx_d.to(tl.int64))  # type: ignore[attr-defined]
    tl.store(argmax_ptr + 1 * argmax_row + pid, best_idx_t.to(tl.int64))  # type: ignore[attr-defined]

    if TERMS & _CE:
        logit_at_target = tl.load(logits_ptr + best_idx_t).cast(tl.float32)
        tl.store(loss_ptr + 3 * loss_row + pid, lse_d - logit_at_target)
        tl.store(stats_ptr + 6 * stats_row + pid, best_idx_t.to(tl.float32))  # type: ignore[attr-defined]

    if TERMS & _DIST:
        lse_t = m_t + tl.log(z_t)
        tl.store(stats_ptr + 2 * stats_row + pid, m_t)
        tl.store(stats_ptr + 3 * stats_row + pid, z_t)

        kl = s_kl / z_t + lse_d - lse_t
        rkl = s_rkl / z_d + lse_t - lse_d

        # Pass 2: JSD and TV need both finished log-sum-exps per element.
        jsd_draft = 0.0
        jsd_target = 0.0
        tv_overlap = 0.0
        tv_s = 0.0
        if TERMS & (_JSD | _TV):
            for i in range(0, n_cols, BLOCK_SIZE):
                offsets = i + tl.arange(0, BLOCK_SIZE)
                mask = offsets < n_cols
                x = tl.load(logits_ptr + offsets, mask=mask, other=0.0).cast(tl.float32)
                y = tl.load(targets_ptr + offsets, mask=mask, other=0.0).cast(
                    tl.float32
                )
                ldp = x - lse_d
                ltp = y - lse_t
                dp = tl.exp(ldp)
                tp = tl.exp(ltp)
                if TERMS & _JSD:
                    lmp = _log_mix(ldp, ltp)
                    jsd_draft += tl.sum(tl.where(mask, dp * (ldp - lmp), 0.0))
                    jsd_target += tl.sum(tl.where(mask, tp * (ltp - lmp), 0.0))
                if TERMS & _TV:
                    tv_overlap += tl.sum(tl.where(mask, tl.minimum(dp, tp), 0.0))
                    tv_s += tl.sum(tl.where(mask & (dp <= tp), dp, 0.0))

        if TERMS & _KL:
            tl.store(loss_ptr + 0 * loss_row + pid, kl)
        if TERMS & _RKL:
            tl.store(loss_ptr + 1 * loss_row + pid, rkl)
            tl.store(stats_ptr + 4 * stats_row + pid, rkl)
        if TERMS & _JSD:
            tl.store(loss_ptr + 2 * loss_row + pid, 0.5 * (jsd_draft + jsd_target))
            tl.store(stats_ptr + 5 * stats_row + pid, jsd_draft)
        if TERMS & _TV:
            tl.store(loss_ptr + 4 * loss_row + pid, 1.0 - tv_overlap)
            tl.store(stats_ptr + 7 * stats_row + pid, tv_s)


@_autotune_tile
@triton.jit
def loss_backward_kernel(  # noqa: C901 -- constexpr TERMS branches
    logits_ptr,
    targets_ptr,
    grad_in_ptr,
    grad_out_ptr,
    grad_out_row,
    stats_ptr,
    stats_row,
    n_cols,
    TERMS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Recompute probabilities from the saved stats; write the weighted sum of
    every TERMS gradient."""
    pid = tl.program_id(0).to(tl.int64)
    logits_ptr += pid * n_cols
    grad_in_ptr += pid * n_cols

    go_kl = 0.0
    go_rkl = 0.0
    go_jsd = 0.0
    go_ce = 0.0
    go_tv = 0.0
    if TERMS & _KL:
        go_kl = tl.load(grad_out_ptr + 0 * grad_out_row + pid).cast(tl.float32)
    if TERMS & _RKL:
        go_rkl = tl.load(grad_out_ptr + 1 * grad_out_row + pid).cast(tl.float32)
    if TERMS & _JSD:
        go_jsd = tl.load(grad_out_ptr + 2 * grad_out_row + pid).cast(tl.float32)
    if TERMS & _CE:
        go_ce = tl.load(grad_out_ptr + 3 * grad_out_row + pid).cast(tl.float32)
    if TERMS & _TV:
        go_tv = tl.load(grad_out_ptr + 4 * grad_out_row + pid).cast(tl.float32)

    go_any = tl.abs(go_kl) + tl.abs(go_rkl) + tl.abs(go_jsd) + tl.abs(go_ce)
    go_any += tl.abs(go_tv)
    if go_any == 0.0:
        for i in range(0, n_cols, BLOCK_SIZE):
            offsets = i + tl.arange(0, BLOCK_SIZE)
            mask = offsets < n_cols
            tl.store(grad_in_ptr + offsets, 0.0, mask=mask)
        return

    m_d = tl.load(stats_ptr + 0 * stats_row + pid)
    z_d = tl.load(stats_ptr + 1 * stats_row + pid)
    lse_d = m_d + tl.log(z_d)
    if TERMS & _DIST:
        # targets_ptr is None for CE-only launches; only offset it when used.
        targets_ptr += pid * n_cols
        m_t = tl.load(stats_ptr + 2 * stats_row + pid)
        z_t = tl.load(stats_ptr + 3 * stats_row + pid)
        lse_t = m_t + tl.log(z_t)
    if TERMS & _RKL:
        rkl_loss = tl.load(stats_ptr + 4 * stats_row + pid)
    if TERMS & _JSD:
        jsd_kl_mix = tl.load(stats_ptr + 5 * stats_row + pid)
    if TERMS & _CE:
        target_idx = tl.load(stats_ptr + 6 * stats_row + pid).to(tl.int32)
    if TERMS & _TV:
        tv_s = tl.load(stats_ptr + 7 * stats_row + pid)

    for i in range(0, n_cols, BLOCK_SIZE):
        offsets = i + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_cols
        x = tl.load(logits_ptr + offsets, mask=mask, other=0.0).cast(tl.float32)
        ldp = x - lse_d
        dp = tl.exp(ldp)
        grad = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
        if TERMS & _CE:
            grad += go_ce * (dp - (offsets == target_idx).to(tl.float32))
        if TERMS & _DIST:
            y = tl.load(targets_ptr + offsets, mask=mask, other=0.0).cast(tl.float32)
            ltp = y - lse_t
            tp = tl.exp(ltp)
            if TERMS & _KL:
                grad += go_kl * (dp - tp)
            if TERMS & _RKL:
                grad += go_rkl * (dp * (ldp - ltp - rkl_loss))
            if TERMS & _JSD:
                grad += go_jsd * (0.5 * dp * (ldp - _log_mix(ldp, ltp) - jsd_kl_mix))
            if TERMS & _TV:
                grad += go_tv * (-dp * ((dp <= tp).to(tl.float32) - tv_s))
        tl.store(grad_in_ptr + offsets, grad, mask=mask)


class _FusedLoss(torch.autograd.Function):
    """Shared autograd wrapper over the term bitmasks.

    ``terms`` selects what the forward computes; ``grad_terms`` (a subset)
    selects what backward differentiates. Terms outside ``grad_terms`` are
    metric-only: computed in the same pass, never differentiated.

    Returns ``[_N_TERMS, B, T]`` fp32 per-position losses indexed by
    ``TERM_SLOTS`` (slots outside ``terms`` are uninitialized and must not be
    read) and ``[2, B, T]`` int64 first-index argmax of logits and targets
    (non-differentiable). Targets intentionally receive no gradient; use the
    eager implementation when target gradients are required.
    """

    @staticmethod
    def forward(ctx, logits, targets, terms, grad_terms):
        B, T, V = logits.shape
        logits_flat = logits.contiguous().view(B * T, V)
        targets_flat = targets.contiguous().view(B * T, V)
        losses = torch.empty(_N_TERMS, B * T, device=logits.device, dtype=torch.float32)
        argmax = torch.empty(2, B * T, device=logits.device, dtype=torch.int64)
        stats = torch.empty(_N_STATS, B * T, device=logits.device, dtype=torch.float32)
        loss_forward_kernel[(B * T,)](
            logits_flat,
            targets_flat,
            losses,
            losses.stride(0),
            argmax,
            argmax.stride(0),
            stats,
            stats.stride(0),
            V,
            TERMS=terms,
        )
        # CE backward only needs the target argmax cached in stats, so when CE
        # is the only differentiated term the targets are released after
        # forward.
        ctx.save_for_backward(
            logits_flat, targets_flat if grad_terms & _DIST_MASK else None, stats
        )
        ctx.mark_non_differentiable(argmax)
        ctx.grad_terms = grad_terms
        ctx.shape = (B, T, V)
        return losses.view(_N_TERMS, B, T), argmax.view(2, B, T)

    @staticmethod
    def backward(ctx, grad_output, _grad_argmax):
        logits_flat, targets_flat, stats = ctx.saved_tensors
        if ctx.grad_terms == 0:
            return None, None, None, None
        B, T, V = ctx.shape
        grad_out = grad_output.contiguous().view(_N_TERMS, B * T)
        grad_in = torch.empty_like(logits_flat)
        loss_backward_kernel[(B * T,)](
            logits_flat,
            targets_flat,
            grad_in,
            grad_out,
            grad_out.stride(0),
            stats,
            stats.stride(0),
            V,
            TERMS=ctx.grad_terms,
        )
        return grad_in.view(B, T, V), None, None, None


_NLA_EPS = 1e-5


def terms_mask(terms: Sequence[str]) -> int:
    mask = 0
    for term in terms:
        mask |= _BITS[term]
    return mask


def fused_terms(
    logits: torch.Tensor, targets: torch.Tensor, terms: Sequence[str]
) -> dict[str, torch.Tensor]:
    """Per-position primitive losses ``[B, T]`` for each name in ``terms``.

    One forward kernel pass computes them all and one backward pass writes a
    single combined gradient. Names are keys of ``TERM_SLOTS``.
    """
    mask = terms_mask(terms)
    losses, _ = _FusedLoss.apply(logits, targets, mask, mask)
    return {term: losses[TERM_SLOTS[term]] for term in terms}


def compose_loss(
    name: str, terms: dict[str, torch.Tensor], eta: float = 3.0
) -> torch.Tensor:
    """Build a named loss from its primitive terms (see ``LOSS_TERMS``)."""
    if name == "nla":
        # negative log acceptance, -log(alpha) with alpha = 1 - TV
        return -torch.log((1.0 - terms["tv"]).clamp_min(_NLA_EPS))
    if name == "lk_hybrid":
        kl, tv = terms["kl_div"], terms["tv"]
        weight = torch.exp(-eta * (1.0 - tv).detach())
        return weight * kl + (1.0 - weight) * tv
    return terms[name]


@dataclass
class FusedOutputs:
    """Everything one fused kernel call produces for a loss config."""

    losses: dict[str, torch.Tensor]
    """Per-position ``[B, T]`` loss per requested name (differentiable)."""
    metrics: dict[str, torch.Tensor]
    """Per-position ``[B, T]`` primitive terms requested as metrics (detached)."""
    pred_ids: torch.Tensor
    """``[B, T]`` int64 ``argmax(logits, -1)``, first index on ties."""
    target_ids: torch.Tensor
    """``[B, T]`` int64 ``argmax(targets, -1)``, first index on ties."""


def fused_compound(
    logits: torch.Tensor,
    targets: torch.Tensor,
    names: Sequence[str],
    metric_terms: Sequence[str] = (),
) -> FusedOutputs:
    """Named losses, metric-only terms and both argmaxes from one kernel call.

    ``names`` are keys of ``LOSS_TERMS``; the union of their primitive terms
    is differentiated. ``metric_terms`` are primitive terms (keys of
    ``TERM_SLOTS``) that are computed in the same pass but never
    differentiated, e.g. TV for an acceptance-rate metric; a metric term that
    is also a loss term reuses that output.
    """
    loss_terms = sorted(
        {t for name in names for t in LOSS_TERMS[name]}, key=lambda t: _BITS[t]
    )
    grad_mask = terms_mask(loss_terms)
    losses, argmax = _FusedLoss.apply(
        logits, targets, grad_mask | terms_mask(metric_terms), grad_mask
    )
    terms = {t: losses[TERM_SLOTS[t]] for t in loss_terms}
    return FusedOutputs(
        losses={name: compose_loss(name, terms) for name in names},
        metrics={t: losses[TERM_SLOTS[t]].detach() for t in metric_terms},
        pred_ids=argmax[0],
        target_ids=argmax[1],
    )


def fused_compound_losses(
    logits: torch.Tensor, targets: torch.Tensor, names: Sequence[str]
) -> dict[str, torch.Tensor]:
    """Per-position ``[B, T]`` loss for each named loss, from one kernel call.

    ``names`` are keys of ``LOSS_TERMS``; the union of their primitive terms
    is computed in a single forward pass.
    """
    return fused_compound(logits, targets, names).losses


def fused_kl_div_loss(logits, targets):
    """Per-position forward KL ``[1, T]``; fused twin of ``kl_div_loss``."""
    return fused_terms(logits, targets, ("kl_div",))["kl_div"]


def fused_reverse_kl_div_loss(logits, targets):
    """Per-position reverse KL ``[1, T]``; fused twin of ``reverse_kl_div_loss``."""
    return fused_terms(logits, targets, ("rkl",))["rkl"]


def fused_js_div_loss(logits, targets):
    """Per-position JSD ``[1, T]``; fused twin of ``js_div_loss``."""
    return fused_terms(logits, targets, ("jsd",))["jsd"]


def fused_ce_loss(logits, targets):
    """Per-position CE vs argmax(targets) ``[1, T]``; fused twin of ``ce_loss``."""
    return fused_terms(logits, targets, ("ce",))["ce"]


def fused_tv_loss(logits, targets):
    """Per-position TV distance ``[1, T]`` from draft/target logits (fused Triton)."""
    return fused_terms(logits, targets, ("tv",))["tv"]


def fused_nla_loss(logits, targets):
    """Per-position negative-log-acceptance ``[1, T] = -log(alpha)``; composes on TV."""
    return compose_loss("nla", fused_terms(logits, targets, ("tv",)))


def fused_lk_hybrid_loss(logits, targets, eta: float = 3.0):
    """Per-position hybrid LK ``[1, T]``; fused KL + TV from one kernel pass,
    composed as in ``lk_hybrid_loss`` (blend weight detached).
    """
    terms = fused_terms(logits, targets, ("kl_div", "tv"))
    return compose_loss("lk_hybrid", terms, eta=eta)
