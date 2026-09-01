"""Fused Triton kernels for the dense draft/target logit losses.

One kernel pair serves every fused loss configured in ``losses.utils``: forward
streams the selected reduction from online-softmax statistics; backward
recomputes probabilities from five saved scalars per row and applies the
closed-form gradient. No ``[T, V]`` intermediate is ever materialized or
saved -- the memory win over the eager losses. ``OP`` picks the reduction at
Triton specialization time; ``nla`` and ``lk_hybrid`` are ``[1, T]`` torch
compositions of the primitives instead of kernel branches.

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
# BLOCK_SIZE) and inline block-size tuning thresholds; exempt this file from the
# pep8-naming and magic-value lints rather than fight those conventions.
# ruff: noqa: N803, N806, PLR2004

import torch
import triton
import triton.language as tl

MAX_FUSED_SIZE = 131072
# Ascend NPU's Unified Buffer (~192 KB) cannot fit the double-row load these
# kernels perform (logits + targets per block) beyond 4096 elements per block;
# triton-ascend raises "ub overflow" at BLOCK_SIZE >= 8192. CE is single-row
# but shares the same cap so all _FusedLoss OPs use one path.
MAX_FUSED_SIZE_NPU = 4096

# tl.constexpr instances: Triton kernels may only read globals wrapped this way.
_LOG2 = tl.constexpr(0.6931471805599453)

# Reduction selector, baked into the kernel at specialization time. Pass
# ``.value`` (a plain int) across the autograd boundary -- Dynamo cannot
# represent a constexpr object as an autograd.Function argument and would
# graph-break; the ``OP: tl.constexpr`` parameter re-wraps it in the kernel.
_OP_KL = tl.constexpr(0)
_OP_RKL = tl.constexpr(1)
_OP_JSD = tl.constexpr(2)
_OP_CE = tl.constexpr(3)
_OP_TV = tl.constexpr(4)

# stats [_N_STATS, n_rows]: [0]/[1] draft row max / sum-exp, [2]/[3] same for
# targets (unused by ce), [4] per-OP -- kl_div/rkl: row loss L | jsd:
# KL(dp||mix) | tv: s_s | ce: argmax as float32 (exact: vocab < 2**24).
_N_STATS = 5


def _calculate_settings(n, device):
    max_size = MAX_FUSED_SIZE_NPU if device.type == "npu" else MAX_FUSED_SIZE
    BLOCK_SIZE = min(triton.next_power_of_2(n), max_size)
    # triton-ascend does not require extra num_warps tuning
    num_warps = 4
    if BLOCK_SIZE >= 32768:
        num_warps = 32
    elif BLOCK_SIZE >= 8192:
        num_warps = 16
    elif BLOCK_SIZE >= 2048:
        num_warps = 8
    if getattr(torch.version, "hip", None) is not None:  # AMD wavefronts are 64-wide
        num_warps //= 2
    return BLOCK_SIZE, num_warps


@triton.jit
def _online_stats(row_ptr, n_cols, BLOCK_SIZE: tl.constexpr):
    """Online max (m) and sum-exp (d) over one row."""
    m = float("-inf")
    d = 0.0
    for i in range(0, n_cols, BLOCK_SIZE):
        offsets = i + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_cols
        x = tl.load(row_ptr + offsets, mask=mask, other=float("-inf")).cast(tl.float32)
        block_max = tl.max(tl.where(mask, x, float("-inf")))
        m_new = tl.maximum(m, block_max)
        d = d * tl.exp(m - m_new) + tl.sum(tl.where(mask, tl.exp(x - m_new), 0.0))
        m = m_new
    return m, d


@triton.jit
def _log_mix(ldp, ltp):
    """log((exp(ldp) + exp(ltp)) / 2): the JSD mixture, stable in log space."""
    mx = tl.maximum(ldp, ltp)
    return mx + tl.log(tl.exp(ldp - mx) + tl.exp(ltp - mx)) - _LOG2


@triton.jit
def loss_forward_kernel(  # noqa: C901 -- constexpr OP branches, pruned per instance
    logits_ptr,
    targets_ptr,
    loss_ptr,
    stats_ptr,
    stats_row,
    n_cols,
    OP: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """One row per program: online softmax stats, then the OP's reduction.

    Masked tail lanes can hold inf/nan before ``tl.where`` selects; they
    never enter an accumulator.
    """
    pid = tl.program_id(0).to(tl.int64)
    logits_ptr += pid * n_cols
    targets_ptr += pid * n_cols

    m_d, z_d = _online_stats(logits_ptr, n_cols, BLOCK_SIZE)
    lse_d = m_d + tl.log(z_d)
    tl.store(stats_ptr + 0 * stats_row + pid, m_d)
    tl.store(stats_ptr + 1 * stats_row + pid, z_d)

    if OP == _OP_CE:
        # Only the target argmax is needed; no target softmax stats.
        best_val = float("-inf")
        best_idx = 0
        for i in range(0, n_cols, BLOCK_SIZE):
            offsets = i + tl.arange(0, BLOCK_SIZE)
            mask = offsets < n_cols
            y = tl.load(targets_ptr + offsets, mask=mask, other=float("-inf")).cast(
                tl.float32
            )
            block_max = tl.max(y)
            block_arg = tl.argmax(y, axis=0)
            # strict > keeps the first occurrence, the torch.argmax convention
            update = block_max > best_val
            best_idx = tl.where(update, i + block_arg, best_idx)
            best_val = tl.where(update, block_max, best_val)
        logit_at_target = tl.load(logits_ptr + best_idx).cast(tl.float32)
        tl.store(loss_ptr + pid, lse_d - logit_at_target)
        # mypy reads traced Triton code as Python and types best_idx as int
        tl.store(stats_ptr + 4 * stats_row + pid, best_idx.to(tl.float32))  # type: ignore[attr-defined]
    else:
        m_t, z_t = _online_stats(targets_ptr, n_cols, BLOCK_SIZE)
        lse_t = m_t + tl.log(z_t)
        tl.store(stats_ptr + 2 * stats_row + pid, m_t)
        tl.store(stats_ptr + 3 * stats_row + pid, z_t)

        acc = 0.0
        extra = 0.0
        for i in range(0, n_cols, BLOCK_SIZE):
            offsets = i + tl.arange(0, BLOCK_SIZE)
            mask = offsets < n_cols
            x = tl.load(logits_ptr + offsets, mask=mask, other=0.0).cast(tl.float32)
            y = tl.load(targets_ptr + offsets, mask=mask, other=0.0).cast(tl.float32)
            ldp = x - lse_d
            ltp = y - lse_t
            dp = tl.exp(ldp)
            tp = tl.exp(ltp)
            if OP == _OP_KL:
                acc += tl.sum(tl.where(mask, tp * (ltp - ldp), 0.0))
            elif OP == _OP_RKL:
                acc += tl.sum(tl.where(mask, dp * (ldp - ltp), 0.0))
            elif OP == _OP_JSD:
                lmp = _log_mix(ldp, ltp)
                acc += tl.sum(tl.where(mask, dp * (ldp - lmp), 0.0))
                extra += tl.sum(tl.where(mask, tp * (ltp - lmp), 0.0))
            elif OP == _OP_TV:
                acc += tl.sum(tl.where(mask, tl.minimum(dp, tp), 0.0))
                extra += tl.sum(tl.where(mask & (dp <= tp), dp, 0.0))

        # Lint exemptions: `in`/merged compares aren't constexpr-foldable here.
        if OP == _OP_KL or OP == _OP_RKL:  # noqa: PLR1714, SIM109
            tl.store(loss_ptr + pid, acc)
            tl.store(stats_ptr + 4 * stats_row + pid, acc)
        elif OP == _OP_JSD:
            tl.store(loss_ptr + pid, 0.5 * (acc + extra))
            tl.store(stats_ptr + 4 * stats_row + pid, acc)
        elif OP == _OP_TV:
            tl.store(loss_ptr + pid, 1.0 - acc)
            tl.store(stats_ptr + 4 * stats_row + pid, extra)


@triton.jit
def loss_backward_kernel(
    logits_ptr,
    targets_ptr,
    grad_in_ptr,
    grad_out_ptr,
    stats_ptr,
    stats_row,
    n_cols,
    OP: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Recompute probabilities from the saved stats and apply the OP gradient."""
    pid = tl.program_id(0).to(tl.int64)
    logits_ptr += pid * n_cols
    grad_in_ptr += pid * n_cols

    go = tl.load(grad_out_ptr + pid).cast(tl.float32)
    if go == 0.0:
        for i in range(0, n_cols, BLOCK_SIZE):
            offsets = i + tl.arange(0, BLOCK_SIZE)
            mask = offsets < n_cols
            tl.store(grad_in_ptr + offsets, 0.0, mask=mask)
        return

    m_d = tl.load(stats_ptr + 0 * stats_row + pid)
    z_d = tl.load(stats_ptr + 1 * stats_row + pid)
    lse_d = m_d + tl.log(z_d)
    extra = tl.load(stats_ptr + 4 * stats_row + pid)
    if OP != _OP_CE:
        # CE passes None for targets_ptr.
        targets_ptr += pid * n_cols
        m_t = tl.load(stats_ptr + 2 * stats_row + pid)
        z_t = tl.load(stats_ptr + 3 * stats_row + pid)
        lse_t = m_t + tl.log(z_t)
    else:
        target_idx = extra.to(tl.int32)

    for i in range(0, n_cols, BLOCK_SIZE):
        offsets = i + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_cols
        x = tl.load(logits_ptr + offsets, mask=mask, other=0.0).cast(tl.float32)
        ldp = x - lse_d
        dp = tl.exp(ldp)
        if OP == _OP_CE:
            grad = dp - (offsets == target_idx).to(tl.float32)
        else:
            y = tl.load(targets_ptr + offsets, mask=mask, other=0.0).cast(tl.float32)
            ltp = y - lse_t
            tp = tl.exp(ltp)
            if OP == _OP_KL:
                grad = dp - tp
            elif OP == _OP_RKL:
                grad = dp * (ldp - ltp - extra)
            elif OP == _OP_JSD:
                grad = 0.5 * dp * (ldp - _log_mix(ldp, ltp) - extra)
            elif OP == _OP_TV:
                grad = -dp * ((dp <= tp).to(tl.float32) - extra)
        tl.store(grad_in_ptr + offsets, go * grad, mask=mask)


class _FusedLoss(torch.autograd.Function):
    """Shared autograd wrapper; ``op`` selects the loss reduction.

    Targets intentionally receive no gradient; use the eager implementation when
    target gradients are required.
    """

    @staticmethod
    def forward(ctx, logits, targets, op):
        B, T, V = logits.shape
        logits_flat = logits.contiguous().view(B * T, V)
        targets_flat = targets.contiguous().view(B * T, V)
        loss = torch.empty(B * T, device=logits.device, dtype=torch.float32)
        stats = torch.empty(_N_STATS, B * T, device=logits.device, dtype=torch.float32)
        BLOCK_SIZE, num_warps = _calculate_settings(V, logits_flat.device)
        loss_forward_kernel[(B * T,)](
            logits_flat,
            targets_flat,
            loss,
            stats,
            stats.stride(0),
            V,
            OP=op,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=num_warps,
        )
        # CE backward only needs the target argmax cached in stats.
        ctx.save_for_backward(
            logits_flat, None if op == _OP_CE.value else targets_flat, stats
        )
        ctx.op = op
        ctx.shape = (B, T, V)
        ctx.settings = (BLOCK_SIZE, num_warps)
        return loss.view(B, T)

    @staticmethod
    def backward(ctx, grad_output):
        logits_flat, targets_flat, stats = ctx.saved_tensors
        B, T, V = ctx.shape
        BLOCK_SIZE, num_warps = ctx.settings
        grad_in = torch.empty_like(logits_flat)
        loss_backward_kernel[(B * T,)](
            logits_flat,
            targets_flat,
            grad_in,
            grad_output.contiguous().view(-1),
            stats,
            stats.stride(0),
            V,
            OP=ctx.op,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=num_warps,
        )
        return grad_in.view(B, T, V), None, None


_NLA_EPS = 1e-5


def fused_kl_div_loss(logits, targets):
    """Per-position forward KL ``[1, T]``; fused twin of ``kl_div_loss``."""
    return _FusedLoss.apply(logits, targets, _OP_KL.value)


def fused_reverse_kl_div_loss(logits, targets):
    """Per-position reverse KL ``[1, T]``; fused twin of ``reverse_kl_div_loss``."""
    return _FusedLoss.apply(logits, targets, _OP_RKL.value)


def fused_js_div_loss(logits, targets):
    """Per-position JSD ``[1, T]``; fused twin of ``js_div_loss``."""
    return _FusedLoss.apply(logits, targets, _OP_JSD.value)


def fused_ce_loss(logits, targets):
    """Per-position CE vs argmax(targets) ``[1, T]``; fused twin of ``ce_loss``."""
    return _FusedLoss.apply(logits, targets, _OP_CE.value)


def fused_tv_loss(logits, targets):
    """Per-position TV distance ``[1, T]`` from draft/target logits (fused Triton)."""
    return _FusedLoss.apply(logits, targets, _OP_TV.value)


def fused_nla_loss(logits, targets):
    """Per-position negative-log-acceptance ``[1, T] = -log(alpha)``; composes on TV."""
    return -torch.log((1.0 - fused_tv_loss(logits, targets)).clamp_min(_NLA_EPS))


def fused_lk_hybrid_loss(logits, targets, eta: float = 3.0):
    """Per-position hybrid LK ``[1, T]``; fused KL + TV composed as in
    ``lk_hybrid_loss`` (blend weight detached).
    """
    kl = fused_kl_div_loss(logits, targets)
    tv = fused_tv_loss(logits, targets)
    weight = torch.exp(-eta * (1.0 - tv).detach())
    return weight * kl + (1.0 - weight) * tv
