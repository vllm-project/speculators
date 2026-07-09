"""Fused Triton kernels for the acceptance-rate losses, computed from logits.

The primitive is the draft/target overlap ``alpha = sum_v min(p_v, q_v)`` (the
acceptance rate); ``fused_tv_loss`` returns ``1 - alpha`` and ``fused_nla_loss``
returns ``-log(alpha)``, both per-position ``[1, T]`` to match ``_LOSS_FN_MAP``.
Softmax is fused in, so the ``[T, V]`` distributions are never materialized or
saved for backward (the memory win over the eager losses); only five per-row
scalars are kept.

Derived from SpecForge specforge/core/loss.py (Apache-2.0, Unsloth/Liger lineage);
TV gradient sign convention matches Liger ops/tvd.py.
"""

import torch
import triton
import triton.language as tl

MAX_FUSED_SIZE = 131072


def _calculate_settings(n):
    BLOCK_SIZE = min(triton.next_power_of_2(n), MAX_FUSED_SIZE)
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
def tv_forward_kernel(
    logits_ptr,
    targets_ptr,
    loss_ptr,
    stats_ptr,
    stats_row,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    """TV forward: online softmax stats for both rows, then an overlap pass.

    ``stats`` is [5, n_rows] row-major; scalar k of row ``pid`` is at k*stats_row + pid.
    """
    pid = tl.program_id(0).to(tl.int64)
    logits_ptr += pid * n_cols
    targets_ptr += pid * n_cols

    m_p, d_p = _online_stats(logits_ptr, n_cols, BLOCK_SIZE)
    m_q, d_q = _online_stats(targets_ptr, n_cols, BLOCK_SIZE)

    overlap = 0.0
    s_s = 0.0
    for i in range(0, n_cols, BLOCK_SIZE):
        offsets = i + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_cols
        x = tl.load(logits_ptr + offsets, mask=mask, other=0.0).cast(tl.float32)
        y = tl.load(targets_ptr + offsets, mask=mask, other=0.0).cast(tl.float32)
        p = tl.exp(x - m_p) / d_p
        q = tl.exp(y - m_q) / d_q
        overlap += tl.sum(tl.where(mask, tl.minimum(p, q), 0.0))
        s_s += tl.sum(tl.where(mask & (p <= q), p, 0.0))

    tl.store(loss_ptr + pid, 1.0 - overlap)
    tl.store(stats_ptr + 0 * stats_row + pid, m_p)
    tl.store(stats_ptr + 1 * stats_row + pid, d_p)
    tl.store(stats_ptr + 2 * stats_row + pid, m_q)
    tl.store(stats_ptr + 3 * stats_row + pid, d_q)
    tl.store(stats_ptr + 4 * stats_row + pid, s_s)


@triton.jit
def tv_backward_kernel(
    logits_ptr,
    targets_ptr,
    grad_in_ptr,
    grad_out_ptr,
    stats_ptr,
    stats_row,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0).to(tl.int64)
    logits_ptr += pid * n_cols
    targets_ptr += pid * n_cols
    grad_in_ptr += pid * n_cols

    go = tl.load(grad_out_ptr + pid).cast(tl.float32)
    if go == 0.0:
        for i in range(0, n_cols, BLOCK_SIZE):
            offsets = i + tl.arange(0, BLOCK_SIZE)
            mask = offsets < n_cols
            tl.store(grad_in_ptr + offsets, 0.0, mask=mask)
        return

    m_p = tl.load(stats_ptr + 0 * stats_row + pid)
    d_p = tl.load(stats_ptr + 1 * stats_row + pid)
    m_q = tl.load(stats_ptr + 2 * stats_row + pid)
    d_q = tl.load(stats_ptr + 3 * stats_row + pid)
    s_s = tl.load(stats_ptr + 4 * stats_row + pid)

    for i in range(0, n_cols, BLOCK_SIZE):
        offsets = i + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_cols
        x = tl.load(logits_ptr + offsets, mask=mask, other=0.0).cast(tl.float32)
        y = tl.load(targets_ptr + offsets, mask=mask, other=0.0).cast(tl.float32)
        p = tl.exp(x - m_p) / d_p
        q = tl.exp(y - m_q) / d_q
        ind = (p <= q).to(tl.float32)
        grad = -go * p * (ind - s_s)
        tl.store(grad_in_ptr + offsets, grad, mask=mask)


class FusedTVLoss(torch.autograd.Function):
    @staticmethod
    def forward(ctx, logits, targets):
        B, T, V = logits.shape
        logits_flat = logits.contiguous().view(B * T, V)
        targets_flat = targets.contiguous().view(B * T, V)
        loss = torch.empty(B * T, device=logits.device, dtype=torch.float32)
        stats = torch.empty(5, B * T, device=logits.device, dtype=torch.float32)
        BLOCK_SIZE, num_warps = _calculate_settings(V)
        tv_forward_kernel[(B * T,)](
            logits_flat,
            targets_flat,
            loss,
            stats,
            stats.stride(0),
            V,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=num_warps,
        )
        ctx.save_for_backward(logits_flat, targets_flat, stats)
        ctx.shape = (B, T, V)
        ctx.settings = (BLOCK_SIZE, num_warps)
        return loss.view(B, T)

    @staticmethod
    def backward(ctx, grad_output):
        logits_flat, targets_flat, stats = ctx.saved_tensors
        B, T, V = ctx.shape
        BLOCK_SIZE, num_warps = ctx.settings
        grad_in = torch.empty_like(logits_flat)
        tv_backward_kernel[(B * T,)](
            logits_flat,
            targets_flat,
            grad_in,
            grad_output.contiguous().view(-1),
            stats,
            stats.stride(0),
            V,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=num_warps,
        )
        return grad_in.view(B, T, V), None


_EPS = 1e-5  # matches models/metrics.py neg_log_acceptance_loss


def fused_tv_loss(logits, targets):
    """Per-position TV distance ``[1, T]`` from draft/target logits (fused Triton)."""
    return FusedTVLoss.apply(logits, targets)


def fused_nla_loss(logits, targets):
    """Per-position negative-log-acceptance ``[1, T] = -log(alpha)``; composes on TV."""
    return -torch.log((1.0 - fused_tv_loss(logits, targets)).clamp_min(_EPS))
