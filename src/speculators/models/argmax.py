"""Row-wise argmax over a wide last dimension (Triton), matching torch.argmax.

``torch.argmax(logits, dim=-1)`` over draft-vocabulary logits (e.g. 4096 x
151936) runs at roughly two thirds of HBM bandwidth on H100. This kernel
streams each row once with per-lane running maxima and runs near the read
roofline. Semantics match ``torch.argmax``: the first index of the maximum
wins, NaN compares greater than everything (first NaN wins), and an all
``-inf`` row gives index 0.
"""

# ruff: noqa: N803, N806

import torch
import triton
import triton.language as tl

_INT32_MAX = tl.constexpr(2147483647)


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": block}, num_warps=warps)
        for block in (1024, 2048, 4096, 8192)
        for warps in (4, 8, 16)
    ],
    key=["n_cols"],
)
@triton.jit
def _row_argmax_kernel(x_ptr, out_ptr, n_cols, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0).to(tl.int64)
    x_ptr += pid * n_cols
    best = tl.full([BLOCK_SIZE], float("-inf"), tl.float32)
    best_idx = tl.zeros([BLOCK_SIZE], tl.int32)
    nan_idx = tl.full([BLOCK_SIZE], _INT32_MAX, tl.int32)
    for i in range(0, n_cols, BLOCK_SIZE):
        offsets = i + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_cols
        x = tl.load(x_ptr + offsets, mask=mask, other=float("-inf")).cast(tl.float32)
        # strict > keeps each lane's first occurrence of its maximum
        update = x > best
        best = tl.where(update, x, best)
        best_idx = tl.where(update, offsets, best_idx)
        nan_idx = tl.minimum(nan_idx, tl.where(x != x, offsets, _INT32_MAX))
    row_max = tl.max(best, 0)
    idx = tl.min(tl.where(best == row_max, best_idx, _INT32_MAX), 0)
    idx = tl.where(row_max == float("-inf"), 0, idx)
    first_nan = tl.min(nan_idx, 0)
    idx = tl.where(first_nan != _INT32_MAX, first_nan, idx)
    tl.store(out_ptr + pid, idx.to(tl.int64))


_KERNEL_DTYPES = (torch.bfloat16, torch.float16, torch.float32)


def argmax_last_dim(x: torch.Tensor) -> torch.Tensor:
    """``torch.argmax(x, dim=-1)``; uses the Triton kernel for CUDA inputs."""
    n_cols = x.shape[-1] if x.dim() else 0
    if (
        not x.is_cuda
        or x.dtype not in _KERNEL_DTYPES
        or not x.is_contiguous()
        or x.numel() == 0
        or n_cols >= 2**31
    ):
        return torch.argmax(x, dim=-1)
    rows = x.numel() // n_cols
    out = torch.empty(rows, dtype=torch.int64, device=x.device)
    _row_argmax_kernel[(rows,)](x, out, n_cols)
    return out.view(x.shape[:-1])
