"""Bandwidth-bound argmax over the vocabulary dimension.

Draft models take a full-vocab argmax of every logits tensor (draft tokens,
accuracy targets). At 4096 x 151936 bf16 that is a 1.2 GB read, and
``torch.argmax`` reaches ~2 TB/s on an H100. The one-row-per-program Triton
kernel below streams the row with per-lane running maxima and reduces once
at the end, reaching ~3 TB/s.

The result is exactly ``torch.argmax(x, dim=-1)``: ties resolve to the first
index, and NaN counts as the maximum (the first NaN wins).
"""

# Triton kernels idiomatically use uppercase constexpr names.
# ruff: noqa: N803

import torch

try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover - exercised on Triton-less installs
    triton = None

_INT32_MAX = 2**31 - 1


if triton is not None:
    # Triton kernels may only read globals wrapped as tl.constexpr.
    _INT32_MAX_C = tl.constexpr(_INT32_MAX)

    @triton.autotune(
        configs=[
            triton.Config({"BLOCK_SIZE": block}, num_warps=warps)
            for block in (1024, 2048, 4096, 8192, 16384)
            for warps in (4, 8, 16)
            if 4 <= block // (warps * 32) <= 128  # noqa: PLR2004
        ],
        key=["n_cols"],
    )
    @triton.jit
    def _argmax_kernel(x_ptr, out_ptr, n_cols, BLOCK_SIZE: tl.constexpr):
        pid = tl.program_id(0).to(tl.int64)
        x_ptr += pid * n_cols
        best_val = tl.full((BLOCK_SIZE,), float("-inf"), tl.float32)
        best_idx = tl.zeros((BLOCK_SIZE,), tl.int32)
        for i in range(0, n_cols, BLOCK_SIZE):
            offsets = i + tl.arange(0, BLOCK_SIZE)
            x = tl.load(x_ptr + offsets, mask=offsets < n_cols, other=float("-inf"))
            x = x.to(tl.float32)
            # Strict > keeps each lane's first maximum; a NaN replaces any
            # non-NaN, and later NaNs never replace it.
            update = (x > best_val) | ((x != x) & (best_val == best_val))
            best_val = tl.where(update, x, best_val)
            best_idx = tl.where(update, offsets, best_idx)
        # Across lanes: the smallest index holding the row maximum, or the
        # smallest index holding a NaN if there is one.
        is_nan = best_val != best_val
        any_nan = tl.max(is_nan.to(tl.int32)) > 0
        row_max = tl.max(tl.where(is_nan, float("-inf"), best_val))
        candidate = tl.where(any_nan, is_nan, best_val == row_max)
        idx = tl.min(tl.where(candidate, best_idx, _INT32_MAX_C))
        tl.store(out_ptr + pid, idx.to(tl.int64))


def argmax_last_dim(x: torch.Tensor) -> torch.Tensor:
    """``torch.argmax(x, dim=-1)``, via the Triton kernel when it applies."""
    if (
        triton is None
        or not x.is_cuda
        or not x.is_floating_point()
        or not x.is_contiguous()
        or x.numel() == 0
        or x.shape[-1] > _INT32_MAX
    ):
        return torch.argmax(x, dim=-1)
    n_cols = x.shape[-1]
    rows = x.detach().view(-1, n_cols)
    out = torch.empty(rows.shape[0], device=x.device, dtype=torch.long)
    _argmax_kernel[(rows.shape[0],)](rows, out, n_cols)
    return out.view(x.shape[:-1])
