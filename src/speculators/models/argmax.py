"""Bandwidth-bound last-dim argmax for ``[..., vocab]`` logits.

``torch.argmax`` over a ``[T, V]`` bf16 logit tensor at a ~152k vocab reads at
roughly 60% of H100 HBM bandwidth. A row-per-program Triton reduction that
streams each row in register-resident tiles gets close to the copy roofline.
The result is bitwise identical to ``torch.argmax(x, dim=-1)``: the first index
of the maximum wins ties, and a NaN (which ``torch.argmax`` treats as the
maximum) wins at its first occurrence.
"""

# Triton kernels idiomatically use uppercase constexpr names.
# ruff: noqa: N803

import torch

try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover - CPU-only installs
    triton = None

_INT32_MAX = 2147483647

if triton is not None:
    # Sentinel for "no NaN seen yet"; a constexpr so the kernel can read it.
    _NO_NAN = tl.constexpr(_INT32_MAX)

    @triton.autotune(
        configs=[
            triton.Config({"BLOCK_SIZE": block}, num_warps=warps)
            for block in (2048, 4096, 8192, 16384)
            for warps in (4, 8, 16)
        ],
        key=["n_cols"],
    )
    @triton.jit
    def _row_argmax_kernel(x_ptr, out_ptr, n_cols, stride, BLOCK_SIZE: tl.constexpr):
        row = tl.program_id(0).to(tl.int64)
        row_ptr = x_ptr + row * stride
        cols = tl.arange(0, BLOCK_SIZE)
        best_val = tl.full((), float("-inf"), tl.float32)
        best_idx = tl.full((), 0, tl.int32)
        first_nan = tl.full((), _NO_NAN, tl.int32)
        for start in range(0, n_cols, BLOCK_SIZE):
            idx = start + cols
            vals = tl.load(row_ptr + idx, mask=idx < n_cols, other=float("-inf"))
            vals = vals.to(tl.float32)
            is_nan = vals != vals
            first_nan = tl.minimum(first_nan, tl.min(tl.where(is_nan, idx, _NO_NAN)))
            vals = tl.where(is_nan, float("-inf"), vals)
            tile_max, tile_arg = tl.max(
                vals, axis=0, return_indices=True, return_indices_tie_break_left=True
            )
            # Strict > keeps the earliest tile on ties across tiles.
            better = tile_max > best_val
            best_idx = tl.where(better, start + tile_arg, best_idx)
            best_val = tl.where(better, tile_max, best_val)
        result = tl.where(first_nan != _NO_NAN, first_nan, best_idx)
        tl.store(out_ptr + row, result.to(tl.int64))


def row_argmax(x: torch.Tensor) -> torch.Tensor:
    """``torch.argmax(x, dim=-1)``, via a Triton kernel for CUDA tensors."""
    if (
        triton is None
        or not x.is_cuda
        or x.stride(-1) != 1
        or x.shape[-1] >= _INT32_MAX
    ):
        return torch.argmax(x, dim=-1)
    rows = x.detach().reshape(-1, x.shape[-1])
    out = torch.empty(rows.shape[0], dtype=torch.long, device=x.device)
    if rows.shape[0] > 0:
        _row_argmax_kernel[(rows.shape[0],)](rows, out, rows.shape[1], rows.stride(0))
    return out.reshape(x.shape[:-1])
