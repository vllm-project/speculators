"""Muon with a Newton-Schulz iteration that exploits Gram-matrix symmetry.

Each Newton-Schulz step of ``torch.optim.Muon`` runs three bf16 matmuls::

    A = X @ X.T                    # symmetric
    B = b * A + c * A @ A          # symmetric (A @ A == A @ A.T)
    X = a * X + B @ X

The first two produce symmetric matrices, yet a dense GEMM computes every
element of them. :func:`zeropower_via_newtonschulz` computes only the upper
block-triangle of ``A`` and ``B`` -- ``SYMMETRIC_BLOCKS`` row blocks, each a
GEMM against the rows at and below it -- and mirrors it into the lower half
with one Triton launch. Each element is the same dot product cuBLAS computes
for the dense matmul, and on H100 the result is bit-identical to
``torch.optim.Muon``'s; 4 blocks do 10/16 of the Gram FLOPs.

Around those matmuls, each matrix's elementwise work runs as two Triton kernels
instead of five full-tensor passes: the momentum blend with its bf16 cast, and
the decoupled weight decay with the orthogonalized step. Both spell the
arithmetic the way torch's lerp and ``mul_``/``add_`` evaluate it, so the fused
path is bitwise equal to ``torch.optim.Muon`` as well.

The update, momentum, weight decay, learning-rate adjustment, and state layout
are ``torch.optim.Muon``'s unchanged, so checkpoints interoperate and the class
keeps the ``Muon`` name the trainer logs learning rates under.
"""

# ruff: noqa: N803

import torch
import triton
import triton.language as tl
from torch import Tensor
from torch.distributed.tensor import DTensor
from torch.optim._muon import (
    _adjust_lr,
    _to_scalar,
    _zeropower_via_newtonschulz,
)

__all__ = ["Muon", "zeropower_via_newtonschulz"]

# Elements per program of the fused momentum kernel, and the tile of the fused
# apply kernel (2D so a transposed update still reads in coalesced runs).
_MOMENTUM_BLOCK = 4096
_APPLY_TILE = 64

# Row blocks of the Gram product. Measured on H100 for a [4096, 12288] update:
# the dense X @ X.T takes 0.51 ms, 2 blocks 0.39 ms, 4 blocks 0.36 ms, and 8
# blocks lose it again to the narrow trailing GEMMs.
SYMMETRIC_BLOCKS = 4
# Below this many rows the Gram matmuls are too small for the split to pay.
_MIN_SYMMETRIC_ROWS = 2048
_MIRROR_TILE = 64

# Input validation mirroring ``torch.optim._muon._zeropower_via_newtonschulz``.
_MATRIX_NDIM = 2
_MAX_NS_STEPS = 100
_NS_COEFFICIENT_COUNT = 3


@triton.jit
def _mirror_lower_blocks_kernel(A, stride, block_rows, TILE: tl.constexpr):
    """Copy the transpose of the upper off-diagonal blocks into the lower ones."""
    ti = tl.program_id(0)
    tj = tl.program_id(1)
    # Diagonal blocks were computed whole by their GEMM; leave them untouched.
    if (ti * TILE) // block_rows <= (tj * TILE) // block_rows:
        return
    rows = ti * TILE + tl.arange(0, TILE)
    cols = tj * TILE + tl.arange(0, TILE)
    upper = tl.load(A + cols[None, :] * stride + rows[:, None])
    tl.store(A + rows[:, None] * stride + cols[None, :], upper)


def _mirror_lower_blocks(out: Tensor, block_rows: int) -> None:
    size = out.size(0)
    if out.is_cuda:
        grid = (size // _MIRROR_TILE, size // _MIRROR_TILE)
        _mirror_lower_blocks_kernel[grid](
            out, out.stride(0), block_rows, TILE=_MIRROR_TILE
        )
        return
    for start in range(block_rows, size, block_rows):
        out[start : start + block_rows, :start].copy_(
            out[:start, start : start + block_rows].T
        )


def _symmetric_gram(
    x: Tensor,
    out: Tensor,
    blocks: int,
    *,
    bias: Tensor | None = None,
    beta: float = 0.0,
    alpha: float = 1.0,
) -> Tensor:
    """``out = beta * bias + alpha * x @ x.T`` for a symmetric ``bias``.

    Computes the upper block-triangle with one GEMM per row block, then mirrors.
    """
    block_rows = x.size(0) // blocks
    for i in range(blocks):
        start, end = i * block_rows, (i + 1) * block_rows
        upper = out[start:end, start:]
        if bias is None:
            torch.mm(x[start:end], x[start:].T, out=upper)
        else:
            torch.addmm(
                bias[start:end, start:],
                x[start:end],
                x[start:].T,
                beta=beta,
                alpha=alpha,
                out=upper,
            )
    _mirror_lower_blocks(out, block_rows)
    return out


def _use_symmetric(rows: int) -> bool:
    return rows >= _MIN_SYMMETRIC_ROWS and rows % (SYMMETRIC_BLOCKS * _MIRROR_TILE) == 0


def zeropower_via_newtonschulz(
    grad: Tensor, ns_coefficients: tuple[float, float, float], ns_steps: int, eps: float
) -> Tensor:
    """``torch.optim._muon._zeropower_via_newtonschulz`` with symmetric Gram GEMMs.

    Falls back to the reference implementation for DTensors and for matrices
    whose short side is too small (or not tile-aligned) for the split to pay.
    """
    if isinstance(grad, DTensor) or not _use_symmetric(min(grad.shape)):
        return _zeropower_via_newtonschulz(grad, ns_coefficients, ns_steps, eps)
    if ns_steps >= _MAX_NS_STEPS:
        raise ValueError(
            "Number of steps must be less than 100 for computational efficiency"
        )
    if len(grad.shape) != _MATRIX_NDIM:
        raise ValueError("Input tensor gradient must be a 2D matrix")
    if len(ns_coefficients) != _NS_COEFFICIENT_COUNT:
        raise ValueError("Coefficients must be a tuple of exactly 3 values")
    a, b, c = ns_coefficients
    ortho_grad = grad.bfloat16()
    if grad.size(0) > grad.size(1):
        ortho_grad = ortho_grad.T
    ortho_grad.div_(ortho_grad.norm().clamp(min=eps))
    rows = ortho_grad.size(0)
    gram = ortho_grad.new_empty(rows, rows)
    gram_update = torch.empty_like(gram)
    for _ in range(ns_steps):
        _symmetric_gram(ortho_grad, gram, SYMMETRIC_BLOCKS)
        _symmetric_gram(gram, gram_update, SYMMETRIC_BLOCKS, bias=gram, beta=b, alpha=c)
        ortho_grad = torch.addmm(ortho_grad, gram_update, ortho_grad, beta=a)

    if grad.size(0) > grad.size(1):
        ortho_grad = ortho_grad.T
    return ortho_grad


@triton.jit
def _nesterov_momentum_kernel(
    GRAD,
    BUF,
    UPDATE,
    numel,
    weight,
    momentum,
    BLOCK: tl.constexpr,
    NESTEROV: tl.constexpr,
):
    """``buf.lerp_(grad, weight)``; ``update = grad.lerp(buf, momentum)`` in bf16.

    Spelled as torch's lerp kernel evaluates them (``weight < 0.5`` from the
    start, ``momentum >= 0.5`` back from the end with ``1 - momentum`` in fp32).
    """
    offs = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < numel
    grad = tl.load(GRAD + offs, mask=mask)
    buf = tl.load(BUF + offs, mask=mask)
    buf = buf + weight * (grad - buf)
    tl.store(BUF + offs, buf, mask=mask)
    update = buf - (buf - grad) * (1.0 - momentum) if NESTEROV else buf
    tl.store(UPDATE + offs, update.to(tl.bfloat16), mask=mask)


@triton.jit
def _apply_update_kernel(
    PARAM,
    UPDATE,
    rows,
    cols,
    update_row_stride,
    update_col_stride,
    decay,
    step_size,
    TILE: tl.constexpr,
):
    """``param.mul_(decay).add_(update, alpha=step_size)`` for a strided update."""
    r = tl.program_id(0).to(tl.int64) * TILE + tl.arange(0, TILE)
    c = tl.program_id(1).to(tl.int64) * TILE + tl.arange(0, TILE)
    mask = (r[:, None] < rows) & (c[None, :] < cols)
    param_ptrs = PARAM + r[:, None] * cols + c[None, :]
    param = tl.load(param_ptrs, mask=mask)
    update = tl.load(
        UPDATE + r[:, None] * update_row_stride + c[None, :] * update_col_stride,
        mask=mask,
    )
    # Round the decay before the FMA, as torch's mul_ then add_ do.
    decayed = param * decay
    tl.store(param_ptrs, tl.fma(update.to(tl.float32), step_size, decayed), mask=mask)


def _fusable(param: Tensor, grad: Tensor, buf: Tensor) -> bool:
    return (
        param.is_cuda
        and not isinstance(param, DTensor)
        and param.dtype == grad.dtype == buf.dtype == torch.float32
        and param.is_contiguous()
        and grad.is_contiguous()
        and buf.is_contiguous()
    )


def _momentum_update(
    grad: Tensor, buf: Tensor, momentum: float, nesterov: bool
) -> Tensor:
    """Momentum step returning the bf16 Newton-Schulz input in one pass."""
    update = torch.empty_like(grad, dtype=torch.bfloat16)
    numel = grad.numel()
    grid = (triton.cdiv(numel, _MOMENTUM_BLOCK),)
    _nesterov_momentum_kernel[grid](
        grad,
        buf,
        update,
        numel,
        1 - momentum,
        momentum,
        BLOCK=_MOMENTUM_BLOCK,
        NESTEROV=nesterov,
    )
    return update


def _apply_update(
    param: Tensor, update: Tensor, decay: float, step_size: float
) -> None:
    rows, cols = param.shape
    grid = (triton.cdiv(rows, _APPLY_TILE), triton.cdiv(cols, _APPLY_TILE))
    _apply_update_kernel[grid](
        param,
        update,
        rows,
        cols,
        update.stride(0),
        update.stride(1),
        decay,
        step_size,
        TILE=_APPLY_TILE,
    )


def _muon_update(
    params: list[Tensor],
    grads: list[Tensor],
    muon_momentum_bufs: list[Tensor],
    *,
    lr: float,
    weight_decay: float,
    momentum: float,
    nesterov: bool,
    ns_coefficients: tuple[float, float, float],
    ns_steps: int,
    eps: float,
    adjust_lr_fn: str | None,
    has_complex: bool,
) -> None:
    """``torch.optim._muon._single_tensor_muon`` over the symmetric iteration.

    The body is upstream's, with :func:`zeropower_via_newtonschulz` in place of
    ``_zeropower_via_newtonschulz`` and the elementwise halves fused where the
    tensors allow it.
    """
    lr = _to_scalar(lr)
    if has_complex:
        raise ValueError("Complex parameters are not supported")

    for i, param in enumerate(params):
        grad = grads[i]
        if grad.ndim != _MATRIX_NDIM:
            raise ValueError("Param gradient must be a 2D matrix")

        buf = muon_momentum_bufs[i]
        fused = _fusable(param, grad, buf)
        if fused:
            update = _momentum_update(grad, buf, momentum, nesterov)
        else:
            buf.lerp_(grad, 1 - momentum)
            update = grad.lerp(buf, momentum) if nesterov else buf

        update = zeropower_via_newtonschulz(update, ns_coefficients, ns_steps, eps)

        adjusted_lr = _adjust_lr(lr, adjust_lr_fn, param.shape)

        if fused:
            _apply_update(param, update, 1 - lr * weight_decay, -adjusted_lr)
        else:
            param.mul_(1 - lr * weight_decay)
            param.add_(update, alpha=-adjusted_lr)


class Muon(torch.optim.Muon):
    """``torch.optim.Muon`` using :func:`zeropower_via_newtonschulz`.

    On CUDA the per-matrix elementwise work is fused: the momentum update and
    the bf16 cast of the Newton-Schulz input run as one kernel, the weight decay
    and the orthogonalized update as another, instead of five full-matrix
    passes.

    Hyperparameters, state (``momentum_buffer``), and the update rule are those
    of ``torch.optim.Muon``; only the GEMMs and the kernel boundaries change.
    """

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            params: list[Tensor] = []
            grads: list[Tensor] = []
            momentum_bufs: list[Tensor] = []
            has_complex = self._init_group(group, params, grads, momentum_bufs)

            _muon_update(
                params,
                grads,
                momentum_bufs,
                lr=group["lr"],
                weight_decay=group["weight_decay"],
                momentum=group["momentum"],
                nesterov=group["nesterov"],
                ns_coefficients=group["ns_coefficients"],
                ns_steps=group["ns_steps"],
                eps=group["eps"],
                adjust_lr_fn=group["adjust_lr_fn"],
                has_complex=has_complex,
            )
        return loss
