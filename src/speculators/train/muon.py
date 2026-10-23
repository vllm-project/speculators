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
    ``_zeropower_via_newtonschulz``.
    """
    lr = _to_scalar(lr)
    if has_complex:
        raise ValueError("Complex parameters are not supported")

    for i, param in enumerate(params):
        grad = grads[i]
        if grad.ndim != _MATRIX_NDIM:
            raise ValueError("Param gradient must be a 2D matrix")

        buf = muon_momentum_bufs[i]
        buf.lerp_(grad, 1 - momentum)
        update = grad.lerp(buf, momentum) if nesterov else buf

        update = zeropower_via_newtonschulz(update, ns_coefficients, ns_steps, eps)

        adjusted_lr = _adjust_lr(lr, adjust_lr_fn, param.shape)

        param.mul_(1 - lr * weight_decay)
        param.add_(update, alpha=-adjusted_lr)


class Muon(torch.optim.Muon):
    """``torch.optim.Muon`` using :func:`zeropower_via_newtonschulz`.

    Hyperparameters, state (``momentum_buffer``), and the update rule are those
    of ``torch.optim.Muon``; only the orthogonalization's GEMMs change.
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
