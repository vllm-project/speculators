"""Optimizer construction for speculator training.

Provides a single entry point, :func:`build_optimizers`, that returns the list of
optimizers the trainer should drive. The default ("adamw") returns a single AdamW
optimizer over all parameters, preserving the historical behavior. The "muon" option
returns two optimizers: ``torch.optim.Muon`` over the 2D weight matrices (which is all
Muon supports) and ``torch.optim.AdamW`` over everything else (norms, biases, and the
embedding / LM-head matrices, following standard Muon practice).

Muon works transparently for both single-GPU and multi-GPU (FSDP2) training: when the
model is sharded with ``fully_shard`` the parameters become ``DTensor``s and Muon's
Newton-Schulz orthogonalization dispatches across ranks automatically.

Muon's Newton-Schulz iteration forms the Gram matrix ``X @ X.T`` of every update each
step. That product is symmetric, so :class:`Muon` computes only its upper block
triangle and mirrors the rest, which skips a quarter of the dominant matmul for the
wide (MLP-shaped) matrices while leaving the update itself unchanged. The elementwise
halves of each update -- the momentum blend with its bf16 cast and norm, and the
decoupled weight decay with the orthogonalized step -- run as one compiled kernel each
instead of a chain of full-tensor passes.

AdamW runs its update in a single fused kernel by default. ``--optimizer-implementation
eager`` selects PyTorch's default (foreach) path instead, for accelerators or dtypes
where the fused kernel is unsupported.
"""

import logging

import torch
from torch import Tensor
from torch.distributed.tensor import DTensor
from torch.nn import Module
from torch.optim._muon import _adjust_lr, _zeropower_via_newtonschulz
from torch.optim.optimizer import _to_scalar

from speculators.models.utils import conditional_torch_compile

logger = logging.getLogger("speculators")

# Names of parameters that are 2D but should still be optimized with AdamW rather than
# Muon, following the convention from Keller Jordan's Muon (embeddings, embedding-like
# codebooks, Markov vocabulary factors, and output heads are excluded from the
# orthogonalized update).
_ADAMW_NAME_HINTS = (
    "embed_tokens",
    "lm_head",
    "codebook",
    "markov_w1",
    "markov_w2",
)

# Muon only orthogonalizes 2D weight matrices.
_MATRIX_NDIM = 2

# The block-triangular Gram product pays off only when the reduction dimension is long
# relative to the Gram's side (the MLP projections) and the blocks stay large; for
# square, short-K, or small matrices the blocks run less efficiently than one matmul.
_SYMMETRIC_GRAM_MIN_ASPECT = 2
_SYMMETRIC_GRAM_MIN_BLOCK = 1024
_SYMMETRIC_GRAM_BLOCKS = 4


def _symmetric_gram(x: Tensor, out: Tensor, num_blocks: int) -> Tensor:
    """Write ``x @ x.T`` into ``out`` computing only the upper block triangle.

    Column block ``j`` of the upper triangle is ``x[:end_j] @ x[j].T``; the strictly
    lower blocks are copied from their transposes.
    """
    block = x.size(0) // num_blocks
    for j in range(num_blocks):
        start, end = j * block, (j + 1) * block
        torch.mm(x[:end], x[start:end].T, out=out[:end, start:end])
    for j in range(1, num_blocks):
        start, end = j * block, (j + 1) * block
        out[start:end, :start].copy_(out[:start, start:end].T)
    return out


def _use_symmetric_gram(side: int, depth: int) -> bool:
    return (
        depth >= _SYMMETRIC_GRAM_MIN_ASPECT * side
        and side % _SYMMETRIC_GRAM_BLOCKS == 0
        and side >= _SYMMETRIC_GRAM_MIN_BLOCK * _SYMMETRIC_GRAM_BLOCKS
    )


def _newtonschulz(
    ortho_grad: Tensor, ns_coefficients: tuple[float, float, float], ns_steps: int
) -> Tensor:
    """``torch.optim.Muon``'s Newton-Schulz loop on an already normalized bf16 input.

    Same polynomial, coefficients, and bf16 precision as
    ``torch.optim._muon._zeropower_via_newtonschulz``; for shapes where it pays off
    the symmetric ``X @ X.T`` is formed block-triangularly.
    """
    a, b, c = ns_coefficients
    transposed = ortho_grad.size(0) > ortho_grad.size(1)
    if transposed:
        ortho_grad = ortho_grad.T
    side, depth = ortho_grad.shape
    symmetric = _use_symmetric_gram(side, depth)
    if symmetric:
        gram_matrix = ortho_grad.new_empty(side, side)
    for _ in range(ns_steps):
        if symmetric:
            _symmetric_gram(ortho_grad, gram_matrix, _SYMMETRIC_GRAM_BLOCKS)
        else:
            gram_matrix = ortho_grad @ ortho_grad.T
        gram_update = torch.addmm(
            gram_matrix, gram_matrix, gram_matrix, beta=b, alpha=c
        )
        ortho_grad = torch.addmm(ortho_grad, gram_update, ortho_grad, beta=a)
    return ortho_grad.T if transposed else ortho_grad


@conditional_torch_compile(fullgraph=True)
def _muon_momentum(
    grad: Tensor, buf: Tensor, momentum: float, nesterov: bool
) -> tuple[Tensor, Tensor]:
    """Update ``buf`` in place; return the bf16 Newton-Schulz input and its norm."""
    buf.lerp_(grad, 1 - momentum)
    update = grad.lerp(buf, momentum) if nesterov else buf
    update = update.bfloat16()
    return update, update.norm()


@conditional_torch_compile(fullgraph=True)
def _muon_apply(param: Tensor, update: Tensor, decay: Tensor, step: Tensor) -> None:
    """``param = param * decay + update * step`` in one pass.

    ``decay`` and ``step`` are 0-d CPU tensors so a changing learning rate does not
    recompile.
    """
    param.mul_(decay).add_(update.to(param.dtype) * step)


class Muon(torch.optim.Muon):
    """``torch.optim.Muon`` with a block-triangular Gram and fused elementwise passes.

    Hyperparameters, state (``momentum_buffer``), and the update rule are those of
    ``torch.optim.Muon``. DTensor (FSDP2) parameters take the stock eager path, whose
    matmuls dispatch across ranks.
    """

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            params: list[Tensor] = []
            grads: list[Tensor] = []
            bufs: list[Tensor] = []
            self._init_group(group, params, grads, bufs)

            lr = _to_scalar(group["lr"])
            momentum = group["momentum"]
            nesterov = group["nesterov"]
            coefficients, ns_steps = group["ns_coefficients"], group["ns_steps"]
            decay = 1 - lr * group["weight_decay"]
            for param, grad, buf in zip(params, grads, bufs, strict=True):
                adjusted_lr = _adjust_lr(lr, group["adjust_lr_fn"], param.shape)
                if isinstance(param, DTensor):
                    buf.lerp_(grad, 1 - momentum)
                    update = grad.lerp(buf, momentum) if nesterov else buf
                    update = _zeropower_via_newtonschulz(
                        update, coefficients, ns_steps, group["eps"]
                    )
                    param.mul_(decay)
                    param.add_(update, alpha=-adjusted_lr)
                    continue
                update, norm = _muon_momentum(grad, buf, momentum, nesterov)
                update.div_(norm.clamp(min=group["eps"]))
                update = _newtonschulz(update, coefficients, ns_steps)
                _muon_apply(
                    param, update, torch.tensor(decay), torch.tensor(-adjusted_lr)
                )
        return loss


def split_named_params_for_muon(
    model: Module,
) -> tuple[list[tuple[str, Tensor]], list[tuple[str, Tensor]]]:
    """Split a model's trainable parameters into Muon and AdamW groups.

    A parameter goes to Muon iff it requires gradients, is a 2D matrix with both
    dimensions > 1, and is not an embedding, codebook, or vocabulary-output weight;
    everything else goes to AdamW. Degenerate 2D weights (``[1, N]`` / ``[N, 1]``
    vectors) route to AdamW -- Muon orthogonalizes matrices, not vectors, and crashes
    on them under FSDP2.

    :param model: The model whose parameters should be partitioned.
    :return: A ``(muon_params, adamw_params)`` tuple of named parameter lists.
    """
    muon_params: list[tuple[str, Tensor]] = []
    adamw_params: list[tuple[str, Tensor]] = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if (
            param.ndim == _MATRIX_NDIM
            and min(param.shape) > 1  # exclude degenerate [1, N] / [N, 1] vectors
            and not any(hint in name for hint in _ADAMW_NAME_HINTS)
        ):
            muon_params.append((name, param))
        else:
            adamw_params.append((name, param))
    return muon_params, adamw_params


def build_optimizers(model: Module, config) -> list[torch.optim.Optimizer]:
    """Build the optimizer(s) for a training run based on ``config.optimizer``.

    :param model: The model to optimize.
    :param config: A ``TrainerConfig`` holding the optimizer hyperparameters.
    :return: A list of optimizers for the trainer to step in tandem. The default
        "adamw" returns a single optimizer; "muon" returns ``[Muon, AdamW]``.
    """
    fused = config.optimizer_implementation == "fused"

    if config.optimizer == "adamw":
        return [
            torch.optim.AdamW(
                model.named_parameters(),
                lr=config.lr,
                weight_decay=config.weight_decay,
                fused=fused,
            )
        ]

    if config.optimizer == "muon":
        muon_params, adamw_params = split_named_params_for_muon(model)
        logger.info(
            "Muon optimizer: %d 2D params via Muon, %d params via AdamW.",
            len(muon_params),
            len(adamw_params),
        )

        optimizers: list[torch.optim.Optimizer] = []
        if muon_params:
            optimizers.append(
                Muon(
                    muon_params,
                    lr=config.muon_lr,
                    momentum=config.muon_momentum,
                    weight_decay=config.muon_weight_decay,
                    ns_steps=config.muon_ns_steps,
                    adjust_lr_fn=config.muon_adjust_lr_fn,
                )
            )
        if adamw_params:
            optimizers.append(
                torch.optim.AdamW(
                    adamw_params,
                    lr=config.lr,
                    weight_decay=config.weight_decay,
                    fused=fused,
                )
            )
        if not optimizers:
            raise ValueError("No trainable parameters found to optimize.")
        return optimizers

    raise ValueError(f"Unsupported optimizer: {config.optimizer!r}")
