"""Optimizer construction for speculator training.

Provides a single entry point, :func:`build_optimizers`, that returns the list of
optimizers the trainer should drive. The default ("adamw") returns a single AdamW
optimizer over all parameters, preserving the historical behavior. The "muon" option
returns two optimizers: :class:`BatchedMuon` (a ``torch.optim.Muon`` subclass whose
Newton-Schulz iterations batch same-shape matrices into ``bmm`` calls) over the 2D
weight matrices, and ``torch.optim.AdamW`` over everything else (norms, biases, and
the embedding / LM-head matrices, following standard Muon practice).

Muon works transparently for both single-GPU and multi-GPU (FSDP2) training: when the
model is sharded with ``fully_shard`` the parameters become ``DTensor``s.
:class:`BatchedMuon` detects that case and defers to the stock per-parameter
orthogonalization, which dispatches across ranks automatically.
"""

import logging
import math

import torch
from torch import Tensor
from torch.nn import Module

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


def _muon_adjusted_lr(lr: float, adjust_lr_fn: str | None, shape: torch.Size) -> float:
    """Muon's shape-based LR adjustment (mirrors ``torch.optim.Muon``).

    "original" scales by ``sqrt(max(1, rows / cols))`` of the stored shape;
    "match_rms_adamw" scales by ``0.2 * sqrt(max(rows, cols))`` so the
    orthogonalized update matches AdamW's update RMS.
    """
    if adjust_lr_fn is None or adjust_lr_fn == "original":
        ratio = math.sqrt(max(1.0, shape[0] / shape[1]))
    elif adjust_lr_fn == "match_rms_adamw":
        ratio = 0.2 * math.sqrt(max(shape))
    else:
        ratio = 1.0
    return lr * ratio


def _is_plain_dense(tensor: Tensor) -> bool:
    """Whether ``tensor`` is a plain dense Tensor/Parameter.

    ``DTensor`` (FSDP2) and other dispatching subclasses surface either as a
    type other than ``Tensor``/``Parameter`` or as a subclass-typed ``.data``
    (``Parameter`` wrapping a ``DTensor``); both must take the per-parameter
    path, where the Newton-Schulz matmuls dispatch across ranks.
    """
    return type(tensor) in (torch.Tensor, torch.nn.Parameter) and (
        type(tensor.data) is torch.Tensor
    )


class BatchedMuon(torch.optim.Muon):
    """``torch.optim.Muon`` with same-shape Newton-Schulz iterations batched.

    The stock optimizer orthogonalizes every 2D weight on its own: for the
    single-layer eagle3 draft that is 8 matrices x 5 Newton-Schulz steps x 3
    GEMM launches -- roughly 120 small kernels per optimizer step. Matrices
    that share a stored shape run the identical iteration as one ``bmm``
    batch, and the elementwise momentum / weight-decay / update ops run as
    ``foreach`` batches within each shape group, where Muon's shape-dependent
    LR adjustment is constant.

    The arithmetic per matrix is unchanged: cuBLAS runs the same kernel for a
    batch slice as for the standalone ``mm``, the Frobenius norm reduces over
    the same elements, and the ``foreach`` elementwise ops apply the same
    per-element formulas -- so the update is bitwise equal to the stock path
    on dense tensors. Parameters that are not plain dense tensors
    (``DTensor`` under FSDP2) fall back to the stock per-parameter loop.
    """

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            if not self._batched_group_step(group):
                self._per_parameter_group_step(group)
        return loss

    def _batched_group_step(self, group) -> bool:
        """Update one param group with batched ops; False if it must not batch."""
        params: list[Tensor] = []
        grads: list[Tensor] = []
        momentum_bufs: list[Tensor] = []
        self._init_group(group, params, grads, momentum_bufs)
        if not params:
            return True
        if not all(map(_is_plain_dense, (*params, *grads, *momentum_bufs))):
            return False

        lr = group["lr"]
        if isinstance(lr, Tensor):
            lr = lr.item()
        momentum = group["momentum"]

        # Momentum + Nesterov extrapolation, then decoupled weight decay: the
        # same elementwise ops torch.optim.Muon issues per parameter, fused.
        # (The foreach ops are torch's own batched-elementwise entry points.)
        torch._foreach_lerp_(momentum_bufs, grads, 1 - momentum)
        updates = (
            torch._foreach_lerp(grads, momentum_bufs, momentum)  # noqa: SLF001
            if group["nesterov"]
            else list(momentum_bufs)
        )
        torch._foreach_mul_(params, 1 - lr * group["weight_decay"])

        # Newton-Schulz over same-shape batches. Grouping by dtype and device
        # as well keeps the stack homogeneous; the LR adjustment depends only
        # on the stored shape, so it is constant inside each batch.
        by_key: dict[tuple, list[int]] = {}
        for i, update in enumerate(updates):
            key = (update.shape, update.dtype, update.device)
            by_key.setdefault(key, []).append(i)
        for (shape, _, _), idxs in by_key.items():
            transposed = shape[0] > shape[1]  # orient wide, like stock Muon
            stacked = torch.stack(
                [updates[i].T if transposed else updates[i] for i in idxs]
            )
            ortho = self._newtonschulz5(stacked, group)
            adjusted_lr = _muon_adjusted_lr(lr, group["adjust_lr_fn"], shape)
            torch._foreach_add_(
                [params[i] for i in idxs],
                [o.T if transposed else o for o in ortho.unbind(0)],
                alpha=-adjusted_lr,
            )
        return True

    @staticmethod
    def _newtonschulz5(stacked: Tensor, group) -> Tensor:
        """Quintic Newton-Schulz iteration over a ``[batch, m, n]`` stack (m <= n).

        Identical arithmetic to ``torch.optim.Muon``'s per-matrix iteration:
        normalize by the Frobenius norm, then repeatedly apply the quintic
        polynomial ``a*X + (b*G + c*G@G) @ X`` with ``G = X @ X.T``.
        """
        a, b, c = group["ns_coefficients"]
        ortho = stacked.bfloat16()
        norms = ortho.view(ortho.size(0), -1).norm(dim=1).clamp(min=group["eps"])
        ortho = ortho / norms.view(-1, 1, 1)
        for _ in range(group["ns_steps"]):
            gram = torch.bmm(ortho, ortho.transpose(1, 2))
            gram_update = torch.baddbmm(gram, gram, gram, beta=b, alpha=c)
            ortho = torch.baddbmm(ortho, gram_update, ortho, beta=a)
        return ortho

    def _per_parameter_group_step(self, group) -> None:
        """Stock per-parameter loop, for tensors the batched path cannot stack.

        Mirrors ``torch.optim.Muon``'s single-tensor implementation so
        ``DTensor`` parameters (FSDP2) keep the sharded Newton-Schulz
        dispatch the stock optimizer provides.
        """
        lr = group["lr"]
        if isinstance(lr, Tensor):
            lr = lr.item()
        a, b, c = group["ns_coefficients"]

        params: list[Tensor] = []
        grads: list[Tensor] = []
        momentum_bufs: list[Tensor] = []
        self._init_group(group, params, grads, momentum_bufs)

        for i, param in enumerate(params):
            grad = grads[i]
            momentum_bufs[i].lerp_(grad, 1 - group["momentum"])
            update = (
                grad.lerp(momentum_bufs[i], group["momentum"])
                if group["nesterov"]
                else momentum_bufs[i]
            )

            ortho = update.bfloat16()
            if update.size(0) > update.size(1):
                ortho = ortho.T
            ortho.div_(ortho.norm().clamp(min=group["eps"]))
            for _ in range(group["ns_steps"]):
                gram = ortho @ ortho.T
                gram_update = torch.addmm(gram, gram, gram, beta=b, alpha=c)
                ortho = torch.addmm(ortho, gram_update, ortho, beta=a)
            if update.size(0) > update.size(1):
                ortho = ortho.T

            param.mul_(1 - lr * group["weight_decay"])
            param.add_(
                ortho,
                alpha=-_muon_adjusted_lr(lr, group["adjust_lr_fn"], param.shape),
            )


def build_optimizers(model: Module, config) -> list[torch.optim.Optimizer]:
    """Build the optimizer(s) for a training run based on ``config.optimizer``.

    :param model: The model to optimize.
    :param config: A ``TrainerConfig`` holding the optimizer hyperparameters.
    :return: A list of optimizers for the trainer to step in tandem. The default
        "adamw" returns a single optimizer; "muon" returns ``[BatchedMuon, AdamW]``.
    """
    if config.optimizer == "adamw":
        return [
            torch.optim.AdamW(
                model.named_parameters(),
                lr=config.lr,
                weight_decay=config.weight_decay,
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
                BatchedMuon(
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
                )
            )
        if not optimizers:
            raise ValueError("No trainable parameters found to optimize.")
        return optimizers

    raise ValueError(f"Unsupported optimizer: {config.optimizer!r}")
