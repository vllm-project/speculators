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
"""

import logging

import torch
from torch import Tensor
from torch.distributed.tensor import DTensor
from torch.nn import Module
from torch.optim._muon import _adjust_lr, _zeropower_via_newtonschulz

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


class Muon(torch.optim.Muon):
    """``torch.optim.Muon`` with the Newton-Schulz iterations batched by shape.

    The stock implementation orthogonalizes one matrix at a time: ~5 x 3 small
    matmuls plus a handful of pointwise ops per parameter, which leaves a small
    draft model's optimizer step bound by kernel launches. Matrices that share a
    (transposed-to-wide) shape -- e.g. k/v/o projections, or MLP gate/up/down --
    are instead stacked and iterated with ``bmm``/``baddbmm``. The per-matrix
    normalization, momentum, weight decay and LR adjustment are the stock ops,
    applied per parameter, and state (``momentum_buffer``) is unchanged, so
    checkpoints are interchangeable with ``torch.optim.Muon``.

    Sharded (``DTensor``) parameters use the stock path, which already handles
    their distributed orthogonalization.
    """

    @torch.no_grad()
    def step(self, closure=None):
        if any(
            isinstance(p, DTensor)
            for group in self.param_groups
            for p in group["params"]
        ):
            return super().step(closure)

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            params: list[Tensor] = []
            grads: list[Tensor] = []
            bufs: list[Tensor] = []
            if self._init_group(group, params, grads, bufs):
                raise ValueError("Complex parameters are not supported")
            momentum = group["momentum"]

            updates = []
            for grad, buf in zip(grads, bufs, strict=True):
                if grad.ndim != 2:  # noqa: PLR2004
                    raise ValueError("Param gradient must be a 2D matrix")
                buf.lerp_(grad, 1 - momentum)
                updates.append(grad.lerp(buf, momentum) if group["nesterov"] else buf)

            orthogonalized = _batched_zeropower_via_newtonschulz(
                updates, group["ns_coefficients"], group["ns_steps"], group["eps"]
            )

            lr = group["lr"]
            for param, update in zip(params, orthogonalized, strict=True):
                adjusted_lr = _adjust_lr(lr, group["adjust_lr_fn"], param.shape)
                param.mul_(1 - lr * group["weight_decay"])
                param.add_(update, alpha=-adjusted_lr)
        return loss


def _batched_zeropower_via_newtonschulz(
    grads: list[Tensor],
    ns_coefficients: tuple[float, float, float],
    ns_steps: int,
    eps: float,
) -> list[Tensor]:
    """``_zeropower_via_newtonschulz`` over a list, batching same-shape matrices.

    Mirrors the stock op by op: bf16 cast, transpose to wide, per-matrix
    spectral normalization, then ``ns_steps`` quintic iterations, with each
    ``mm``/``addmm`` replaced by its batched twin.
    """
    a, b, c = ns_coefficients
    out: list[Tensor | None] = [None] * len(grads)
    by_shape: dict[torch.Size, list[int]] = {}
    for i, grad in enumerate(grads):
        wide = grad.size(0) <= grad.size(1)
        by_shape.setdefault(grad.shape if wide else grad.mT.shape, []).append(i)

    for indices in by_shape.values():
        if len(indices) == 1:
            i = indices[0]
            out[i] = _zeropower_via_newtonschulz(
                grads[i], ns_coefficients, ns_steps, eps
            )
            continue
        mats = []
        for i in indices:
            mat = grads[i].bfloat16()
            if grads[i].size(0) > grads[i].size(1):
                mat = mat.T
            mats.append(mat.div(mat.norm().clamp(min=eps)))
        ortho = torch.stack(mats)
        for _ in range(ns_steps):
            gram = ortho @ ortho.mT
            gram_update = torch.baddbmm(gram, gram, gram, beta=b, alpha=c)
            ortho = torch.baddbmm(ortho, gram_update, ortho, beta=a)
        for i, mat in zip(indices, ortho.unbind(0), strict=True):
            out[i] = mat.T if grads[i].size(0) > grads[i].size(1) else mat
    return out  # type: ignore[return-value]


def build_optimizers(model: Module, config) -> list[torch.optim.Optimizer]:
    """Build the optimizer(s) for a training run based on ``config.optimizer``.

    :param model: The model to optimize.
    :param config: A ``TrainerConfig`` holding the optimizer hyperparameters.
    :return: A list of optimizers for the trainer to step in tandem. The default
        "adamw" returns a single optimizer; "muon" returns ``[Muon, AdamW]``.
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
                )
            )
        if not optimizers:
            raise ValueError("No trainable parameters found to optimize.")
        return optimizers

    raise ValueError(f"Unsupported optimizer: {config.optimizer!r}")
