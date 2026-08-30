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
from torch.nn import Module

logger = logging.getLogger("speculators")

# Names of parameters that are 2D but should still be optimized with AdamW rather than
# Muon, following the convention from Keller Jordan's Muon (embeddings and the output
# head and embedding-like codebooks are excluded from the orthogonalized update).
_ADAMW_NAME_HINTS = ("embed_tokens", "lm_head", "codebook")

# Muon only orthogonalizes 2D weight matrices.
_MATRIX_NDIM = 2


def split_named_params_for_muon(
    model: Module,
) -> tuple[list[tuple[str, Tensor]], list[tuple[str, Tensor]]]:
    """Split a model's trainable parameters into Muon and AdamW groups.

    A parameter goes to Muon iff it requires gradients, is a 2D matrix with both
    dimensions > 1, and is not an embedding, codebook, or LM-head weight; everything
    else goes to AdamW. Degenerate 2D weights (``[1, N]`` / ``[N, 1]`` vectors) route
    to AdamW -- Muon orthogonalizes matrices, not vectors, and crashes on them under
    FSDP2.

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


def make_fp32_masters(model: Module) -> dict[str, Tensor]:
    """Give every low-precision trainable parameter an fp32 master copy.

    Returns ``{parameter name: master}``. Frozen parameters are skipped: they
    never receive an update, and leaving them in an optimizer group alongside
    fp32 masters would put two dtypes in one group, which torch's fused Adam
    path rejects.

    Under bf16 autocast the parameters themselves are bf16, so stepping them
    directly rounds every update to bf16's ~8-bit mantissa. Early in training an
    update is large enough to survive that; once the LR schedule decays it falls
    below the representable step at the weight's magnitude and is silently
    rounded away -- training flattens while the gradients stay healthy.
    """
    masters: dict[str, Tensor] = {}
    for name, param in model.named_parameters():
        if not param.requires_grad or param.dtype == torch.float32:
            continue
        masters[name] = param.detach().clone().float().requires_grad_(True)
    return masters


def _swap_in_masters(
    named: list[tuple[str, Tensor]], masters: dict[str, Tensor]
) -> list[tuple[str, Tensor]]:
    """Replace each parameter by its fp32 master, dropping the frozen ones.

    Frozen parameters never receive an update, and a group holding both them and
    the fp32 masters would mix dtypes, which torch's grouped Adam step rejects.
    """
    if not masters:
        return named
    swapped: list[tuple[str, Tensor]] = []
    for name, param in named:
        if name in masters:
            swapped.append((name, masters[name]))
        elif param.requires_grad and param.dtype == torch.float32:
            swapped.append((name, param))
    return swapped


def build_optimizers(
    model: Module, config
) -> tuple[list[torch.optim.Optimizer], list[tuple[Tensor, Tensor]]]:
    """Build the optimizer(s) for a training run based on ``config.optimizer``.

    :param model: The model to optimize.
    :param config: A ``TrainerConfig`` holding the optimizer hyperparameters.
    :return: The optimizers for the trainer to step in tandem, and the
        ``(parameter, fp32 master)`` pairs it must move gradients through --
        empty unless ``config.fp32_master_weights`` is set. The default "adamw"
        returns a single optimizer; "muon" returns ``[Muon, AdamW]``.
    """
    masters = (
        make_fp32_masters(model)
        if getattr(config, "fp32_master_weights", False)
        else {}
    )
    if masters:
        logger.info("fp32 master weights: %d parameters.", len(masters))

    pairs: list[tuple[Tensor, Tensor]] = [
        (param, masters[name])
        for name, param in model.named_parameters()
        if name in masters
    ]

    if config.optimizer == "adamw":
        return [
            torch.optim.AdamW(
                _swap_in_masters(list(model.named_parameters()), masters),
                lr=config.lr,
                weight_decay=config.weight_decay,
            )
        ], pairs

    if config.optimizer == "muon":
        muon_params, adamw_params = split_named_params_for_muon(model)
        muon_params = _swap_in_masters(muon_params, masters)
        adamw_params = _swap_in_masters(adamw_params, masters)
        logger.info(
            "Muon optimizer: %d 2D params via Muon, %d params via AdamW.",
            len(muon_params),
            len(adamw_params),
        )

        optimizers: list[torch.optim.Optimizer] = []
        if muon_params:
            optimizers.append(
                torch.optim.Muon(
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
        return optimizers, pairs

    raise ValueError(f"Unsupported optimizer: {config.optimizer!r}")
