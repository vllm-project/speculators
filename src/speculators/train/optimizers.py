"""Optimizer construction for speculator training.

Provides a single entry point, :func:`build_optimizers`, that returns the list of
optimizers the trainer should drive. The default ("adamw") returns a single AdamW
optimizer over all parameters, preserving the historical behavior. The "muon" option
returns two optimizers: ``torch.optim.Muon`` over the supported 2D weight matrices and
``torch.optim.AdamW`` over everything else. The AdamW optimizer has a separate
parameter group at ``muon_lr`` for the 2D matrices Muon excludes (embeddings, LM
heads, and other vocabulary-indexed tables), preserving their larger learning rate
without requiring another optimizer.

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
) -> tuple[
    list[tuple[str, Tensor]], list[tuple[str, Tensor]], list[tuple[str, Tensor]]
]:
    """Split trainable parameters by optimizer and AdamW hyperparameters.

    A parameter goes to Muon iff it requires gradients, is a 2D matrix with both
    dimensions > 1, and is not an embedding, codebook, or vocabulary-output weight. A
    semantically excluded 2D matrix goes to ``muon_excluded_params`` so AdamW can
    update it with Muon's learning rate and weight decay. Everything else -- norms,
    biases, and degenerate 2D weights (``[1, N]`` / ``[N, 1]`` vectors) -- goes to the
    base AdamW group; Muon orthogonalizes matrices, not vectors, and crashes on them
    under FSDP2.

    :param model: The model whose parameters should be partitioned.
    :return: A ``(muon_params, adamw_params, muon_excluded_params)`` tuple.
    """
    muon_params: list[tuple[str, Tensor]] = []
    adamw_params: list[tuple[str, Tensor]] = []
    muon_excluded_params: list[tuple[str, Tensor]] = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.ndim != _MATRIX_NDIM or min(param.shape) == 1:
            adamw_params.append((name, param))
        elif any(hint in name for hint in _ADAMW_NAME_HINTS):
            muon_excluded_params.append((name, param))
        else:
            muon_params.append((name, param))
    return muon_params, adamw_params, muon_excluded_params


def build_optimizers(model: Module, config) -> list[torch.optim.Optimizer]:
    """Build the optimizer(s) for a training run based on ``config.optimizer``.

    :param model: The model to optimize.
    :param config: A ``TrainerConfig`` holding the optimizer hyperparameters.
    :return: A list of optimizers for the trainer to step in tandem. The default
        "adamw" returns a single optimizer; "muon" returns ``[Muon, AdamW]`` when
        both parameter types are present.
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
        muon_params, adamw_params, muon_excluded_params = split_named_params_for_muon(
            model
        )
        logger.info(
            "Muon optimizer: %d via Muon, %d via AdamW, %d excluded matrices "
            "via the AdamW muon_lr group.",
            len(muon_params),
            len(adamw_params),
            len(muon_excluded_params),
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
        adamw_param_groups = []
        if adamw_params:
            adamw_param_groups.append({"params": adamw_params})
        if muon_excluded_params:
            adamw_param_groups.append(
                {
                    "params": muon_excluded_params,
                    "lr": config.muon_lr,
                    "weight_decay": config.muon_weight_decay,
                }
            )
        if adamw_param_groups:
            optimizers.append(
                torch.optim.AdamW(
                    adamw_param_groups,
                    lr=config.lr,
                    weight_decay=config.weight_decay,
                )
            )
        if not optimizers:
            raise ValueError("No trainable parameters found to optimize.")
        return optimizers

    raise ValueError(f"Unsupported optimizer: {config.optimizer!r}")
