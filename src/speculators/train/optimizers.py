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

# Prefix of the LiLiCorr transformer head's parameters (``self.lilicorr`` on
# LiLiCorrDraftModel), used to give the head its own learning rate separate
# from the DFlash backbone.
_LILICORR_HEAD_PREFIX = "lilicorr."


def _split_by_prefix(
    named_params: list[tuple[str, Tensor]], prefix: str
) -> tuple[list[tuple[str, Tensor]], list[tuple[str, Tensor]]]:
    """Split named params into (not matching prefix, matching prefix)."""
    rest = [(n, p) for n, p in named_params if not n.startswith(prefix)]
    matching = [(n, p) for n, p in named_params if n.startswith(prefix)]
    return rest, matching


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
    # Falls back to config.lr when the model has no LiLiCorr head, or the head's LR
    # was never overridden (lilicorr_lr defaults to lr in that case).
    lilicorr_lr = config.lilicorr_lr if config.lilicorr_lr is not None else config.lr

    if config.optimizer == "adamw":
        backbone_params, head_params = _split_by_prefix(
            list(model.named_parameters()), _LILICORR_HEAD_PREFIX
        )
        param_groups = [{"params": [p for _, p in backbone_params], "lr": config.lr}]
        if head_params:
            param_groups.append(
                {"params": [p for _, p in head_params], "lr": lilicorr_lr}
            )
        return [
            torch.optim.AdamW(param_groups, weight_decay=config.weight_decay)
        ]

    if config.optimizer == "muon":
        muon_params, adamw_params = split_named_params_for_muon(model)
        muon_backbone, muon_head = _split_by_prefix(
            muon_params, _LILICORR_HEAD_PREFIX
        )
        adamw_backbone, adamw_head = _split_by_prefix(
            adamw_params, _LILICORR_HEAD_PREFIX
        )
        logger.info(
            "Muon optimizer: %d 2D params via Muon, %d params via AdamW "
            "(%d LiLiCorr head params at lr=%g).",
            len(muon_params),
            len(adamw_params),
            len(muon_head) + len(adamw_head),
            lilicorr_lr,
        )

        optimizers: list[torch.optim.Optimizer] = []
        if muon_backbone:
            optimizers.append(
                torch.optim.Muon(
                    muon_backbone,
                    lr=config.muon_lr,
                    momentum=config.muon_momentum,
                    weight_decay=config.muon_weight_decay,
                    ns_steps=config.muon_ns_steps,
                    adjust_lr_fn=config.muon_adjust_lr_fn,
                )
            )
        if muon_head:
            optimizers.append(
                torch.optim.Muon(
                    muon_head,
                    lr=lilicorr_lr,
                    momentum=config.muon_momentum,
                    weight_decay=config.muon_weight_decay,
                    ns_steps=config.muon_ns_steps,
                    adjust_lr_fn=config.muon_adjust_lr_fn,
                )
            )
        if adamw_backbone:
            optimizers.append(
                torch.optim.AdamW(
                    adamw_backbone,
                    lr=config.lr,
                    weight_decay=config.weight_decay,
                )
            )
        if adamw_head:
            optimizers.append(
                torch.optim.AdamW(
                    adamw_head,
                    lr=lilicorr_lr,
                    weight_decay=config.weight_decay,
                )
            )
        if not optimizers:
            raise ValueError("No trainable parameters found to optimize.")
        return optimizers

    raise ValueError(f"Unsupported optimizer: {config.optimizer!r}")
