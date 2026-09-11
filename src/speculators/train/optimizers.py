"""Optimizer construction for speculator training.

Provides a single entry point, :func:`build_optimizers`, that returns the list of
optimizers the trainer should drive. The default ("adamw") returns a single AdamW
optimizer over all parameters, preserving the historical behavior. The "muon" option
returns two optimizers: ``torch.optim.Muon`` over the supported 2D weight matrices and
``torch.optim.AdamW`` over everything else. Fresh, output-adjacent token-transition
factors use a dedicated AdamW parameter group at ``muon_lr``; other Muon exclusions
retain the base AdamW hyperparameters.

Muon works transparently for both single-GPU and multi-GPU (FSDP2) training: when the
model is sharded with ``fully_shard`` the parameters become ``DTensor``s and Muon's
Newton-Schulz orthogonalization dispatches across ranks automatically.
"""

import logging

import torch
from torch import Tensor
from torch.nn import Embedding, Module

logger = logging.getLogger("speculators")

# Muon is intended for feature-to-feature matrices, not vocabulary-indexed tables.
# Embeddings are also detected structurally; these hints cover output heads and custom
# codebooks implemented as raw parameters.
_ADAMW_NAME_HINTS = (
    "embed_tokens",
    "lm_head",
    "codebook",
)

# Fresh, output-adjacent token-transition factors use AdamW for their vocabulary axes
# but keep Muon's larger learning rate and weight decay. Match complete suffixes so
# unrelated embeddings and codebooks remain in the base AdamW group.
_TRANSITION_PARAM_SUFFIXES = (
    "markov_w1.weight",
    "markov_w2.weight",
    "predecessor_codebook",
    "predecessor_codebook.weight",
    "successor_codebook",
    "successor_codebook.weight",
)

# Muon only orthogonalizes 2D weight matrices.
_MATRIX_NDIM = 2


def _matches_param_suffix(name: str, suffixes: tuple[str, ...]) -> bool:
    return any(name == suffix or name.endswith(f".{suffix}") for suffix in suffixes)


def split_named_params_for_muon(
    model: Module,
) -> tuple[
    list[tuple[str, Tensor]], list[tuple[str, Tensor]], list[tuple[str, Tensor]]
]:
    """Split trainable parameters by optimizer and AdamW hyperparameters.

    Fresh token-transition factors go to ``transition_params`` so AdamW can update
    them with Muon's learning rate and weight decay. Other embeddings, codebooks, and
    vocabulary-output weights use the base AdamW group. A remaining parameter goes to
    Muon iff it is a 2D matrix with both dimensions > 1; norms, biases, and degenerate
    2D weights (``[1, N]`` / ``[N, 1]`` vectors) use base AdamW because Muon
    orthogonalizes matrices, not vectors, and crashes on them under FSDP2.

    :param model: The model whose parameters should be partitioned.
    :return: A ``(muon_params, adamw_params, transition_params)`` tuple.
    """
    embedding_param_ids = {
        id(param)
        for module in model.modules()
        if isinstance(module, Embedding)
        for param in module.parameters(recurse=False)
    }

    muon_params: list[tuple[str, Tensor]] = []
    adamw_params: list[tuple[str, Tensor]] = []
    transition_params: list[tuple[str, Tensor]] = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if _matches_param_suffix(name, _TRANSITION_PARAM_SUFFIXES):
            transition_params.append((name, param))
        elif (
            param.ndim != _MATRIX_NDIM
            or min(param.shape) == 1
            or id(param) in embedding_param_ids
            or any(hint in name for hint in _ADAMW_NAME_HINTS)
        ):
            adamw_params.append((name, param))
        else:
            muon_params.append((name, param))
    return muon_params, adamw_params, transition_params


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
        muon_params, adamw_params, transition_params = split_named_params_for_muon(
            model
        )
        logger.info(
            "Muon optimizer: %d via Muon, %d via base AdamW, %d transition factors "
            "via AdamW at muon_lr.",
            len(muon_params),
            len(adamw_params),
            len(transition_params),
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
        if transition_params:
            adamw_param_groups.append(
                {
                    "params": transition_params,
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
