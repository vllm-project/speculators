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

The "dion3" option substitutes Dion3 (``microsoft/dion``) for Muon over the identical
parameter split. Dion3 orthogonalizes only a ``dion_fraction`` of the momentum matrix's
rows per step and megabatches the sharded transfer, which is why its advantage grows
with world size: Muon has to reassemble whole matrices from their shards to run
Newton-Schulz, so sharding buys it nothing. ``dion`` is not published on PyPI and is
therefore imported lazily rather than declared as a dependency; install it with
``pip install git+https://github.com/microsoft/dion.git``.
"""

import functools
import logging

import torch
from torch import Tensor
from torch.nn import Module

logger = logging.getLogger("speculators")

# Names of parameters that are 2D but should still be optimized with AdamW rather than
# Muon, following the convention from Keller Jordan's Muon (embeddings and the output
# head are excluded from the orthogonalized update).
_ADAMW_NAME_HINTS = ("embed_tokens", "lm_head")

# Muon only orthogonalizes 2D weight matrices.
_MATRIX_NDIM = 2


def split_named_params_for_muon(
    model: Module,
) -> tuple[list[tuple[str, Tensor]], list[tuple[str, Tensor]]]:
    """Split a model's trainable parameters into Muon and AdamW groups.

    A parameter goes to Muon iff it requires gradients, is a 2D matrix with both
    dimensions > 1, and is not an embedding or LM-head weight; everything else goes to
    AdamW. Degenerate 2D weights (``[1, N]`` / ``[N, 1]`` vectors) route to AdamW --
    Muon orthogonalizes matrices, not vectors, and crashes on them under FSDP2.

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
        return optimizers

    if config.optimizer == "dion3":
        return _build_dion3(model, config)

    raise ValueError(f"Unsupported optimizer: {config.optimizer!r}")


def _device_mesh_of(params: list[Tensor]) -> object | None:
    """Return the device mesh the parameters are sharded over, if any.

    Taken from the parameters themselves rather than rebuilt with
    ``init_device_mesh``, so it is by construction the same mesh ``fully_shard``
    used. Without it Dion3 silently runs its single-device orthonormalization
    path, which is where its multi-GPU advantage comes from.
    """
    for param in params:
        mesh = getattr(param, "device_mesh", None)
        if mesh is not None:
            return mesh
    return None


def _static_shape_step(optimizer: torch.optim.Optimizer) -> torch.optim.Optimizer:
    """Run ``optimizer.step`` with dynamic shapes disabled.

    torch 2.13's inductor miscompiles Dion3's ``@torch.compile(fullgraph=True)``
    per-neuron normalization once dynamo promotes its shapes to dynamic, which
    happens on the second distinct matrix shape in a step: the generated Triton
    kernel reads a value defined inside the reduction loop from the epilogue
    after that loop closes, and fails to compile with
    ``NameError: tmp<N> is not defined``.

    An optimizer's shapes are fixed by the parameter list, so there is nothing to
    gain from dynamic shapes here. Scoping the setting to ``step`` rather than
    setting it process-wide leaves the model's own ``torch.compile`` alone --
    setting it globally measurably inflates the forward pass.
    """
    inner = optimizer.step

    @functools.wraps(inner)
    def step(*args, **kwargs):
        with torch._dynamo.config.patch(  # noqa: SLF001
            automatic_dynamic_shapes=False, cache_size_limit=64
        ):
            return inner(*args, **kwargs)

    optimizer.step = step  # type: ignore[method-assign]
    return optimizer


def _build_dion3(model: Module, config) -> list[torch.optim.Optimizer]:
    """Build a single Dion3 optimizer covering both parameter groups."""
    try:
        # Imported lazily on purpose: dion is not on PyPI, so it cannot be a
        # declared dependency and may legitimately be absent.
        from dion import Dion3  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover - depends on optional install
        raise ImportError(
            "--optimizer dion3 requires the 'dion' package, which is not published "
            "on PyPI. Install it with:\n"
            "    pip install git+https://github.com/microsoft/dion.git"
        ) from exc

    matrix_named, scalar_named = split_named_params_for_muon(model)
    if not matrix_named:
        raise ValueError("No trainable 2D parameters found to optimize.")
    matrix = [p for _, p in matrix_named]
    scalar = [p for _, p in scalar_named]

    logger.info(
        "Dion3 optimizer: %d 2D params via Dion3 (fraction=%s, selection_scope=%s), "
        "%d params via AdamW.",
        len(matrix),
        config.dion_fraction,
        config.dion_selection_scope,
        len(scalar),
    )

    # Dion3 consumes both groups itself, so this returns one optimizer where the
    # muon path returns two. The trainer and checkpointer both take a list.
    param_groups: list[dict] = [
        {
            "params": matrix,
            "algorithm": "nordion2",
            "lr": config.muon_lr,
            "weight_decay": config.muon_weight_decay,
        }
    ]
    if scalar:
        param_groups.append(
            {
                "params": scalar,
                "algorithm": "adamw",
                "lr": config.lr,
                "weight_decay": config.weight_decay,
            }
        )

    optimizer = Dion3(
        param_groups,
        lr=config.muon_lr,
        mu=config.muon_momentum,
        weight_decay=config.muon_weight_decay,
        fraction=config.dion_fraction,
        # dion's "rms_norm" is 0.2*sqrt(max(fan_out, fan_in)), the same expression
        # as torch Muon's "match_rms_adamw", so a given --muon-lr means the same
        # effective step size on both. dion's own default is a different scale.
        adjust_lr="rms_norm",
        selection_scope=config.dion_selection_scope,
        distributed_mesh=_device_mesh_of(matrix),
    )
    return [_static_shape_step(optimizer)]
