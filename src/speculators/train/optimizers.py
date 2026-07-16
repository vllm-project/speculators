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

When ``fp32_masters=True`` each optimizer is wrapped in :class:`FP32MasterOptimizer`,
which keeps an fp32 master copy of the trainable parameters so that updates smaller
than a bf16 ulp still accumulate. This is the single-device counterpart of the FSDP2
mixed-precision setup (fp32 sharded params + bf16 ``param_dtype``), which needs no
wrapper — see ``Trainer.setup_model``.
"""

import logging
from collections.abc import Callable, Iterable

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

    A parameter goes to the Muon group iff it requires gradients, is 2D, and is not an
    embedding or LM-head weight. All other trainable parameters go to the AdamW group.

    :param model: The model whose parameters should be partitioned.
    :return: A ``(muon_params, adamw_params)`` tuple of named parameter lists.
    """
    muon_params: list[tuple[str, Tensor]] = []
    adamw_params: list[tuple[str, Tensor]] = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.ndim == _MATRIX_NDIM and not any(
            hint in name for hint in _ADAMW_NAME_HINTS
        ):
            muon_params.append((name, param))
        else:
            adamw_params.append((name, param))
    return muon_params, adamw_params


NamedParams = list[tuple[str, Tensor]]


class FP32MasterOptimizer(torch.optim.Optimizer):
    """Wrap an optimizer with fp32 master weights for low-precision training.

    The model's (typically bf16) parameters remain the ones used for forward and
    backward. This wrapper keeps an fp32 copy of each trainable parameter and, on
    every :meth:`step`, copies the low-precision gradients into fp32, steps the
    inner optimizer on the fp32 masters (so its moments are fp32 too), and writes
    the downcast result back into the model parameters. Updates smaller than a
    bf16 ulp therefore still accumulate across steps instead of rounding away.

    ``param_groups`` and ``state`` are shared with the inner optimizer, so LR
    schedulers can drive this wrapper directly. :meth:`state_dict` embeds the
    fp32 masters alongside the inner optimizer state so resume is lossless;
    legacy state dicts saved from a bare optimizer load transparently (torch
    casts their float state to the masters' fp32).
    """

    def __init__(
        self,
        named_params: Iterable[tuple[str, Tensor]],
        inner_factory: Callable[[NamedParams], torch.optim.Optimizer],
    ):
        # Intentionally no super().__init__: param_groups/state/defaults are
        # mirrored from the inner optimizer below so schedulers and state-dict
        # consumers see a single coherent optimizer.
        trainable = [(name, p) for name, p in named_params if p.requires_grad]
        if not trainable:
            raise ValueError("No trainable parameters found to optimize.")
        self._model_params = [p for _, p in trainable]
        self._masters = [p.detach().clone().to(torch.float32) for _, p in trainable]
        self.inner = inner_factory(
            [
                (name, master)
                for (name, _), master in zip(trainable, self._masters, strict=True)
            ]
        )
        self.defaults = self.inner.defaults
        self._mirror_inner()

    def _mirror_inner(self) -> None:
        self.param_groups = self.inner.param_groups
        self.state = self.inner.state

    @torch.no_grad()
    def step(self, closure=None):  # type: ignore[override]
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for param, master in zip(self._model_params, self._masters, strict=True):
            if param.grad is None:
                master.grad = None
                continue
            if master.grad is None:
                master.grad = torch.empty_like(master)
            master.grad.copy_(param.grad)  # exact bf16 -> fp32 upcast
        self.inner.step()
        for param, master in zip(self._model_params, self._masters, strict=True):
            param.data.copy_(master)  # fp32 -> model dtype downcast
        return loss

    def zero_grad(self, set_to_none: bool = True) -> None:  # type: ignore[override]
        for param in self._model_params:
            if param.grad is None:
                continue
            if set_to_none:
                param.grad = None
            else:
                param.grad.detach_()
                param.grad.zero_()

    def state_dict(self) -> dict:  # type: ignore[override]
        return {
            "inner": self.inner.state_dict(),
            "fp32_masters": [master.detach() for master in self._masters],
        }

    def load_state_dict(self, state_dict: dict) -> None:  # type: ignore[override]
        if "fp32_masters" in state_dict:
            saved_masters = state_dict["fp32_masters"]
            for master, saved in zip(self._masters, saved_masters, strict=True):
                master.detach().copy_(saved)
            self.inner.load_state_dict(state_dict["inner"])
            # Keep the low-precision model params consistent with the restored
            # masters (the model checkpoint is saved at reduced precision).
            with torch.no_grad():
                for param, master in zip(
                    self._model_params, self._masters, strict=True
                ):
                    param.data.copy_(master)
        else:
            # Legacy checkpoint saved from a bare optimizer over the model
            # params; torch casts its float state to the masters' fp32.
            try:
                self.inner.load_state_dict(state_dict)
            except ValueError as err:
                raise ValueError(
                    "Failed to load a legacy optimizer checkpoint into "
                    "FP32MasterOptimizer: it was saved by an older version "
                    "without fp32 master weights and with a different "
                    "parameter grouping. Remove the checkpoint's "
                    "optimizer_state_dict.pt to resume from the model weights "
                    "with a fresh optimizer state."
                ) from err
        # inner.load_state_dict rebinds param_groups/state; re-mirror them.
        self._mirror_inner()

    def add_param_group(self, param_group: dict) -> None:  # type: ignore[override]
        raise NotImplementedError(
            "FP32MasterOptimizer does not support adding param groups after "
            "construction."
        )


def build_optimizers(
    model: Module, config, fp32_masters: bool = False
) -> list[torch.optim.Optimizer]:
    """Build the optimizer(s) for a training run based on ``config.optimizer``.

    :param model: The model to optimize.
    :param config: A ``TrainerConfig`` holding the optimizer hyperparameters.
    :param fp32_masters: When True, wrap each optimizer in
        :class:`FP32MasterOptimizer` so updates accumulate in fp32 master weights.
        Used for single-device runs; FSDP2 runs keep the sharded params in fp32
        instead (see ``Trainer.setup_model``) and need no wrapper.
    :return: A list of optimizers for the trainer to step in tandem. The default
        "adamw" returns a single optimizer; "muon" returns ``[Muon, AdamW]``.
    """

    def adamw_factory(named_params: NamedParams) -> torch.optim.Optimizer:
        return torch.optim.AdamW(
            named_params,
            lr=config.lr,
            weight_decay=config.weight_decay,
        )

    def muon_factory(named_params: NamedParams) -> torch.optim.Optimizer:
        return torch.optim.Muon(
            named_params,
            lr=config.muon_lr,
            momentum=config.muon_momentum,
            weight_decay=config.muon_weight_decay,
            ns_steps=config.muon_ns_steps,
            adjust_lr_fn=config.muon_adjust_lr_fn,
        )

    def finalize(
        named_params: NamedParams,
        factory: Callable[[NamedParams], torch.optim.Optimizer],
    ) -> torch.optim.Optimizer:
        if fp32_masters:
            return FP32MasterOptimizer(named_params, factory)
        return factory(named_params)

    if config.optimizer == "adamw":
        return [finalize(list(model.named_parameters()), adamw_factory)]

    if config.optimizer == "muon":
        muon_params, adamw_params = split_named_params_for_muon(model)
        logger.info(
            "Muon optimizer: %d 2D params via Muon, %d params via AdamW.",
            len(muon_params),
            len(adamw_params),
        )

        optimizers: list[torch.optim.Optimizer] = []
        if muon_params:
            optimizers.append(finalize(muon_params, muon_factory))
        if adamw_params:
            optimizers.append(finalize(adamw_params, adamw_factory))
        if not optimizers:
            raise ValueError("No trainable parameters found to optimize.")
        return optimizers

    raise ValueError(f"Unsupported optimizer: {config.optimizer!r}")
