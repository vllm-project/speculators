"""Standalone DFlash / DSpark loss head on synthetic inputs.

The loss head is everything between the drafter's final (post-norm) hidden
states and the scalar loss: the ``lm_head`` projection (plus DSpark's Markov
logit bias), the no-grad verifier projection that produces the targets, and
the model's own ``compute_metrics``. ``loss_head`` runs exactly that slice on
inputs from ``make_inputs``, so ``scripts/loss_head_benchmark.py`` can time it
and the golden tests can pin its numerics without building a model.
"""

from __future__ import annotations

import dataclasses
import functools

import torch
import torch.nn.functional as F  # noqa: N812

from speculators.losses import resolve_loss_config
from speculators.losses.utils import LossConfig
from speculators.models.dflash import metrics as dflash_metrics
from speculators.models.dspark import metrics as dspark_metrics

__all__ = [
    "IMPLEMENTATIONS",
    "MODELS",
    "WEIGHTINGS",
    "HeadConfig",
    "HeadInputs",
    "cached_loss_config",
    "loss_head",
    "make_inputs",
]

MODELS = ("dflash", "dspark")
WEIGHTINGS = ("fixed-exp-decay", "dpace")
IMPLEMENTATIONS = ("fused", "eager")
# Relative noise between the draft and verifier sides in make_inputs.
DRAFT_NOISE = 0.5
# Model defaults (models/{dflash,dspark}/config.py).
SAMPLE_FROM_ANCHOR = {"dflash": False, "dspark": True}


@dataclasses.dataclass(frozen=True)
class HeadConfig:
    """One loss-head configuration (a point of the benchmark matrix)."""

    model: str
    rows: int
    vocab: int
    hidden: int
    block_size: int
    markov_rank: int
    loss: str
    weighting: str
    implementation: str
    dtype: str

    @property
    def num_rows(self) -> int:
        """Rows rounded down to whole blocks (D-PACE reshapes per block)."""
        return (self.rows // self.block_size) * self.block_size


@dataclasses.dataclass
class HeadInputs:
    """Leaf tensors of the loss head; ``weight`` and ``verifier_weight`` are frozen."""

    hidden: torch.Tensor  # [N, H], requires grad
    weight: torch.Tensor  # [V, H], lm_head
    verifier_hidden: torch.Tensor  # [N, H], post-norm, pre-gathered
    verifier_weight: torch.Tensor  # [V, H], verifier_lm_head
    loss_mask: torch.Tensor  # [1, N]
    markov_input: torch.Tensor | None = None  # [N, r], requires grad (DSpark)
    markov_weight: torch.Tensor | None = None  # [V, r], requires grad (DSpark)

    def trainable(self) -> list[torch.Tensor]:
        tensors = [self.hidden, self.markov_input, self.markov_weight]
        return [t for t in tensors if t is not None]


def make_inputs(
    cfg: HeadConfig, device: torch.device | str, seed: int = 0
) -> HeadInputs:
    """Random loss-head inputs: logits of std ~2, draft close to the verifier.

    The draft side is the verifier side plus noise, as for a trained drafter,
    so argmax agreement (the accuracy metrics) is partial rather than ~0.
    """
    dtype = getattr(torch, cfg.dtype)
    n, h, v = cfg.num_rows, cfg.hidden, cfg.vocab
    gen = torch.Generator(device=device).manual_seed(seed)

    def randn(*shape: int, scale: float = 1.0) -> torch.Tensor:
        out = torch.randn(*shape, generator=gen, device=device, dtype=dtype)
        return out.mul_(scale)

    verifier_weight = randn(v, h, scale=2.0 / h**0.5)
    weight = verifier_weight + randn(v, h, scale=DRAFT_NOISE * 2.0 / h**0.5)
    verifier_hidden = randn(n, h)
    hidden = verifier_hidden + randn(n, h, scale=DRAFT_NOISE)
    inputs = HeadInputs(
        hidden=hidden.requires_grad_(True),
        weight=weight,
        verifier_hidden=verifier_hidden,
        verifier_weight=verifier_weight,
        loss_mask=(
            torch.rand(1, n, generator=gen, device=device) < 0.9  # noqa: PLR2004
        ).to(torch.long),
    )
    if not SAMPLE_FROM_ANCHOR[cfg.model]:
        # The model masks slot 0 of every block (the anchor) when it does not
        # sample from it; mirror that so the masked rows match training.
        inputs.loss_mask.view(-1, cfg.block_size)[:, 0] = 0
    if cfg.model == "dspark":
        r = cfg.markov_rank
        inputs.markov_input = randn(n, r).requires_grad_(True)
        inputs.markov_weight = randn(v, r, scale=0.1 / r**0.5).requires_grad_(True)
    return inputs


def loss_head(cfg: HeadConfig, inputs: HeadInputs) -> tuple[torch.Tensor, dict]:
    """Run the materialised loss head with the model's own ``compute_metrics``.

    Eager, without ``torch.autocast`` or ``torch.compile``: inputs are created
    in ``cfg.dtype``, as the trainer precasts the frozen heads. DSpark runs
    without its confidence head (per-row work, not ``[N, V]``).
    """
    loss_config = cached_loss_config(cfg.loss, cfg.implementation)
    n = cfg.num_rows
    with torch.no_grad():
        # verifier_lm_head(verifier_norm(h_v[:, idx])); the norm is outside the
        # head and the rows are already gathered.
        targets = F.linear(inputs.verifier_hidden, inputs.verifier_weight)
        targets = targets.unsqueeze(0)
    logits = F.linear(inputs.hidden, inputs.weight)

    if cfg.model == "dflash":
        return dflash_metrics.compute_metrics(
            logits.unsqueeze(0),
            targets,
            inputs.loss_mask,
            block_size=cfg.block_size,
            loss_config=loss_config,
            per_position_loss_weight=cfg.weighting,
            sample_from_anchor=SAMPLE_FROM_ANCHOR["dflash"],
        )

    if inputs.markov_input is None or inputs.markov_weight is None:
        raise ValueError("DSpark loss head needs markov_input and markov_weight")
    # DSpark: logits.view(nb, block, V) + markov_w2(u), as in dspark/core.py.
    num_blocks = n // cfg.block_size
    markov_bias = F.linear(inputs.markov_input, inputs.markov_weight)
    logits = (
        logits.view(num_blocks, cfg.block_size, -1)
        + markov_bias.view(num_blocks, cfg.block_size, -1)
    ).view(1, n, -1)
    tv_loss_fn = cached_loss_config("tv", cfg.implementation)["tv"][0]
    return dspark_metrics.compute_metrics(
        logits,
        targets,
        None,
        inputs.loss_mask,
        cfg.block_size,
        loss_config,
        tv_loss_fn=tv_loss_fn,
        per_position_loss_weight=cfg.weighting,
        sample_from_anchor=SAMPLE_FROM_ANCHOR["dspark"],
    )


@functools.cache
def cached_loss_config(loss: str, implementation: str) -> LossConfig:
    """``resolve_loss_config``, cached so it stays out of timed regions.

    The result is shared across calls; callers only read it.
    """
    return resolve_loss_config(loss, implementation)
