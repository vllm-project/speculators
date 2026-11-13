"""Shared loss selection, weighting, and reduction utilities."""

import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import torch

from speculators.losses import eager

_LOSS_REDUCTION_EPS = 1e-5

LossConfig = dict[
    str, tuple[Callable[[torch.Tensor, torch.Tensor], torch.Tensor], float]
]


def dflash_loss_decay(
    pos_idx: torch.Tensor, gamma: float, sample_from_anchor: bool = False, **_kwargs
):
    """Compute DFlash-style exponential decay weights per position.

    When ``sample_from_anchor`` is False (DFlash), position 0 is the anchor and
    gets weight 0; position 1 gets weight 1, and subsequent positions decay as
    ``exp(-(pos - 1) / gamma)``.

    When ``sample_from_anchor`` is True (DSpark), position 0 is the first real
    prediction and gets weight 1; subsequent positions decay as
    ``exp(-pos / gamma)``.

    Args:
        pos_idx: Position indices within each speculative block.
        gamma: Decay rate (higher = slower decay).
        sample_from_anchor: Whether position 0 is a prediction (DSpark) or the
            anchor input (DFlash).

    Returns:
        Decay multiplier tensor with same shape as pos_idx.
    """
    if sample_from_anchor:
        # DSpark: slot 0 predicts the token after the anchor, so decay from 0.
        return torch.exp(-pos_idx / gamma)
    # DFlash: slot 0 is the anchor (no prediction); decay from slot 1.
    # pos_idx = 0 1 2 3 0 1 2 3, block_size = 4
    decay_mult = torch.exp(-((pos_idx - 1).clamp(min=0)) / gamma)
    # decay_mult = e^-(0 0 1 2 0 0 1 2) / gamma
    decay_mult = decay_mult * (pos_idx != 0).to(decay_mult.dtype)
    # w = 0 1 e^-1/gamma e^-2/gamma 0 1 e^-1/gamma e^-2/gamma
    return decay_mult  # noqa: RET504


def exp_loss_decay(pos_idx: torch.Tensor, gamma: float, **_kwargs):
    """Compute simple exponential decay weights as gamma^pos_idx.

    Args:
        pos_idx: Position indices within each speculative block.
        gamma: Base of the exponent (typically in (0, 1]).

    Returns:
        Decay multiplier tensor with same shape as pos_idx.
    """
    return gamma**pos_idx


def dpace_loss_decay(
    pos_idx: torch.Tensor,  # noqa: ARG001
    loss_mask: torch.Tensor,
    block_size: int,
    dpace_alpha: float,
    elementwise_loss: torch.Tensor,
    **_kwargs,
):
    """
    Per-position block-drafting loss weight based on D-PACE

    Args:
        elementwise_loss: requires to be cross-entropy loss, negative log-likelihood
            of per-position confidence
        dpace_alpha: confidence smoothing constant

    Returns:
        Decay multiplier tensor with same shape as pos_idx.
    """
    with torch.no_grad():
        # convert CE to per-position confidence
        q = torch.exp(-elementwise_loss).float()

        # reshape loss to [num_anchors, block_size]
        # for intra-block cumulative multiplication
        if q.shape[1] % block_size != 0:
            raise ValueError(
                f"q.shape[1] ({q.shape[1]}) must be divisible by "
                f"block_size ({block_size})"
            )
        num_anchors = q.shape[1] // block_size
        q = q.reshape(num_anchors, block_size)
        mask = loss_mask.reshape(num_anchors, block_size).to(q.dtype)

        # smoothed confidence for numerical stability
        smooth = (1.0 - dpace_alpha) * q + dpace_alpha
        smooth = torch.where(mask > 0, smooth, torch.ones_like(smooth))

        # prefix cumulative production
        prefix = torch.cumprod(smooth, dim=-1)

        # suffix summation: flip -> cumsum -> flip
        weight = torch.flip(
            torch.cumsum(torch.flip(prefix * mask, dims=[-1]), dim=-1), dims=[-1]
        )
        weight = weight * mask

    # reshape weight
    return weight.reshape(1, -1)


def _fused_kernel(name: str):
    """Import a fused loss by name."""
    try:
        from speculators.losses import fused as mod  # noqa: PLC0415
    except ImportError as error:
        raise RuntimeError(
            "Fused losses require Triton; use --loss-implementation eager for "
            "compatibility testing."
        ) from error
    return getattr(mod, name)


def kl_div_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    return _fused_kernel("fused_kl_div_loss")(logits, targets)


def reverse_kl_div_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    return _fused_kernel("fused_reverse_kl_div_loss")(logits, targets)


def js_div_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    return _fused_kernel("fused_js_div_loss")(logits, targets)


def ce_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    return _fused_kernel("fused_ce_loss")(logits, targets)


def tv_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    return _fused_kernel("fused_tv_loss")(logits, targets)


def neg_log_acceptance_loss(
    logits: torch.Tensor, targets: torch.Tensor
) -> torch.Tensor:
    return _fused_kernel("fused_nla_loss")(logits, targets)


def lk_hybrid_loss(
    logits: torch.Tensor, targets: torch.Tensor, eta: float = 3.0
) -> torch.Tensor:
    return _fused_kernel("fused_lk_hybrid_loss")(logits, targets, eta=eta)


_FUSED_LOSS_FN_MAP: dict[str, Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = {
    "kl_div": kl_div_loss,
    "rkl": reverse_kl_div_loss,
    "jsd": js_div_loss,
    "ce": ce_loss,
    "tv": tv_loss,
    "nla": neg_log_acceptance_loss,
    "lk_hybrid": lk_hybrid_loss,
}


# Fused map functions back to their names: compound_loss routes a config whose
# terms are all fused through a single kernel call.
_FUSED_NAME_BY_FN = {fn: name for name, fn in _FUSED_LOSS_FN_MAP.items()}


_EAGER_LOSS_FN_MAP: dict[str, Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = {
    "kl_div": eager.kl_div_loss,
    "rkl": eager.reverse_kl_div_loss,
    "jsd": eager.js_div_loss,
    "ce": eager.ce_loss,
    "tv": eager.tv_loss,
    "nla": eager.neg_log_acceptance_loss,
    "lk_hybrid": eager.lk_hybrid_loss,
}


_LOSS_FN_MAPS = {"fused": _FUSED_LOSS_FN_MAP, "eager": _EAGER_LOSS_FN_MAP}


def resolve_loss_config(spec: str, implementation: str = "fused") -> LossConfig:
    """Parse a loss spec into ``{name: (loss_fn, weight)}``.

    Accepts either a plain loss name (``"kl_div"``) or a JSON dict mapping
    loss names to weights (``'{"ce": 0.1, "tv": 0.9}'``).
    """
    if implementation == "fused_linear":
        # Models that support it (DFlash) resolve their config with "fused"
        # and take the hidden-state path themselves.
        raise ValueError(
            "loss_implementation='fused_linear' is only supported by DFlash; "
            "use 'fused' or 'eager' for this model"
        )
    try:
        loss_fn_map = _LOSS_FN_MAPS[implementation]
    except KeyError:
        raise ValueError(
            f"Unknown loss implementation '{implementation}'. Choose from: "
            f"{sorted(_LOSS_FN_MAPS)}"
        ) from None

    if spec in loss_fn_map:
        return {spec: (loss_fn_map[spec], 1.0)}

    try:
        parsed = json.loads(spec)
    except json.JSONDecodeError:
        raise ValueError(
            f"Unknown loss function '{spec}'. Pass a known name "
            f"({sorted(loss_fn_map)}) or a JSON dict, "
            f'e.g. \'{{"ce": 0.1, "tv": 0.9}}\'.'
        ) from None

    if not isinstance(parsed, dict) or not parsed:
        raise ValueError(
            "Loss config must be a non-empty JSON dict mapping loss names to weights, "
            f'e.g. \'{{"ce": 0.1, "tv": 0.9}}\'. Got: {spec}'
        )

    config: LossConfig = {}
    for name, weight in parsed.items():
        if name not in loss_fn_map:
            raise ValueError(
                f"Unknown loss function '{name}' in loss config. "
                f"Choose from: {sorted(loss_fn_map)}"
            )
        if not isinstance(weight, (int, float)):
            raise ValueError(
                f"Loss weight for '{name}' must be a number, "
                f"got {type(weight).__name__}"
            )
        config[name] = (loss_fn_map[name], float(weight))

    return config


def compound_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    loss_mask: torch.Tensor,
    pos_idx: torch.Tensor,
    loss_config: LossConfig,
    decay_fn: Callable[..., torch.Tensor] | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute a weighted sum of loss terms.

    Each entry in *loss_config* maps a name to ``(loss_fn, weight)``; the
    result is ``sum(weight * loss_function(logits, targets, ..., loss_fn))``
    over all entries.

    When every term is a fused loss, all terms come from one kernel call
    (``fused_compound_losses``): the draft/target softmax statistics are
    computed once and backward writes a single combined logits gradient.
    A config that mixes fused and eager terms runs each term on its own.

    Returns the total loss and a dict of per-term (unweighted) scalar losses
    keyed as ``"{name}_loss"``.  When the config contains a single term the
    dict is empty (the overall loss already captures it).
    """
    out = _compound_loss(
        logits,
        targets,
        loss_mask,
        pos_idx,
        loss_config,
        decay_fn,
        metric_terms=(),
        with_ids=False,
    )
    return out.loss, out.term_losses


@dataclass
class CompoundLossOutput:
    """``compound_loss`` results plus what the same logits pass can provide."""

    loss: torch.Tensor
    """Weighted total loss (scalar)."""
    term_losses: dict[str, torch.Tensor]
    """Per-term scalar losses ``"{name}_loss"`` (empty for a single term)."""
    pred_ids: torch.Tensor
    """``[1, T]`` ``argmax(logits, -1)``, first index on ties."""
    target_ids: torch.Tensor
    """``[1, T]`` ``argmax(targets, -1)``, first index on ties."""
    metric_terms: dict[str, torch.Tensor]
    """Requested per-position ``[1, T]`` primitive terms, detached. Only
    present when they came from the fused pass; callers compute them
    otherwise."""


def compound_loss_with_outputs(
    logits: torch.Tensor,
    targets: torch.Tensor,
    loss_mask: torch.Tensor,
    pos_idx: torch.Tensor,
    loss_config: LossConfig,
    decay_fn: Callable[..., torch.Tensor] | None = None,
    metric_terms: Sequence[str] = (),
) -> CompoundLossOutput:
    """``compound_loss`` plus both argmaxes and metric-only terms.

    With an all-fused config the argmaxes and ``metric_terms`` (primitive
    terms such as ``"tv"``) come from the same kernel pass as the loss, so no
    extra read of the ``[1, T, V]`` tensors happens. Otherwise the argmaxes
    fall back to ``torch.argmax`` and ``metric_terms`` is left empty.
    """
    return _compound_loss(
        logits,
        targets,
        loss_mask,
        pos_idx,
        loss_config,
        decay_fn,
        metric_terms=metric_terms,
        with_ids=True,
    )


def _compound_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    loss_mask: torch.Tensor,
    pos_idx: torch.Tensor,
    loss_config: LossConfig,
    decay_fn: Callable[..., torch.Tensor] | None,
    metric_terms: Sequence[str],
    with_ids: bool,
) -> CompoundLossOutput:
    fused_names = [_FUSED_NAME_BY_FN.get(fn) for fn, _ in loss_config.values()]
    metrics: dict[str, torch.Tensor] = {}
    if fused_names and all(name is not None for name in fused_names):
        fused = _fused_kernel("fused_compound")(
            logits, targets, fused_names, metric_terms
        )
        elementwise = [fused.losses[name] for name in fused_names]
        pred_ids, target_ids = fused.pred_ids, fused.target_ids
        metrics = fused.metrics
    else:
        elementwise = [fn(logits, targets) for fn, _ in loss_config.values()]
        if with_ids:
            pred_ids = torch.argmax(logits, dim=-1)
            target_ids = torch.argmax(targets, dim=-1)
        else:
            pred_ids = target_ids = torch.empty(0, dtype=torch.long)
    total, term_losses = _weighted_sum(
        loss_config, elementwise, loss_mask, pos_idx, decay_fn
    )
    return CompoundLossOutput(total, term_losses, pred_ids, target_ids, metrics)


def compound_loss_from_hidden(
    hidden: torch.Tensor,
    weight: torch.Tensor,
    targets: torch.Tensor,
    loss_mask: torch.Tensor,
    pos_idx: torch.Tensor,
    loss_config: LossConfig,
    decay_fn: Callable[..., torch.Tensor] | None = None,
    metric_terms: Sequence[str] = (),
    chunk_rows: int | None = None,
    target_weight: torch.Tensor | None = None,
    grad_mode: str = "auto",
    logits_dtype: str = "input",
    rows: torch.Tensor | None = None,
) -> CompoundLossOutput:
    """``compound_loss_with_outputs`` from hidden states and the ``lm_head``
    weight, without materializing the ``[1, T, V]`` draft logits.

    The logits are produced chunk by chunk inside ``fused_linear_compound``;
    every term of ``loss_config`` must be a fused loss. With
    ``target_weight`` (the verifier ``lm_head``), ``targets`` holds the
    post-norm verifier hidden states and the target logits are recomputed
    per chunk too. ``grad_mode``, ``logits_dtype`` and ``rows`` are passed to
    ``fused_linear_compound``.
    """
    names: list[str] = []
    for fn, _ in loss_config.values():
        name = _FUSED_NAME_BY_FN.get(fn)
        if name is None:
            raise ValueError(
                "compound_loss_from_hidden needs a loss config resolved with "
                "the 'fused' implementation"
            )
        names.append(name)
    try:
        from speculators.losses import fused_linear  # noqa: PLC0415
    except ImportError as error:
        raise RuntimeError("Fused linear losses require Triton.") from error
    out = fused_linear.fused_linear_compound(
        hidden,
        weight,
        targets,
        names,
        metric_terms,
        chunk_rows=chunk_rows,
        target_weight=target_weight,
        grad_mode=grad_mode,
        logits_dtype=logits_dtype,
        rows=rows,
    )
    elementwise = [out.losses[name] for name in names]
    total, term_losses = _weighted_sum(
        loss_config, elementwise, loss_mask, pos_idx, decay_fn
    )
    return CompoundLossOutput(
        total, term_losses, out.pred_ids, out.target_ids, out.metrics
    )


def _weighted_sum(loss_config, elementwise, loss_mask, pos_idx, decay_fn):
    """Reduce each term's per-position loss and sum them with their weights."""
    total = torch.zeros((), device=loss_mask.device, dtype=torch.float32)
    term_losses: dict[str, torch.Tensor] = {}
    multi = len(loss_config) > 1
    for (name, (_, weight)), term_elementwise in zip(
        loss_config.items(), elementwise, strict=True
    ):
        term = _reduce_elementwise_loss(term_elementwise, loss_mask, pos_idx, decay_fn)
        if multi:
            term_losses[f"{name}_loss"] = term.detach()
        total = total + weight * term
    return total, term_losses


def loss_function(
    logits: torch.Tensor,  # shape: [1, seq_len, draft_vocab_size]
    targets: torch.Tensor,  # shape: [1, seq_len, draft_vocab_size]
    loss_mask: torch.Tensor,  # shape: [1, seq_len]
    pos_idx: torch.Tensor,  # shape: [1, seq_len]
    loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = kl_div_loss,
    decay_fn: Callable[..., torch.Tensor] | None = None,
):
    """Compute masked, optionally position-decayed training loss.

    Args:
        logits: Draft model logits.
        targets: Target model logits.
        loss_mask: Boolean mask selecting positions to include in the loss.
        pos_idx: Position indices within each speculative block.
        loss_fn: Per-position loss function (default: kl_div_loss).
        decay_fn: Optional position-dependent decay weighting function.

    Returns:
        Scalar mean loss across the batch.
    """
    return _reduce_elementwise_loss(
        loss_fn(logits, targets), loss_mask, pos_idx, decay_fn
    )


def _reduce_elementwise_loss(
    elementwise_loss: torch.Tensor,  # shape: [1, seq_len]
    loss_mask: torch.Tensor,  # shape: [1, seq_len]
    pos_idx: torch.Tensor,  # shape: [1, seq_len]
    decay_fn: Callable[..., torch.Tensor] | None,
) -> torch.Tensor:
    """Mask, optionally decay, and average a per-position loss (see
    ``loss_function``).
    """
    loss_mask = loss_mask.to(elementwise_loss.dtype)
    elementwise_loss = elementwise_loss * loss_mask

    if decay_fn is not None:
        decay_mult = decay_fn(
            pos_idx.to(elementwise_loss.dtype), elementwise_loss=elementwise_loss
        )
        elementwise_loss = elementwise_loss * decay_mult

    denominator = loss_mask.sum(dim=1) + _LOSS_REDUCTION_EPS

    batch_loss = torch.sum(elementwise_loss, dim=1) / denominator  # shape: [1]
    return batch_loss.mean()  # shape: []
