import json
from collections.abc import Callable

import torch

_EPS = 1e-5

LossConfig = dict[
    str, tuple[Callable[[torch.Tensor, torch.Tensor], torch.Tensor], float]
]


def compute_accuracy_single_step(
    pred_ids: torch.Tensor,
    target_ids: torch.Tensor,
    loss_mask: torch.Tensor | None,
    prev_correct: torch.Tensor | None,
):
    correct = pred_ids == target_ids
    cond_total = torch.tensor(correct.numel(), dtype=torch.float, device=correct.device)
    if prev_correct is not None:
        cond_total = prev_correct.sum().float()
        correct = torch.logical_and(prev_correct, correct, out=prev_correct)
    if loss_mask is not None:
        correct = torch.masked_select(correct, loss_mask.to(torch.bool))

    correct_sum = correct.float().sum()
    full_total = torch.tensor(correct.numel(), dtype=torch.float, device=correct.device)

    return correct_sum, full_total, correct_sum, cond_total


@torch.no_grad()
def compute_accuracy_multi_step(
    pred_ids: torch.Tensor,
    target_ids: torch.Tensor,
    loss_mask: torch.Tensor,
    pos_idx: torch.Tensor,
    num_pos: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    correct = pred_ids == target_ids
    correct = torch.masked_select(correct, loss_mask.to(torch.bool))
    pos_idx = torch.masked_select(pos_idx, loss_mask.to(torch.bool))

    correct_per_pos = torch.zeros(num_pos, dtype=torch.float, device=correct.device)
    total_per_pos = torch.zeros(num_pos, dtype=torch.float, device=correct.device)
    correct_per_pos.scatter_add_(0, pos_idx, correct.float())
    total_per_pos.scatter_add_(0, pos_idx, torch.ones_like(correct, dtype=torch.float))

    return correct_per_pos, total_per_pos


def kl_div_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
):
    logits = torch.nn.functional.log_softmax(logits, dim=-1, dtype=torch.float32)
    target_p = torch.nn.functional.softmax(targets, dim=-1, dtype=torch.float32)
    elementwise_loss = torch.nn.functional.kl_div(
        logits, target_p, reduction="none", log_target=False
    ).sum(dim=-1)

    return elementwise_loss  # noqa: RET504


def ce_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
):
    batch_size, seq_len, draft_vocab_size = logits.shape
    target_ids = torch.argmax(targets, dim=-1)

    elementwise_loss = torch.nn.functional.cross_entropy(
        logits.reshape(-1, draft_vocab_size),
        target_ids.reshape(-1),
        reduction="none",
        ignore_index=-100,
    ).reshape(batch_size, seq_len)

    return elementwise_loss  # noqa: RET504


def tv_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
):
    draft_p = torch.nn.functional.softmax(logits, dim=-1, dtype=torch.float32)
    target_p = torch.nn.functional.softmax(targets, dim=-1, dtype=torch.float32)
    overlap = torch.minimum(draft_p, target_p).sum(dim=-1)
    elementwise_loss = 1.0 - overlap

    return elementwise_loss  # noqa: RET504


def dflash_loss_decay(pos_idx: torch.Tensor, gamma: float):
    decay_mult = torch.exp(-((pos_idx - 1).clamp(min=0)) / gamma)
    decay_mult = decay_mult * (pos_idx != 0).to(decay_mult.dtype)
    return decay_mult  # noqa: RET504


def exp_loss_decay(pos_idx: torch.Tensor, gamma: float):
    return gamma**pos_idx


def dpace_loss_weight(
    neg_log_q: torch.Tensor,
    loss_mask: torch.Tensor,
    block_size: int,
    alpha: float = 0.5,
):
    with torch.no_grad():
        q = torch.exp(-neg_log_q).float()

        num_anchors = q.shape[1] // block_size
        q = q.reshape(num_anchors, block_size)
        mask = loss_mask.reshape(num_anchors, block_size).to(q.dtype)

        smooth = (1.0 - alpha) * q + alpha
        smooth = torch.where(mask > 0, smooth, torch.ones_like(smooth))

        prefix = torch.cumprod(smooth, dim=-1)

        weight = torch.flip(
            torch.cumsum(torch.flip(prefix * mask, dims=[-1]), dim=-1), dims=[-1]
        )
        weight = weight * mask

    return weight.reshape(1, -1)


def loss_function(
    logits: torch.Tensor,
    targets: torch.Tensor,
    loss_mask: torch.Tensor,
    pos_idx: torch.Tensor,
    loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = kl_div_loss,
    decay_fn: Callable[[torch.Tensor], torch.Tensor] | None = None,
    decay_mult: torch.Tensor | None = None,
    elementwise_loss: torch.Tensor | None = None,
):
    if elementwise_loss is None:
        elementwise_loss = loss_fn(logits, targets)

    loss_mask = loss_mask.to(elementwise_loss.dtype)
    elementwise_loss = elementwise_loss * loss_mask

    if decay_mult is not None:
        elementwise_loss = elementwise_loss * decay_mult.to(elementwise_loss.dtype)
    elif decay_fn is not None:
        decay_mult = decay_fn(pos_idx.to(elementwise_loss.dtype))
        elementwise_loss = elementwise_loss * decay_mult

    denominator = loss_mask.sum(dim=1) + _EPS

    batch_loss = torch.sum(elementwise_loss, dim=1) / denominator
    return batch_loss.mean()


_LOSS_FN_MAP: dict[str, Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = {
    "kl_div": kl_div_loss,
    "ce": ce_loss,
    "tv": tv_loss,
}


def resolve_loss_config(spec: str) -> LossConfig:
    if spec in _LOSS_FN_MAP:
        return {spec: (_LOSS_FN_MAP[spec], 1.0)}

    try:
        parsed = json.loads(spec)
    except json.JSONDecodeError:
        raise ValueError(
            f"Unknown loss function '{spec}'. Pass a known name "
            f"({sorted(_LOSS_FN_MAP.keys())}) or a JSON dict, "
            f'e.g. \'{{"ce": 0.1, "tv": 0.9}}\'.'
        ) from None

    if not isinstance(parsed, dict) or not parsed:
        raise ValueError(
            "Loss config must be a non-empty JSON dict mapping loss names to weights, "
            f'e.g. \'{{"ce": 0.1, "tv": 0.9}}\'. Got: {spec}'
        )

    config: LossConfig = {}
    for name, weight in parsed.items():
        if name not in _LOSS_FN_MAP:
            raise ValueError(
                f"Unknown loss function '{name}' in loss config. "
                f"Choose from: {sorted(_LOSS_FN_MAP.keys())}"
            )
        if isinstance(weight, bool) or not isinstance(weight, (int, float)):
            raise ValueError(
                f"Loss weight for '{name}' must be a number, "
                f"got {type(weight).__name__}"
            )
        config[name] = (_LOSS_FN_MAP[name], float(weight))

    return config
