"""Unit tests for the D-PARD actor and position credit."""

import pytest
import torch
from torch.nn.functional import binary_cross_entropy_with_logits

from speculators.losses import (
    dflash_loss_decay,
    dpard_loss_decay,
    masked_weighted_mean,
    resolve_loss_config,
)
from speculators.losses.eager import renyi_half_loss, tv_loss
from speculators.models.dspark.metrics import compute_metrics


def _literal_renyi_half(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    log_q = torch.log_softmax(logits.float(), dim=-1)
    log_p = torch.log_softmax(targets.float(), dim=-1)
    return -2.0 * torch.logsumexp(0.5 * (log_p + log_q), dim=-1)


def test_renyi_half_matches_literal_value_and_gradient() -> None:
    torch.manual_seed(7)
    logits = torch.randn(1, 5, 11, requires_grad=True)
    targets = torch.randn(1, 5, 11)

    actual = renyi_half_loss(logits, targets)
    expected = _literal_renyi_half(logits, targets)
    actual_grad = torch.autograd.grad(actual.sum(), logits, retain_graph=True)[0]
    expected_grad = torch.autograd.grad(expected.sum(), logits)[0]

    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(actual_grad, expected_grad)


def test_renyi_half_is_zero_at_identity_and_finite_for_extreme_logits() -> None:
    logits = torch.tensor([[[80.0, -80.0, 0.0]]], requires_grad=True)

    identity = renyi_half_loss(logits, logits.detach())
    torch.testing.assert_close(identity, torch.zeros_like(identity), atol=1e-6, rtol=0)
    opposite = renyi_half_loss(logits, -logits.detach())
    gradient = torch.autograd.grad(opposite.sum(), logits)[0]

    assert torch.isfinite(opposite).all()
    assert torch.isfinite(gradient).all()


def test_dpard_credit_matches_suffix_survival_recurrence() -> None:
    acceptance = torch.tensor([[0.8, 0.6, 0.7]], requires_grad=True)
    mask = torch.ones_like(acceptance)

    credit = dpard_loss_decay(acceptance, mask, block_size=3, dpard_alpha=0.5)

    # s=[0.9,0.8,0.85], prefix=[0.9,0.72,0.612],
    # suffix=[2.232,1.332,0.612].
    torch.testing.assert_close(credit, torch.tensor([[2.232, 1.332, 0.612]]))
    assert not credit.requires_grad


def test_dpard_credit_handles_independent_blocks_and_padding() -> None:
    acceptance = torch.tensor([[0.8, 0.6, 0.7, 0.4, 0.9, 0.2]])
    mask = torch.tensor([[1, 1, 0, 1, 1, 1]], dtype=torch.float32)

    credit = dpard_loss_decay(acceptance, mask, block_size=3, dpard_alpha=0.5)
    first = dpard_loss_decay(
        acceptance[:, :3], mask[:, :3], block_size=3, dpard_alpha=0.5
    )
    second = dpard_loss_decay(
        acceptance[:, 3:], mask[:, 3:], block_size=3, dpard_alpha=0.5
    )

    assert credit[0, 2] == 0
    torch.testing.assert_close(credit, torch.cat([first, second], dim=1))


def test_dpard_credit_respects_unsampled_anchor_slot() -> None:
    acceptance = torch.tensor([[0.2, 0.8, 0.6]])
    mask = torch.ones_like(acceptance)

    credit = dpard_loss_decay(
        acceptance, mask, block_size=3, dpard_alpha=0.5, start_pos=1
    )

    # Position zero is excluded. Active s=[0.9,0.8], prefix=[0.9,0.72].
    torch.testing.assert_close(credit, torch.tensor([[0.0, 1.62, 0.72]]))


def test_weighted_mean_divides_by_valid_position_count() -> None:
    actor = torch.tensor([[1.0, 2.0, 4.0, 8.0], [3.0, 9.0, 9.0, 9.0]])
    credit = torch.tensor([[2.0, 1.0, 3.0, 9.0], [2.0, 7.0, 7.0, 7.0]])
    mask = torch.tensor([[1.0, 1.0, 1.0, 0.0], [1.0, 0.0, 0.0, 0.0]])

    actual = masked_weighted_mean(actor, credit, mask)
    first_sequence = (2.0 * 1.0 + 1.0 * 2.0 + 3.0 * 4.0) / (3.0 + 1e-5)
    second_sequence = 2.0 * 3.0 / (1.0 + 1e-5)
    expected = torch.tensor((first_sequence + second_sequence) / 2.0)

    torch.testing.assert_close(actual, expected)
    assert actual != (actor * credit * mask).sum() / (credit * mask).sum()


def test_dpard_credit_rejects_invalid_shape_and_alpha() -> None:
    acceptance = torch.ones(1, 4)
    mask = torch.ones(1, 4)

    with pytest.raises(ValueError, match="divisible by block_size"):
        dpard_loss_decay(acceptance, mask, block_size=3, dpard_alpha=0.5)
    with pytest.raises(ValueError, match="strictly between"):
        dpard_loss_decay(acceptance, mask, block_size=4, dpard_alpha=1.0)


def test_dpard_metrics_match_literal_valid_position_actor() -> None:
    torch.manual_seed(11)
    logits = torch.randn(1, 6, 7)
    targets = torch.randn(1, 6, 7)
    mask = torch.tensor([[1, 1, 0, 1, 1, 1]], dtype=torch.float32)
    loss_config = resolve_loss_config("renyi_half", "eager")

    loss, metrics = compute_metrics(
        logits,
        targets,
        None,
        mask,
        block_size=3,
        loss_config=loss_config,
        tv_loss_fn=tv_loss,
        per_position_loss_weight="dpard",
        dpard_alpha=0.5,
    )

    actor = renyi_half_loss(logits, targets)
    acceptance = 1.0 - tv_loss(logits, targets)
    credit = dpard_loss_decay(acceptance, mask, 3, 0.5)
    expected = (actor * credit * mask).sum() / mask.sum()
    torch.testing.assert_close(loss, expected)
    torch.testing.assert_close(metrics["dpard_credit_sum"], credit.sum())
    torch.testing.assert_close(metrics["dpard_credit_total"], mask.sum())
    credit_blocks = credit.view(-1, 3)
    mask_blocks = mask.view(-1, 3)
    for pos in range(3):
        torch.testing.assert_close(
            metrics[f"position_{pos}_dpard_credit_sum"],
            (credit_blocks[:, pos] * mask_blocks[:, pos]).sum(),
        )
        torch.testing.assert_close(
            metrics[f"position_{pos}_dpard_credit_total"], mask_blocks[:, pos].sum()
        )


def test_dpard_keeps_confidence_on_fixed_exponential_weights() -> None:
    torch.manual_seed(13)
    logits = torch.randn(1, 6, 7)
    targets = torch.randn(1, 6, 7)
    mask = torch.tensor([[1, 1, 0, 1, 1, 1]], dtype=torch.float32)
    confidence_logits = torch.zeros(1, 6)
    loss_config = resolve_loss_config("renyi_half", "eager")

    actor_loss, _ = compute_metrics(
        logits,
        targets,
        None,
        mask,
        block_size=3,
        loss_config=loss_config,
        tv_loss_fn=tv_loss,
        per_position_loss_weight="dpard",
        dpard_alpha=0.5,
    )
    total_loss, _ = compute_metrics(
        logits,
        targets,
        confidence_logits,
        mask,
        block_size=3,
        loss_config=loss_config,
        tv_loss_fn=tv_loss,
        gamma=4.0,
        confidence_head_alpha=1.0,
        per_position_loss_weight="dpard",
        dpard_alpha=0.5,
    )

    acceptance = (1.0 - tv_loss(logits, targets)).detach()
    bce = binary_cross_entropy_with_logits(
        confidence_logits, acceptance, reduction="none"
    )
    pos_idx = torch.arange(6).remainder(3).unsqueeze(0)
    fixed_weight = dflash_loss_decay(
        pos_idx.float(), gamma=4.0, sample_from_anchor=True
    )
    expected_confidence = (bce * mask * fixed_weight).sum() / mask.sum()
    torch.testing.assert_close(total_loss - actor_loss, expected_confidence)


def test_b16_actor_gradient_has_no_acceptance_credit_gradient() -> None:
    """Check the B16 gradient against the analytic expression."""
    torch.manual_seed(42)
    logits = torch.randn(1, 32, 7, requires_grad=True)
    targets = torch.randn_like(logits)
    mask = torch.tensor([[1.0] * 16 + [1.0] * 9 + [0.0] * 7])
    loss, _ = compute_metrics(
        logits,
        targets,
        None,
        mask,
        block_size=16,
        loss_config=resolve_loss_config("renyi_half", "eager"),
        tv_loss_fn=tv_loss,
        per_position_loss_weight="dpard",
        dpard_alpha=0.5,
    )
    with torch.no_grad():
        p, q = targets.softmax(-1), logits.softmax(-1)
        acceptance = torch.minimum(p, q).sum(-1)
        weight = torch.zeros_like(mask)
        for start, length in [(0, 16), (16, 9)]:
            prefix = torch.ones((), device=acceptance.device)
            for end in range(start, start + length):
                prefix = prefix * (0.5 + 0.5 * acceptance[0, end])
                weight[0, start : end + 1] += prefix
        tilted = (p * q).sqrt()
        tilted = tilted / tilted.sum(-1, keepdim=True)
        expected_grad = (q - tilted) * weight.unsqueeze(-1) / 25.0
    actual_grad = torch.autograd.grad(loss, logits)[0]
    torch.testing.assert_close(actual_grad, expected_grad)
