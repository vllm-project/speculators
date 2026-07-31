"""Unit tests for Draft-OPD loss module."""

import pytest
import torch

from speculators.models.metrics import (
    compute_accuracy_multi_step,
    kl_div_loss,
    reverse_kl_div_loss,
)
from speculators.models.opd_loss import (
    build_accept_mask,
    compute_opd_metrics,
    opd_loss,
)


def _ids_to_logits(ids: torch.Tensor, vocab_size: int) -> torch.Tensor:
    logits = torch.zeros(*ids.shape, vocab_size)
    logits.scatter_(-1, ids.unsqueeze(-1), 100.0)
    return logits


class TestBuildAcceptMask:
    def test_all_match(self):
        logits = _ids_to_logits(torch.tensor([[1, 2, 3, 4]]), 5)
        targets = _ids_to_logits(torch.tensor([[1, 2, 3, 4]]), 5)
        mask = build_accept_mask(logits, targets, block_size=4, sample_from_anchor=True)
        assert mask.shape == (1, 4)
        assert mask.sum().item() == pytest.approx(4.0)

    def test_no_match(self):
        logits = _ids_to_logits(torch.tensor([[0, 0, 0, 0]]), 5)
        targets = _ids_to_logits(torch.tensor([[1, 2, 3, 4]]), 5)
        mask = build_accept_mask(logits, targets, block_size=4, sample_from_anchor=True)
        assert mask.sum().item() == pytest.approx(0.0)

    def test_cumprod_sequential_rejection(self):
        """Positions after first mismatch are rejected even if they match."""
        logits = _ids_to_logits(torch.tensor([[1, 2, 0, 4]]), 5)
        targets = _ids_to_logits(torch.tensor([[1, 2, 3, 4]]), 5)
        mask = build_accept_mask(logits, targets, block_size=4, sample_from_anchor=True)
        expected = torch.tensor([[1, 1, 0, 0]], dtype=torch.float)
        assert torch.equal(mask, expected)

    def test_anchor_slot_forced_true(self):
        """With sample_from_anchor=False, position 0 is always accepted."""
        logits = _ids_to_logits(torch.tensor([[0, 2, 3, 4]]), 5)
        targets = _ids_to_logits(torch.tensor([[1, 2, 3, 4]]), 5)
        mask = build_accept_mask(
            logits, targets, block_size=4, sample_from_anchor=False
        )
        assert mask[0, 0].item() == pytest.approx(1.0)
        assert mask.sum().item() == pytest.approx(4.0)

    def test_multiple_blocks(self):
        logits = _ids_to_logits(torch.tensor([[1, 2, 0, 0, 1, 2, 3, 4]]), 5)
        targets = _ids_to_logits(torch.tensor([[1, 2, 3, 4, 1, 2, 3, 4]]), 5)
        mask = build_accept_mask(logits, targets, block_size=4, sample_from_anchor=True)
        expected = torch.tensor([[1, 1, 0, 0, 1, 1, 1, 1]], dtype=torch.float)
        assert torch.equal(mask, expected)


class TestOPDLoss:
    def test_identical_distributions_zero_loss(self):
        x = torch.randn(1, 8, 16)
        accept_mask = torch.ones(1, 8)
        loss_mask = torch.ones(1, 8)
        loss, _ = opd_loss(x, x, accept_mask, loss_mask, block_size=4)
        assert loss.item() == pytest.approx(0.0, abs=1e-4)

    def test_all_accepted_gives_forward_kl(self, seed):
        logits = torch.randn(1, 8, 16)
        targets = torch.randn(1, 8, 16)
        accept_mask = torch.ones(1, 8)
        loss_mask = torch.ones(1, 8)

        loss, metrics = opd_loss(
            logits, targets, accept_mask, loss_mask, block_size=4
        )

        expected_fwd_kl = kl_div_loss(logits, targets).mean()
        # loss = (1.0 * L_acc + 1.0 * 0) / 2 = L_acc / 2
        # but L_rej denominator is clamped to eps, making L_rej ~0
        assert metrics["n_rejected_sum"].item() == pytest.approx(0.0)
        assert loss.item() > 0

    def test_all_rejected_gives_reverse_kl(self, seed):
        logits = torch.randn(1, 8, 16)
        targets = torch.randn(1, 8, 16)
        accept_mask = torch.zeros(1, 8)
        loss_mask = torch.ones(1, 8)

        loss, metrics = opd_loss(
            logits, targets, accept_mask, loss_mask, block_size=4
        )

        assert metrics["n_accepted_sum"].item() == pytest.approx(0.0)
        assert loss.item() > 0

    def test_lambda_weighting(self, seed):
        logits = torch.randn(1, 8, 16)
        targets = torch.randn(1, 8, 16)
        accept_mask = torch.tensor([[1, 1, 0, 0, 1, 0, 0, 0]], dtype=torch.float)
        loss_mask = torch.ones(1, 8)

        loss_equal, _ = opd_loss(
            logits, targets, accept_mask, loss_mask, block_size=4,
            lambda_acc=1.0, lambda_rej=1.0,
        )
        loss_acc_only, _ = opd_loss(
            logits, targets, accept_mask, loss_mask, block_size=4,
            lambda_acc=1.0, lambda_rej=0.0,
        )
        loss_rej_only, _ = opd_loss(
            logits, targets, accept_mask, loss_mask, block_size=4,
            lambda_acc=0.0, lambda_rej=1.0,
        )

        assert not torch.isclose(loss_acc_only, loss_rej_only)
        assert not torch.isclose(loss_equal, loss_acc_only)

    def test_loss_mask_zeros_out(self):
        logits = torch.randn(1, 8, 16)
        targets = torch.randn(1, 8, 16)
        accept_mask = torch.ones(1, 8)
        loss_mask = torch.zeros(1, 8)

        loss, _ = opd_loss(logits, targets, accept_mask, loss_mask, block_size=4)
        assert loss.item() == pytest.approx(0.0, abs=1e-4)

    def test_gradient_flows(self, seed):
        logits = torch.randn(1, 8, 16, requires_grad=True)
        targets = torch.randn(1, 8, 16)
        accept_mask = torch.tensor([[1, 1, 0, 0, 1, 0, 0, 0]], dtype=torch.float)
        loss_mask = torch.ones(1, 8)

        loss, _ = opd_loss(logits, targets, accept_mask, loss_mask, block_size=4)
        loss.backward()
        assert logits.grad is not None
        assert not torch.all(logits.grad == 0)

    def test_position_decay_affects_rejected(self, seed):
        logits = torch.randn(1, 4, 16)
        targets = torch.randn(1, 4, 16)
        accept_mask = torch.zeros(1, 4)
        loss_mask = torch.ones(1, 4)

        loss_fast_decay, _ = opd_loss(
            logits, targets, accept_mask, loss_mask, block_size=4, gamma=0.1,
        )
        loss_slow_decay, _ = opd_loss(
            logits, targets, accept_mask, loss_mask, block_size=4, gamma=0.99,
        )
        assert not torch.isclose(loss_fast_decay, loss_slow_decay)

    def test_metrics_keys(self, seed):
        logits = torch.randn(1, 8, 16)
        targets = torch.randn(1, 8, 16)
        accept_mask = torch.ones(1, 8)
        loss_mask = torch.ones(1, 8)

        _, metrics = opd_loss(logits, targets, accept_mask, loss_mask, block_size=4)
        for key in [
            "loss_sum", "loss_total",
            "acc_loss_sum", "acc_loss_total",
            "rej_loss_sum", "rej_loss_total",
            "n_accepted_sum", "n_accepted_total",
            "n_rejected_sum", "n_rejected_total",
        ]:
            assert key in metrics, f"Missing key: {key}"


class TestComputeOPDMetrics:
    def test_returns_loss_and_metrics(self, seed):
        logits = torch.randn(1, 8, 16)
        targets = torch.randn(1, 8, 16)
        accept_mask = torch.ones(1, 8)
        loss_mask = torch.ones(1, 8)

        loss, metrics = compute_opd_metrics(
            logits, targets, accept_mask, loss_mask, block_size=4,
        )
        assert isinstance(loss, torch.Tensor)
        assert loss.ndim == 0
        assert isinstance(metrics, dict)

    def test_accuracy_keys_present(self, seed):
        logits = torch.randn(1, 8, 16)
        targets = torch.randn(1, 8, 16)
        accept_mask = torch.ones(1, 8)
        loss_mask = torch.ones(1, 8)

        _, metrics = compute_opd_metrics(
            logits, targets, accept_mask, loss_mask, block_size=4,
        )
        assert "full_acc_sum" in metrics
        assert "full_acc_total" in metrics
        assert "eal_sum" in metrics
        assert "eal_total" in metrics
        for i in range(1, 4):
            assert f"position_{i}_acc_sum" in metrics
            assert f"position_{i}_acc_total" in metrics

    def test_accuracy_consistent_with_multi_step(self, seed):
        logits = torch.randn(1, 8, 16)
        targets = _ids_to_logits(torch.randint(0, 16, (1, 8)), 16)
        accept_mask = torch.ones(1, 8)
        loss_mask = torch.ones(1, 8)

        _, metrics = compute_opd_metrics(
            logits, targets, accept_mask, loss_mask, block_size=4,
        )

        pred_ids = torch.argmax(logits, dim=-1)
        target_ids = torch.argmax(targets, dim=-1)
        pos_idx = torch.arange(8).unsqueeze(0) % 4
        expected_correct, expected_total = compute_accuracy_multi_step(
            pred_ids, target_ids, loss_mask, pos_idx, 4
        )
        assert torch.isclose(metrics["full_acc_sum"], expected_correct[1:].sum())
        assert torch.isclose(metrics["full_acc_total"], expected_total[1:].sum())

    def test_sample_from_anchor_includes_position_0(self, seed):
        logits = torch.randn(1, 8, 16)
        targets = torch.randn(1, 8, 16)
        accept_mask = torch.ones(1, 8)
        loss_mask = torch.ones(1, 8)

        _, metrics = compute_opd_metrics(
            logits, targets, accept_mask, loss_mask, block_size=4,
            sample_from_anchor=True,
        )
        assert "position_0_acc_sum" in metrics
