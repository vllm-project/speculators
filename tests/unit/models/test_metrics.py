"""Tests for shared metrics primitives (speculators.models.metrics)."""

from functools import partial

import pytest
import torch

from speculators.models.metrics import (
    compute_accuracy_single_step,
    dflash_loss_decay,
    exp_loss_decay,
    kl_div_loss,
    loss_function,
)


class TestLossFunction:
    def test_decay_scales_loss(self):
        """Decay at different positions must scale the loss, not cancel out.

        This is the critical invariant for eagle3: exp_loss_decay applied
        at ttt_step=2 with gamma=0.5 should produce loss * 0.25 compared
        to ttt_step=0 (where decay=1.0).
        """
        B, T, V = 1, 16, 8
        torch.manual_seed(42)
        logits = torch.randn(B, T, V)
        targets = torch.randn(B, T, V)
        loss_mask = torch.ones(B, T)
        gamma = 0.5

        pos_idx_step0 = torch.zeros(B, T, dtype=torch.long)
        pos_idx_step2 = torch.full((B, T), 2, dtype=torch.long)

        loss_step0 = loss_function(
            logits,
            targets,
            loss_mask,
            pos_idx_step0,
            loss_fn=kl_div_loss,
            decay_fn=partial(exp_loss_decay, gamma=gamma),
        )
        loss_step2 = loss_function(
            logits,
            targets,
            loss_mask,
            pos_idx_step2,
            loss_fn=kl_div_loss,
            decay_fn=partial(exp_loss_decay, gamma=gamma),
        )

        assert torch.isclose(loss_step2, loss_step0 * gamma**2, rtol=1e-4)

    def test_loss_mask_zeros_out(self):
        """All-zero loss_mask must produce zero loss."""
        logits = torch.randn(1, 8, 5)
        targets = torch.randn(1, 8, 5)
        loss_mask = torch.zeros(1, 8)
        pos_idx = torch.zeros(1, 8, dtype=torch.long)

        loss = loss_function(logits, targets, loss_mask, pos_idx)
        assert loss.item() == pytest.approx(0.0, abs=1e-4)

    def test_no_decay_equals_masked_mean(self):
        """Without decay_fn, loss_function should equal masked mean of loss_fn."""
        torch.manual_seed(7)
        logits = torch.randn(1, 6, 4)
        targets = torch.randn(1, 6, 4)
        loss_mask = torch.tensor([[1, 0, 1, 1, 0, 1]], dtype=torch.float)
        pos_idx = torch.zeros(1, 6, dtype=torch.long)

        result = loss_function(logits, targets, loss_mask, pos_idx, loss_fn=kl_div_loss)

        elementwise = kl_div_loss(logits, targets)
        expected = (elementwise * loss_mask).sum() / (loss_mask.sum() + 1e-5)
        assert torch.isclose(result, expected)


class TestKLDivLoss:
    def test_identical_zero_and_random_nonnegative(self):
        """KL divergence of identical distributions is ~0; random inputs are >= 0."""
        x = torch.randn(1, 4, 8)
        loss_identical = kl_div_loss(x, x)
        assert loss_identical.sum().item() == pytest.approx(0.0, abs=1e-4)

        torch.manual_seed(0)
        loss_random = kl_div_loss(torch.randn(1, 4, 8), torch.randn(1, 4, 8))
        assert (loss_random >= -1e-6).all()


class TestComputeAccuracySingleStep:
    def test_prev_correct_chain(self):
        """Conditional accuracy across ttt steps tracks cumulative correctness.

        Step 0: positions [0,1,2] correct, [3] wrong → full_acc=3/4
        Step 1: positions [0,1,3] correct, [2] wrong → only [0,1] still correct
        Conditional acc = 2 / (3 prev_correct) = 2/3
        prev_correct should be mutated to [T, T, F, F].
        """
        pred_step0 = torch.tensor([[1, 2, 3, 0]])
        tgt_step0 = torch.tensor([[1, 2, 3, 4]])
        prev_correct = torch.ones(1, 4, dtype=torch.bool)

        full_acc_0, cond_acc_0 = compute_accuracy_single_step(
            pred_step0,
            tgt_step0,
            loss_mask=None,
            prev_correct=prev_correct,
        )
        assert full_acc_0 == pytest.approx(3 / 4, abs=1e-4)
        assert cond_acc_0 == pytest.approx(3 / 4, abs=1e-4)
        assert prev_correct.tolist() == [[True, True, True, False]]

        pred_step1 = torch.tensor([[1, 2, 0, 4]])
        tgt_step1 = torch.tensor([[1, 2, 5, 4]])

        full_acc_1, cond_acc_1 = compute_accuracy_single_step(
            pred_step1,
            tgt_step1,
            loss_mask=None,
            prev_correct=prev_correct,
        )
        # [0,1] correct on both steps, [2] was correct but now wrong,
        # [3] was already wrong
        assert prev_correct.tolist() == [[True, True, False, False]]
        assert full_acc_1 == pytest.approx(2 / 4, abs=1e-4)
        assert cond_acc_1 == pytest.approx(2 / 3, abs=1e-4)


class TestDecayFunctions:
    def test_dflash_anchor_zero_and_exp_values(self):
        """dflash pos=0 has zero weight; exp_loss_decay matches gamma^pos_idx."""
        pos = torch.arange(4, dtype=torch.float)
        dflash = dflash_loss_decay(pos, gamma=4.0)
        assert dflash[0].item() == 0.0
        assert dflash[1].item() == pytest.approx(1.0)
        assert dflash[2].item() < dflash[1].item()

        assert exp_loss_decay(torch.tensor(2.0), gamma=0.5).item() == pytest.approx(
            0.25
        )
        assert exp_loss_decay(torch.tensor(0.0), gamma=0.5).item() == pytest.approx(1.0)
