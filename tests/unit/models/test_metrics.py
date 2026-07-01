"""Tests for shared metrics primitives (speculators.models.metrics)."""

from functools import partial

import pytest
import torch

from speculators.models.metrics import (
    ce_loss,
    compute_accuracy_single_step,
    dflash_loss_decay,
    exp_loss_decay,
    kl_div_loss,
    lk_hybrid_loss,
    loss_function,
    neg_log_acceptance_loss,
    resolve_loss_fn,
    tv_loss,
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


class TestTVLoss:
    def test_identical_is_zero(self):
        """TV distance between identical distributions is ~0."""
        x = torch.randn(1, 4, 50)
        loss = tv_loss(x, x)
        assert torch.allclose(loss, torch.zeros(1, 4), atol=1e-6)

    def test_matches_l1_form_shape_and_range(self):
        """Overlap form equals 0.5 * L1; output is [1, seq_len] within [0, 1]."""
        torch.manual_seed(0)
        logits = torch.randn(1, 4, 50)
        targets = torch.randn(1, 4, 50)
        out = tv_loss(logits, targets)

        p = torch.softmax(targets, dim=-1)
        q = torch.softmax(logits, dim=-1)
        l1 = 0.5 * (p - q).abs().sum(dim=-1)

        assert out.shape == (1, 4)
        assert torch.allclose(out, l1, atol=1e-6)
        assert (out >= 0).all()
        assert (out <= 1).all()

    def test_resolve_tv(self):
        """resolve_loss_fn maps 'tv' to tv_loss."""
        assert resolve_loss_fn("tv") is tv_loss


class TestNegLogAcceptanceLoss:
    def test_identical_is_zero(self):
        """Negative log-acceptance of identical distributions is ~0."""
        logits = torch.randn(1, 4, 50)
        loss = neg_log_acceptance_loss(logits, logits)
        assert torch.allclose(loss, torch.zeros(1, 4), atol=1e-5)

    def test_equals_neg_log_overlap_and_shape(self):
        """Loss equals -log(overlap); output is [1, seq_len] and non-negative."""
        torch.manual_seed(0)
        logits = torch.randn(1, 4, 50)
        targets = torch.randn(1, 4, 50)
        out = neg_log_acceptance_loss(logits, targets)

        overlap = 1.0 - tv_loss(logits, targets)  # tv = 1 - alpha
        assert out.shape == (1, 4)
        assert torch.allclose(out, -torch.log(overlap.clamp_min(1e-5)), atol=1e-6)
        assert (out >= 0).all()

    def test_reduces_to_cross_entropy_at_point_mass_target(self):
        """At a point-mass target the loss reduces to cross-entropy."""
        torch.manual_seed(0)
        logits = torch.randn(1, 1, 50)
        targets = torch.full((1, 1, 50), -30.0)
        targets[0, 0, 7] = 30.0  # ~one-hot target
        assert torch.allclose(
            neg_log_acceptance_loss(logits, targets),
            ce_loss(logits, targets),
            atol=1e-3,
        )

    def test_finite_at_zero_overlap(self):
        """The _EPS floor keeps the loss finite when overlap collapses to ~0."""
        logits = torch.full((1, 1, 50), -30.0)
        logits[0, 0, 0] = 30.0
        targets = torch.full((1, 1, 50), -30.0)
        targets[0, 0, 1] = 30.0
        assert torch.isfinite(neg_log_acceptance_loss(logits, targets)).all()

    def test_resolve_nla(self):
        """resolve_loss_fn maps 'nla' to neg_log_acceptance_loss."""
        assert resolve_loss_fn("nla") is neg_log_acceptance_loss


class TestLKHybridLoss:
    def test_eta_zero_reduces_to_kl(self):
        """eta=0 gives lambda=1 everywhere, so the loss is pure KL."""
        torch.manual_seed(0)
        logits, targets = torch.randn(1, 4, 50), torch.randn(1, 4, 50)
        assert torch.allclose(
            lk_hybrid_loss(logits, targets, eta=0.0),
            kl_div_loss(logits, targets),
            atol=1e-6,
        )

    def test_large_eta_reduces_to_tv(self):
        """Large eta drives lambda->0, so the loss approaches pure TV."""
        torch.manual_seed(0)
        logits, targets = torch.randn(1, 4, 50), torch.randn(1, 4, 50)
        assert torch.allclose(
            lk_hybrid_loss(logits, targets, eta=1e6),
            tv_loss(logits, targets),
            atol=1e-5,
        )

    def test_shape_and_finite(self):
        """Output is [1, seq_len] and finite at the default blend setting."""
        torch.manual_seed(0)
        logits, targets = torch.randn(1, 4, 50), torch.randn(1, 4, 50)
        out = lk_hybrid_loss(logits, targets, eta=3.0)
        assert out.shape == (1, 4)
        assert torch.isfinite(out).all()

    def test_alpha_is_detached_in_weight(self):
        """The alpha inside lambda must be stop-gradient.

        Verify the impl's gradient matches the detached form, and that NOT
        detaching would differ.
        """
        torch.manual_seed(0)
        logits = torch.randn(1, 3, 40, requires_grad=True)
        targets = torch.randn(1, 3, 40)
        g_impl = torch.autograd.grad(
            lk_hybrid_loss(logits, targets, eta=3.0).sum(), logits
        )[0]

        def manual(detach):
            dp, tp = torch.softmax(logits, -1), torch.softmax(targets, -1)
            ov = torch.minimum(dp, tp).sum(-1)
            alpha = ov.detach() if detach else ov
            lam = torch.exp(-3.0 * alpha)
            return (lam * kl_div_loss(logits, targets) + (1 - lam) * (1 - ov)).sum()

        g_detached = torch.autograd.grad(manual(True), logits, retain_graph=True)[0]
        g_nodetach = torch.autograd.grad(manual(False), logits)[0]
        assert torch.allclose(g_impl, g_detached, atol=1e-5)
        assert not torch.allclose(g_detached, g_nodetach, atol=1e-4)

    def test_resolve_lk_hybrid(self):
        """resolve_loss_fn maps 'lk_hybrid' to lk_hybrid_loss."""
        assert resolve_loss_fn("lk_hybrid") is lk_hybrid_loss


class TestComputeAccuracySingleStep:
    def test_prev_correct_chain(self):
        """Conditional accuracy across ttt steps tracks cumulative correctness.

        Step 0: positions [0,1,2] correct, [3] wrong → full=3/4
        Step 1: positions [0,1,3] correct, [2] wrong → only [0,1] still correct
        Conditional = 2 correct / 3 prev_correct
        prev_correct should be mutated to [T, T, F, F].
        """
        pred_step0 = torch.tensor([[1, 2, 3, 0]])
        tgt_step0 = torch.tensor([[1, 2, 3, 4]])
        prev_correct = torch.ones(1, 4, dtype=torch.bool)

        full_correct_0, full_total_0, cond_correct_0, cond_total_0 = (
            compute_accuracy_single_step(
                pred_step0,
                tgt_step0,
                loss_mask=None,
                prev_correct=prev_correct,
            )
        )
        assert full_correct_0.item() == pytest.approx(3, abs=1e-4)
        assert full_total_0.item() == pytest.approx(4, abs=1e-4)
        assert cond_correct_0.item() == pytest.approx(3, abs=1e-4)
        assert cond_total_0.item() == pytest.approx(4, abs=1e-4)
        assert prev_correct.tolist() == [[True, True, True, False]]

        pred_step1 = torch.tensor([[1, 2, 0, 4]])
        tgt_step1 = torch.tensor([[1, 2, 5, 4]])

        full_correct_1, full_total_1, cond_correct_1, cond_total_1 = (
            compute_accuracy_single_step(
                pred_step1,
                tgt_step1,
                loss_mask=None,
                prev_correct=prev_correct,
            )
        )
        # [0,1] correct on both steps, [2] was correct but now wrong,
        # [3] was already wrong
        assert prev_correct.tolist() == [[True, True, False, False]]
        assert full_correct_1.item() == pytest.approx(2, abs=1e-4)
        assert full_total_1.item() == pytest.approx(4, abs=1e-4)
        assert cond_correct_1.item() == pytest.approx(2, abs=1e-4)
        assert cond_total_1.item() == pytest.approx(3, abs=1e-4)


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
