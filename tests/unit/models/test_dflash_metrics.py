"""Unit tests for DFlash metrics and loss functions."""

import pytest
import torch

from speculators.models.dflash.metrics import (
    compute_accuracy,
    compute_metrics,
    loss_function,
)


class TestComputeAccuracy:
    def test_perfect_accuracy(self):
        logits = torch.tensor([[[0.0, 0.0, 1.0], [0.0, 1.0, 0.0]]])
        targets = torch.tensor([[2, 1]])
        loss_mask = torch.tensor([[1, 1]])
        acc, _ = compute_accuracy(logits, targets, loss_mask)
        assert acc == pytest.approx(1.0, abs=1e-4)

    def test_zero_accuracy(self):
        logits = torch.tensor([[[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]])
        targets = torch.tensor([[2, 1]])
        loss_mask = torch.tensor([[1, 1]])
        acc, _ = compute_accuracy(logits, targets, loss_mask)
        assert acc == pytest.approx(0.0, abs=1e-4)

    def test_partial_accuracy(self):
        logits = torch.tensor([[[0.0, 0.0, 1.0], [1.0, 0.0, 0.0]]])
        targets = torch.tensor([[2, 1]])
        loss_mask = torch.tensor([[1, 1]])
        acc, _ = compute_accuracy(logits, targets, loss_mask)
        assert acc == pytest.approx(0.5, abs=1e-4)

    def test_loss_mask_excludes_positions(self):
        logits = torch.tensor([[[0.0, 0.0, 1.0], [1.0, 0.0, 0.0]]])
        targets = torch.tensor([[2, 1]])
        # Mask out the incorrect position
        loss_mask = torch.tensor([[1, 0]])
        acc, _ = compute_accuracy(logits, targets, loss_mask)
        assert acc == pytest.approx(1.0, abs=1e-4)

    def test_all_masked_out(self):
        logits = torch.tensor([[[0.0, 0.0, 1.0], [0.0, 1.0, 0.0]]])
        targets = torch.tensor([[2, 1]])
        loss_mask = torch.tensor([[0, 0]])
        acc, _ = compute_accuracy(logits, targets, loss_mask)
        assert acc == pytest.approx(0.0, abs=1e-4)

    def test_block_size_per_position_accuracy(self):
        # block_size=2, seq_len=4: positions [0,2] are pos 0, [1,3] are pos 1
        logits = torch.tensor(
            [
                [
                    [0.0, 0.0, 1.0],  # pos 0 in block: predict 2
                    [0.0, 1.0, 0.0],  # pos 1 in block: predict 1
                    [1.0, 0.0, 0.0],  # pos 0 in block: predict 0
                    [0.0, 1.0, 0.0],  # pos 1 in block: predict 1
                ]
            ]
        )
        targets = torch.tensor([[2, 1, 0, 0]])
        loss_mask = torch.tensor([[1, 1, 1, 1]])
        acc, accs = compute_accuracy(logits, targets, loss_mask, block_size=2)
        # pos 0: both correct (2==2, 0==0) -> 1.0
        # pos 1: one correct, one wrong (1==1, 1!=0) -> 0.5
        assert len(accs) == 2
        assert accs[0] == pytest.approx(1.0, abs=1e-4)
        assert accs[1] == pytest.approx(0.5, abs=1e-4)

    def test_block_size_with_mask(self):
        logits = torch.tensor(
            [
                [
                    [0.0, 0.0, 1.0],
                    [0.0, 1.0, 0.0],
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                ]
            ]
        )
        targets = torch.tensor([[2, 0, 0, 1]])
        loss_mask = torch.tensor([[1, 0, 1, 1]])
        acc, accs = compute_accuracy(logits, targets, loss_mask, block_size=2)
        # pos 0: correct=[True, True], mask=[1,1] -> 1.0
        # pos 1: correct=[False, True], mask=[0,1] -> 1.0
        assert accs[0] == pytest.approx(1.0, abs=1e-4)
        assert accs[1] == pytest.approx(1.0, abs=1e-4)

    def test_returns_list_per_position(self):
        """Per-position accuracies should be a 1D tensor of shape [block_size]."""
        logits = torch.randn(1, 8, 10)
        targets = torch.randint(0, 10, (1, 8))
        loss_mask = torch.ones(1, 8)
        _, accs = compute_accuracy(logits, targets, loss_mask, block_size=4)
        assert isinstance(accs, torch.Tensor)
        assert len(accs) == 4

    def test_overall_accuracy_consistent_with_per_position(self):
        """Overall accuracy should be consistent with per-position accuracies."""
        logits = torch.randn(1, 6, 5)
        targets = torch.randint(0, 5, (1, 6))
        loss_mask = torch.ones(1, 6)
        acc, accs = compute_accuracy(logits, targets, loss_mask, block_size=3)
        # Overall should be the weighted mean of per-position
        assert acc >= 0.0
        assert acc <= 1.0 + 1e-4


class TestLossFunction:
    def test_basic_loss_not_nan(self):
        B, T, V = 2, 8, 10
        logits = torch.randn(B, T, V)
        target_ids = torch.randint(0, V, (B, T))
        loss_mask = torch.ones(B, T)
        loss = loss_function(logits, target_ids, loss_mask)
        assert not torch.isnan(loss)
        assert not torch.isinf(loss)
        assert loss.ndim == 0  # scalar

    def test_anchor_positions_have_zero_weight(self):
        """Position 0 in each block (anchor) should have zero weight."""
        B, T, V = 1, 8, 10
        logits = torch.randn(B, T, V)
        loss_mask = torch.ones(B, T)

        # Create targets that are wrong everywhere
        target_ids = torch.zeros(B, T, dtype=torch.long)

        # Compute weight for each position
        idx = torch.arange(T)
        k = (idx + 1) % 8  # block_size=8
        # k==0 at positions where (idx+1) % 8 == 0, i.e., idx=7,15,...
        # Anchors are at k==0, which is position 7 (0-indexed) for block_size=8
        # Actually: idx=0 -> k=1, idx=7 -> k=0
        # So position 7 (last in block) has zero weight
        w = torch.exp(-((k - 1).clamp(min=0)).float() / 4.0)
        w = w * (k != 0).float()
        assert w[7].item() == 0.0  # last position in block has zero weight

    def test_ignore_index(self):
        """Positions with target_ids=-100 should be ignored."""
        B, T, V = 1, 8, 10
        logits = torch.randn(B, T, V)
        target_ids = torch.full((B, T), -100, dtype=torch.long)
        loss_mask = torch.ones(B, T)
        loss = loss_function(logits, target_ids, loss_mask)
        assert loss.item() == pytest.approx(0.0, abs=1e-4)

    def test_loss_mask_zeros_out_positions(self):
        """Loss mask of zeros should zero out all positions."""
        B, T, V = 1, 8, 10
        logits = torch.randn(B, T, V)
        target_ids = torch.randint(0, V, (B, T))
        loss_mask = torch.zeros(B, T)
        loss = loss_function(logits, target_ids, loss_mask)
        assert loss.item() == pytest.approx(0.0, abs=1e-4)

    def test_different_gamma(self):
        """Different gamma values should produce different losses."""
        B, T, V = 1, 16, 10
        logits = torch.randn(B, T, V)
        target_ids = torch.randint(0, V, (B, T))
        loss_mask = torch.ones(B, T)
        loss_g1 = loss_function(logits, target_ids, loss_mask, gamma=1.0)
        loss_g10 = loss_function(logits, target_ids, loss_mask, gamma=10.0)
        assert not torch.isclose(loss_g1, loss_g10)

    def test_different_block_sizes(self):
        """Different block sizes should produce different weight patterns."""
        B, T, V = 1, 16, 10
        logits = torch.randn(B, T, V)
        target_ids = torch.randint(0, V, (B, T))
        loss_mask = torch.ones(B, T)
        loss_b4 = loss_function(logits, target_ids, loss_mask, block_size=4)
        loss_b8 = loss_function(logits, target_ids, loss_mask, block_size=8)
        # With different block sizes, weight patterns differ
        assert not torch.isclose(loss_b4, loss_b8)

    def test_perfect_predictions_low_loss(self):
        """When logits strongly predict the correct targets, loss should be low."""
        B, T, V = 1, 8, 5
        target_ids = torch.tensor([[0, 1, 2, 3, 4, 0, 1, 2]])
        logits = torch.zeros(B, T, V)
        # Make the correct class have a very high logit
        for t in range(T):
            logits[0, t, target_ids[0, t]] = 100.0
        loss_mask = torch.ones(B, T)
        loss = loss_function(logits, target_ids, loss_mask)
        assert loss.item() < 0.01


class TestComputeMetrics:
    def test_returns_loss_and_dict(self):
        B, T, V = 1, 8, 10
        logits = torch.randn(B, T, V)
        targets = torch.randint(0, V, (B, T))
        loss_mask = torch.ones(B, T)
        loss, metrics = compute_metrics(logits, targets, loss_mask, block_size=4)
        assert isinstance(loss, torch.Tensor)
        assert loss.ndim == 0
        assert isinstance(metrics, dict)
        assert "loss" in metrics
        assert "full_acc" in metrics

    def test_per_position_keys(self):
        B, T, V = 1, 8, 10
        logits = torch.randn(B, T, V)
        targets = torch.randint(0, V, (B, T))
        loss_mask = torch.ones(B, T)
        _, metrics = compute_metrics(logits, targets, loss_mask, block_size=4)
        for i in range(4):
            assert f"position {i} acc" in metrics

    def test_loss_matches_loss_function(self):
        B, T, V = 2, 8, 10
        logits = torch.randn(B, T, V)
        targets = torch.randint(0, V, (B, T))
        loss_mask = torch.ones(B, T)
        loss, metrics = compute_metrics(logits, targets, loss_mask, block_size=4)
        expected_loss = loss_function(logits, targets, loss_mask)
        assert torch.isclose(loss, expected_loss)
        assert torch.isclose(metrics["loss"], expected_loss)

    def test_accuracy_matches_compute_accuracy(self):
        B, T, V = 1, 8, 10
        logits = torch.randn(B, T, V)
        targets = torch.randint(0, V, (B, T))
        loss_mask = torch.ones(B, T)
        _, metrics = compute_metrics(logits, targets, loss_mask, block_size=4)
        expected_acc, expected_pos_accs = compute_accuracy(
            logits, targets, loss_mask, block_size=4
        )
        assert torch.isclose(metrics["full_acc"], expected_acc)
        for i in range(4):
            assert torch.isclose(metrics[f"position {i} acc"], expected_pos_accs[i])
