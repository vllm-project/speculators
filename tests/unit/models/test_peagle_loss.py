"""Unit tests for P-EAGLE loss functions."""

import pytest
import torch

from speculators.models.peagle.loss import (
    cross_entropy_loss,
    kl_div_loss,
    per_depth_accuracy,
    per_depth_loss,
)


class TestCrossEntropyLoss:
    """Tests for cross_entropy_loss."""

    def test_basic_loss(self):
        """Test basic cross-entropy computation."""
        batch, seq_len, vocab_size = 1, 4, 10
        logits = torch.randn(batch, seq_len, vocab_size)
        targets = torch.randint(0, vocab_size, (batch, seq_len))

        loss = cross_entropy_loss(logits, targets)
        assert loss.ndim == 0  # scalar
        assert loss.item() > 0

    def test_perfect_predictions_low_loss(self):
        """Perfect predictions should give very low loss."""
        batch, seq_len, vocab_size = 1, 5, 10
        targets = torch.randint(0, vocab_size, (batch, seq_len))

        # Create logits that strongly predict the targets
        logits = torch.full((batch, seq_len, vocab_size), -10.0)
        for b in range(batch):
            for s in range(seq_len):
                logits[b, s, targets[b, s]] = 10.0

        loss = cross_entropy_loss(logits, targets)
        assert loss.item() < 0.01

    def test_with_loss_mask(self):
        """Test that loss mask correctly excludes positions."""
        batch, seq_len, vocab_size = 1, 6, 5
        logits = torch.randn(batch, seq_len, vocab_size)
        targets = torch.randint(0, vocab_size, (batch, seq_len))

        # All positions
        loss_all = cross_entropy_loss(logits, targets)

        # Only first 3 positions
        mask = torch.tensor([[1, 1, 1, 0, 0, 0]], dtype=torch.float32)
        loss_masked = cross_entropy_loss(logits, targets, mask)

        # Losses should be different since they use different positions
        assert loss_masked.item() != loss_all.item()

    def test_zero_mask(self):
        """All-zero mask should give near-zero loss."""
        batch, seq_len, vocab_size = 1, 4, 10
        logits = torch.randn(batch, seq_len, vocab_size)
        targets = torch.randint(0, vocab_size, (batch, seq_len))
        mask = torch.zeros(batch, seq_len)

        loss = cross_entropy_loss(logits, targets, mask)
        assert abs(loss.item()) < 0.01

    def test_batch_dimension(self):
        """Test with batch_size > 1."""
        batch, seq_len, vocab_size = 4, 8, 20
        logits = torch.randn(batch, seq_len, vocab_size)
        targets = torch.randint(0, vocab_size, (batch, seq_len))

        loss = cross_entropy_loss(logits, targets)
        assert loss.ndim == 0
        assert loss.item() > 0


class TestKlDivLoss:
    """Tests for kl_div_loss."""

    def test_basic_loss(self):
        """Test basic KL divergence computation."""
        batch, seq_len, vocab_size = 1, 4, 10
        logits = torch.randn(batch, seq_len, vocab_size)
        targets = torch.randn(batch, seq_len, vocab_size)

        loss = kl_div_loss(logits, targets)
        assert loss.ndim == 0
        assert loss.item() >= 0

    def test_identical_distributions_zero_loss(self):
        """Same distributions should give zero KL divergence."""
        batch, seq_len, vocab_size = 1, 5, 10
        logits = torch.randn(batch, seq_len, vocab_size)

        loss = kl_div_loss(logits, logits)
        assert loss.item() < 0.01

    def test_with_loss_mask(self):
        """Test KL divergence with loss mask."""
        batch, seq_len, vocab_size = 1, 6, 5
        logits = torch.randn(batch, seq_len, vocab_size)
        targets = torch.randn(batch, seq_len, vocab_size)

        mask = torch.tensor([[1, 1, 1, 0, 0, 0]], dtype=torch.float32)
        loss = kl_div_loss(logits, targets, mask)
        assert loss.ndim == 0


class TestPerDepthLoss:
    """Tests for per_depth_loss."""

    def test_basic_cross_entropy(self):
        """Test per-depth cross-entropy loss aggregation."""
        vocab_size = 10
        all_logits = [
            torch.randn(1, 8, vocab_size),
            torch.randn(1, 4, vocab_size),
            torch.randn(1, 2, vocab_size),
        ]
        all_targets = [
            torch.randint(0, vocab_size, (1, 8)),
            torch.randint(0, vocab_size, (1, 4)),
            torch.randint(0, vocab_size, (1, 2)),
        ]
        all_masks = [None, None, None]

        total_loss, metrics = per_depth_loss(
            all_logits, all_targets, all_masks, loss_fn="cross_entropy"
        )

        assert total_loss.ndim == 0
        assert total_loss.item() > 0
        assert "loss" in metrics
        assert "loss_depth_0" in metrics
        assert "loss_depth_1" in metrics
        assert "loss_depth_2" in metrics

    def test_kl_div_loss_fn(self):
        """Test per-depth KL divergence loss."""
        vocab_size = 10
        all_logits = [
            torch.randn(1, 8, vocab_size),
            torch.randn(1, 4, vocab_size),
        ]
        all_targets = [
            torch.randn(1, 8, vocab_size),
            torch.randn(1, 4, vocab_size),
        ]
        all_masks = [None, None]

        total_loss, metrics = per_depth_loss(
            all_logits, all_targets, all_masks, loss_fn="kl_div"
        )

        assert total_loss.ndim == 0
        assert "loss_depth_0" in metrics
        assert "loss_depth_1" in metrics

    def test_depth_weights(self):
        """Test custom depth loss weights."""
        vocab_size = 10
        all_logits = [
            torch.randn(1, 5, vocab_size),
            torch.randn(1, 5, vocab_size),
        ]
        all_targets = [
            torch.randint(0, vocab_size, (1, 5)),
            torch.randint(0, vocab_size, (1, 5)),
        ]
        all_masks = [None, None]

        # Equal weights
        loss_equal, _ = per_depth_loss(
            all_logits, all_targets, all_masks,
            loss_fn="cross_entropy",
            depth_loss_weights=[1.0, 1.0],
        )

        # Zero weight for depth 1
        loss_weighted, _ = per_depth_loss(
            all_logits, all_targets, all_masks,
            loss_fn="cross_entropy",
            depth_loss_weights=[1.0, 0.0],
        )

        # Weighted should be less (only depth 0)
        assert loss_weighted.item() < loss_equal.item()

    def test_invalid_loss_fn(self):
        """Should raise ValueError for unknown loss function."""
        all_logits = [torch.randn(1, 4, 10)]
        all_targets = [torch.randint(0, 10, (1, 4))]
        all_masks = [None]

        with pytest.raises(ValueError, match="Unknown loss function"):
            per_depth_loss(all_logits, all_targets, all_masks, loss_fn="invalid")

    def test_mismatched_weights_length(self):
        """Should raise ValueError for mismatched weights."""
        all_logits = [torch.randn(1, 4, 10)]
        all_targets = [torch.randint(0, 10, (1, 4))]
        all_masks = [None]

        with pytest.raises(ValueError, match="depth_loss_weights length"):
            per_depth_loss(
                all_logits, all_targets, all_masks,
                depth_loss_weights=[1.0, 2.0],
            )

    def test_empty_logits(self):
        """Test with empty logit tensors (from COD sampling with 0 positions)."""
        vocab_size = 10
        all_logits = [
            torch.randn(1, 5, vocab_size),
            torch.empty(0, 0, vocab_size),  # Empty depth
        ]
        all_targets = [
            torch.randint(0, vocab_size, (1, 5)),
            torch.empty(0, 0, dtype=torch.long),
        ]
        all_masks = [None, None]

        total_loss, metrics = per_depth_loss(
            all_logits, all_targets, all_masks, loss_fn="cross_entropy"
        )

        assert total_loss.item() > 0
        assert metrics["loss_depth_1"].item() == 0.0


class TestPerDepthAccuracy:
    """Tests for per_depth_accuracy."""

    def test_perfect_accuracy(self):
        """Perfect predictions should give 100% accuracy."""
        vocab_size = 10
        target_ids = [torch.randint(0, vocab_size, (1, 5))]

        # Create logits that predict the target
        logits = [torch.full((1, 5, vocab_size), -10.0)]
        for s in range(5):
            logits[0][0, s, target_ids[0][0, s]] = 10.0

        metrics = per_depth_accuracy(logits, target_ids, [None])
        assert abs(metrics["acc_depth_0"].item() - 1.0) < 0.01

    def test_random_accuracy(self):
        """Random predictions should give well below 100%."""
        vocab_size = 100
        target_ids = [torch.randint(0, vocab_size, (1, 100))]
        logits = [torch.randn(1, 100, vocab_size)]

        metrics = per_depth_accuracy(logits, target_ids, [None])
        # Random accuracy should be around 1%
        assert metrics["acc_depth_0"].item() < 0.2

    def test_with_mask(self):
        """Masks should filter positions for accuracy."""
        vocab_size = 5
        target_ids = [torch.tensor([[0, 1, 2, 3, 4]])]

        logits = [torch.full((1, 5, vocab_size), -10.0)]
        # Only first 2 correct
        logits[0][0, 0, 0] = 10.0
        logits[0][0, 1, 1] = 10.0
        logits[0][0, 2, 0] = 10.0  # wrong
        logits[0][0, 3, 0] = 10.0  # wrong
        logits[0][0, 4, 0] = 10.0  # wrong

        # Mask only first 2 positions
        mask = torch.tensor([[1, 1, 0, 0, 0]], dtype=torch.float32)
        metrics = per_depth_accuracy(logits, target_ids, [mask])

        assert abs(metrics["acc_depth_0"].item() - 1.0) < 0.01

    def test_multi_depth(self):
        """Test accuracy across multiple depths."""
        vocab_size = 10
        logits = [
            torch.randn(1, 5, vocab_size),
            torch.randn(1, 3, vocab_size),
        ]
        targets = [
            torch.randint(0, vocab_size, (1, 5)),
            torch.randint(0, vocab_size, (1, 3)),
        ]

        metrics = per_depth_accuracy(logits, targets, [None, None])
        assert "acc_depth_0" in metrics
        assert "acc_depth_1" in metrics
