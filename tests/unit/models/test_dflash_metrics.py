"""Unit tests for DFlash metrics and loss functions."""

from functools import partial

import pytest
import torch

from speculators.models.dflash.metrics import compute_metrics
from speculators.models.metrics import (
    ce_loss,
    compute_accuracy_multi_step,
    dflash_loss_decay,
    loss_function,
)


def _ids_to_logits(ids: torch.Tensor, vocab_size: int) -> torch.Tensor:
    """Convert token IDs to one-hot logits for testing."""
    logits = torch.zeros(*ids.shape, vocab_size)
    logits.scatter_(-1, ids.unsqueeze(-1), 100.0)
    return logits


class TestComputeAccuracy:
    def test_perfect_accuracy(self):
        logits = torch.tensor([[[0.0, 0.0, 1.0], [0.0, 1.0, 0.0]]])
        targets = _ids_to_logits(torch.tensor([[2, 1]]), 3)
        pred_ids = torch.argmax(logits, dim=-1)
        target_ids = torch.argmax(targets, dim=-1)
        loss_mask = torch.tensor([[1, 1]])
        pos_idx = torch.tensor([[0, 1]])
        correct, total = compute_accuracy_multi_step(
            pred_ids, target_ids, loss_mask, pos_idx, 2
        )
        assert correct.sum() / total.sum() == pytest.approx(1.0, abs=1e-4)

    def test_zero_accuracy(self):
        logits = torch.tensor([[[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]])
        targets = _ids_to_logits(torch.tensor([[2, 1]]), 3)
        pred_ids = torch.argmax(logits, dim=-1)
        target_ids = torch.argmax(targets, dim=-1)
        loss_mask = torch.tensor([[1, 1]])
        pos_idx = torch.tensor([[0, 1]])
        correct, total = compute_accuracy_multi_step(
            pred_ids, target_ids, loss_mask, pos_idx, 2
        )
        assert correct.sum() / total.sum() == pytest.approx(0.0, abs=1e-4)

    def test_partial_accuracy(self):
        logits = torch.tensor([[[0.0, 0.0, 1.0], [1.0, 0.0, 0.0]]])
        targets = _ids_to_logits(torch.tensor([[2, 1]]), 3)
        pred_ids = torch.argmax(logits, dim=-1)
        target_ids = torch.argmax(targets, dim=-1)
        loss_mask = torch.tensor([[1, 1]])
        pos_idx = torch.tensor([[0, 1]])
        correct, total = compute_accuracy_multi_step(
            pred_ids, target_ids, loss_mask, pos_idx, 2
        )
        assert correct.sum() / total.sum() == pytest.approx(0.5, abs=1e-4)

    def test_loss_mask_excludes_positions(self):
        logits = torch.tensor([[[0.0, 0.0, 1.0], [1.0, 0.0, 0.0]]])
        targets = _ids_to_logits(torch.tensor([[2, 1]]), 3)
        pred_ids = torch.argmax(logits, dim=-1)
        target_ids = torch.argmax(targets, dim=-1)
        loss_mask = torch.tensor([[1, 0]])
        pos_idx = torch.tensor([[0, 1]])
        correct, total = compute_accuracy_multi_step(
            pred_ids, target_ids, loss_mask, pos_idx, 2
        )
        assert correct.sum() / total.sum() == pytest.approx(1.0, abs=1e-4)

    def test_all_masked_out(self):
        logits = torch.tensor([[[0.0, 0.0, 1.0], [0.0, 1.0, 0.0]]])
        targets = _ids_to_logits(torch.tensor([[2, 1]]), 3)
        pred_ids = torch.argmax(logits, dim=-1)
        target_ids = torch.argmax(targets, dim=-1)
        loss_mask = torch.tensor([[0, 0]])
        pos_idx = torch.tensor([[0, 1]])
        correct, total = compute_accuracy_multi_step(
            pred_ids, target_ids, loss_mask, pos_idx, 2
        )
        assert correct.sum() == pytest.approx(0.0, abs=1e-4)
        assert total.sum() == pytest.approx(0.0, abs=1e-4)

    def test_block_size_per_position_counts(self):
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
        targets = _ids_to_logits(torch.tensor([[2, 1, 0, 0]]), 3)
        pred_ids = torch.argmax(logits, dim=-1)
        target_ids = torch.argmax(targets, dim=-1)
        loss_mask = torch.tensor([[1, 1, 1, 1]])
        pos_idx = torch.arange(4).unsqueeze(0) % 2
        correct, total = compute_accuracy_multi_step(
            pred_ids, target_ids, loss_mask, pos_idx, 2
        )
        assert len(correct) == 2
        # pos 0: 2/2 correct, pos 1: 1/2 correct
        assert correct[0] / total[0] == pytest.approx(1.0, abs=1e-4)
        assert correct[1] / total[1] == pytest.approx(0.5, abs=1e-4)

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
        targets = _ids_to_logits(torch.tensor([[2, 0, 0, 1]]), 3)
        pred_ids = torch.argmax(logits, dim=-1)
        target_ids = torch.argmax(targets, dim=-1)
        loss_mask = torch.tensor([[1, 0, 1, 1]])
        pos_idx = torch.arange(4).unsqueeze(0) % 2
        correct, total = compute_accuracy_multi_step(
            pred_ids, target_ids, loss_mask, pos_idx, 2
        )
        assert correct[0] / total[0] == pytest.approx(1.0, abs=1e-4)
        assert correct[1] / total[1] == pytest.approx(1.0, abs=1e-4)

    def test_returns_tensors_per_position(self):
        """Per-position counts should be 1D tensors of shape [block_size]."""
        logits = torch.randn(1, 8, 10)
        targets = _ids_to_logits(torch.randint(0, 10, (1, 8)), 10)
        pred_ids = torch.argmax(logits, dim=-1)
        target_ids = torch.argmax(targets, dim=-1)
        loss_mask = torch.ones(1, 8)
        pos_idx = torch.arange(8).unsqueeze(0) % 4
        correct, total = compute_accuracy_multi_step(
            pred_ids, target_ids, loss_mask, pos_idx, 4
        )
        assert isinstance(correct, torch.Tensor)
        assert isinstance(total, torch.Tensor)
        assert len(correct) == 4
        assert len(total) == 4

    def test_overall_counts_consistent_with_per_position(self):
        """Sum of per-position counts should equal overall counts."""
        logits = torch.randn(1, 6, 5)
        targets = _ids_to_logits(torch.randint(0, 5, (1, 6)), 5)
        pred_ids = torch.argmax(logits, dim=-1)
        target_ids = torch.argmax(targets, dim=-1)
        loss_mask = torch.ones(1, 6)
        pos_idx = torch.arange(6).unsqueeze(0) % 3
        correct, total = compute_accuracy_multi_step(
            pred_ids, target_ids, loss_mask, pos_idx, 3
        )
        assert total.sum().item() == pytest.approx(6.0, abs=1e-4)
        assert correct.sum().item() <= total.sum().item() + 1e-4


class TestLossFunction:
    def test_basic_loss_not_nan(self):
        B, T, V = 2, 8, 10
        logits = torch.randn(B, T, V)
        targets = _ids_to_logits(torch.randint(0, V, (B, T)), V)
        loss_mask = torch.ones(B, T)
        pos_idx = torch.arange(T).unsqueeze(0).expand(B, -1) % 8
        loss = loss_function(
            logits,
            targets,
            loss_mask,
            pos_idx,
            loss_fn=ce_loss,
            decay_fn=partial(dflash_loss_decay, gamma=4.0),
        )
        assert not torch.isnan(loss)
        assert not torch.isinf(loss)
        assert loss.ndim == 0

    def test_anchor_positions_have_zero_weight(self):
        """Position 0 in each block (anchor) should have zero weight."""
        T = 8
        pos_idx = torch.arange(T)
        decay = dflash_loss_decay(pos_idx.float(), gamma=4.0)
        assert decay[0].item() == 0.0  # anchor position has zero weight

    def test_loss_mask_zeros_out_positions(self):
        """Loss mask of zeros should zero out all positions."""
        B, T, V = 1, 8, 10
        logits = torch.randn(B, T, V)
        targets = _ids_to_logits(torch.randint(0, V, (B, T)), V)
        loss_mask = torch.zeros(B, T)
        pos_idx = torch.arange(T).unsqueeze(0) % 8
        loss = loss_function(
            logits,
            targets,
            loss_mask,
            pos_idx,
            loss_fn=ce_loss,
            decay_fn=partial(dflash_loss_decay, gamma=4.0),
        )
        assert loss.item() == pytest.approx(0.0, abs=1e-4)

    def test_different_gamma(self, seed):
        """Different gamma values should produce different losses."""
        B, T, V = 1, 16, 10
        logits = torch.randn(B, T, V)
        targets = _ids_to_logits(torch.randint(0, V, (B, T)), V)
        loss_mask = torch.ones(B, T)
        pos_idx = torch.arange(T).unsqueeze(0) % 8
        loss_g1 = loss_function(
            logits,
            targets,
            loss_mask,
            pos_idx,
            loss_fn=ce_loss,
            decay_fn=partial(dflash_loss_decay, gamma=1.0),
        )
        loss_g10 = loss_function(
            logits,
            targets,
            loss_mask,
            pos_idx,
            loss_fn=ce_loss,
            decay_fn=partial(dflash_loss_decay, gamma=10.0),
        )
        assert not torch.isclose(loss_g1, loss_g10)

    def test_different_block_sizes(self, seed):
        """Different block sizes should produce different weight patterns."""
        B, T, V = 1, 16, 10
        logits = torch.randn(B, T, V)
        targets = _ids_to_logits(torch.randint(0, V, (B, T)), V)
        loss_mask = torch.ones(B, T)
        pos_idx_b4 = torch.arange(T).unsqueeze(0) % 4
        pos_idx_b8 = torch.arange(T).unsqueeze(0) % 8
        loss_b4 = loss_function(
            logits,
            targets,
            loss_mask,
            pos_idx_b4,
            loss_fn=ce_loss,
            decay_fn=partial(dflash_loss_decay, gamma=4.0),
        )
        loss_b8 = loss_function(
            logits,
            targets,
            loss_mask,
            pos_idx_b8,
            loss_fn=ce_loss,
            decay_fn=partial(dflash_loss_decay, gamma=4.0),
        )
        assert not torch.isclose(loss_b4, loss_b8)

    def test_perfect_predictions_low_loss(self):
        """When logits strongly predict the correct targets, loss should be low."""
        B, T, V = 1, 8, 5
        target_ids = torch.tensor([[0, 1, 2, 3, 4, 0, 1, 2]])
        targets = _ids_to_logits(target_ids, V)
        logits = torch.zeros(B, T, V)
        for t in range(T):
            logits[0, t, target_ids[0, t]] = 100.0
        loss_mask = torch.ones(B, T)
        pos_idx = torch.arange(T).unsqueeze(0) % 8
        loss = loss_function(
            logits,
            targets,
            loss_mask,
            pos_idx,
            loss_fn=ce_loss,
            decay_fn=partial(dflash_loss_decay, gamma=4.0),
        )
        assert loss.item() < 0.01


class TestComputeMetrics:
    def test_returns_loss_and_dict(self):
        B, T, V = 1, 8, 10
        logits = torch.randn(B, T, V)
        targets = _ids_to_logits(torch.randint(0, V, (B, T)), V)
        loss_mask = torch.ones(B, T)
        loss, metrics = compute_metrics(logits, targets, loss_mask, block_size=4)
        assert isinstance(loss, torch.Tensor)
        assert loss.ndim == 0
        assert isinstance(metrics, dict)
        assert "loss sum" in metrics
        assert "loss count" in metrics
        assert "full_acc sum" in metrics
        assert "full_acc count" in metrics

    def test_per_position_keys(self):
        B, T, V = 1, 8, 10
        logits = torch.randn(B, T, V)
        targets = _ids_to_logits(torch.randint(0, V, (B, T)), V)
        loss_mask = torch.ones(B, T)
        _, metrics = compute_metrics(logits, targets, loss_mask, block_size=4)
        assert "position 0 acc sum" not in metrics
        for i in range(1, 4):
            assert f"position {i} acc sum" in metrics
            assert f"position {i} acc count" in metrics

    def test_loss_matches_loss_function(self):
        B, T, V = 1, 8, 10
        logits = torch.randn(B, T, V)
        targets = _ids_to_logits(torch.randint(0, V, (B, T)), V)
        loss_mask = torch.ones(B, T)
        loss, metrics = compute_metrics(logits, targets, loss_mask, block_size=4)
        pos_idx = torch.arange(T).unsqueeze(0) % 4
        expected_loss = loss_function(
            logits,
            targets,
            loss_mask,
            pos_idx,
            loss_fn=ce_loss,
            decay_fn=partial(dflash_loss_decay, gamma=4.0),
        )
        assert torch.isclose(loss, expected_loss)
        assert torch.isclose(metrics["loss sum"], expected_loss)

    def test_counts_match_compute_accuracy(self):
        B, T, V = 1, 8, 10
        logits = torch.randn(B, T, V)
        targets = _ids_to_logits(torch.randint(0, V, (B, T)), V)
        loss_mask = torch.ones(B, T)
        _, metrics = compute_metrics(logits, targets, loss_mask, block_size=4)
        pred_ids = torch.argmax(logits, dim=-1)
        target_ids = torch.argmax(targets, dim=-1)
        pos_idx = torch.arange(T).unsqueeze(0) % 4
        expected_correct, expected_total = compute_accuracy_multi_step(
            pred_ids, target_ids, loss_mask, pos_idx, 4
        )
        # full counts = sum of positions 1+ (position 0 excluded)
        assert torch.isclose(metrics["full_acc sum"], expected_correct[1:].sum())
        assert torch.isclose(metrics["full_acc count"], expected_total[1:].sum())
        for i in range(1, 4):
            assert torch.isclose(metrics[f"position {i} acc sum"], expected_correct[i])
            assert torch.isclose(metrics[f"position {i} acc count"], expected_total[i])
