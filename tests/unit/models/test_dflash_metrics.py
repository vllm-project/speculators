"""Unit tests for DFlash metrics and loss functions."""

from functools import partial

import pytest
import torch

from speculators.losses import dflash_loss_decay, loss_function, resolve_loss_config
from speculators.losses.eager import ce_loss
from speculators.models.dflash.metrics import compute_metrics as _compute_metrics

compute_metrics = partial(
    _compute_metrics, loss_config=resolve_loss_config("kl_div", "eager")
)


def _ids_to_logits(ids: torch.Tensor, vocab_size: int) -> torch.Tensor:
    """Convert token IDs to one-hot logits for testing."""
    logits = torch.zeros(*ids.shape, vocab_size)
    logits.scatter_(-1, ids.unsqueeze(-1), 100.0)
    return logits


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
        assert torch.isclose(metrics["loss_sum"], expected_loss)
        assert set(metrics) == {"loss_sum", "loss_total"}
