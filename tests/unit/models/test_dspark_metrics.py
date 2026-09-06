"""Unit tests for the DSpark loss and metrics."""

from functools import partial
from unittest.mock import Mock

import torch

from speculators.losses import resolve_loss_config
from speculators.losses.eager import tv_loss
from speculators.models.dspark.metrics import compute_metrics as _compute_metrics

compute_metrics = partial(_compute_metrics, tv_loss_fn=tv_loss)
_DEFAULT_LOSS = resolve_loss_config('{"ce": 0.1, "tv": 0.9}', "eager")


def _ids_to_logits(ids: torch.Tensor, vocab_size: int) -> torch.Tensor:
    logits = torch.zeros(*ids.shape, vocab_size)
    logits.scatter_(-1, ids.unsqueeze(-1), 100.0)
    return logits


class TestComputeMetrics:
    def test_no_confidence_head_skips_diagnostic_tv(self):
        diagnostic_tv = Mock(side_effect=AssertionError("unneeded TV evaluation"))
        logits = torch.randn(1, 4, 8, requires_grad=True)
        loss, metrics = compute_metrics(
            logits,
            torch.randn_like(logits),
            None,
            torch.ones(1, 4),
            2,
            loss_config=_DEFAULT_LOSS,
            tv_loss_fn=diagnostic_tv,
        )
        loss.backward()
        diagnostic_tv.assert_not_called()
        assert logits.grad is not None
        assert set(metrics) == {
            "loss_sum",
            "loss_total",
            "ce_loss_sum",
            "ce_loss_total",
            "tv_loss_sum",
            "tv_loss_total",
        }

    def test_empty_loss_mask_keeps_backward_graph_with_zero_gradients(self):
        logits = torch.randn(1, 8, 16, requires_grad=True)
        targets = torch.randn(1, 8, 16)
        confidence_logits = torch.randn(1, 8, requires_grad=True)
        loss_mask = torch.zeros(1, 8)

        loss, _ = compute_metrics(
            logits,
            targets,
            confidence_logits,
            loss_mask,
            block_size=2,
            loss_config=_DEFAULT_LOSS,
        )
        loss.backward()

        assert loss.item() == 0
        assert logits.grad is not None
        assert confidence_logits.grad is not None
        assert torch.count_nonzero(logits.grad) == 0
        assert torch.count_nonzero(confidence_logits.grad) == 0

    def test_confidence_target_is_overlap(self):
        # When draft == target, accept rate == 1, so a confidence logit that is
        # very positive (sigmoid -> 1) yields ~zero abs error.
        ids = torch.tensor([[0, 1, 0, 2]])
        logits = _ids_to_logits(ids, 8)
        targets = logits.clone()
        loss_mask = torch.tensor([[0, 1, 0, 1]], dtype=torch.float32)
        confidence_logits = torch.full((1, 4), 20.0)  # sigmoid ~ 1.0
        _, metrics = compute_metrics(
            logits,
            targets,
            confidence_logits,
            loss_mask,
            block_size=2,
            gamma=4.0,
            loss_config=_DEFAULT_LOSS,
        )
        abs_err = (
            metrics["confidence_abs_error_sum"] / metrics["confidence_abs_error_total"]
        )
        assert float(abs_err) < 1e-2
        assert "confidence_loss_sum" in metrics

    def test_confidence_term_changes_loss(self):
        ids = torch.tensor([[0, 1, 0, 2]])
        logits = _ids_to_logits(ids, 8)
        targets = _ids_to_logits(torch.tensor([[0, 3, 0, 4]]), 8)
        loss_mask = torch.tensor([[0, 1, 0, 1]], dtype=torch.float32)
        loss_no_conf, _ = compute_metrics(
            logits,
            targets,
            None,
            loss_mask,
            block_size=2,
            loss_config=_DEFAULT_LOSS,
        )
        # A badly-calibrated confidence head (predicts accept~1 when accept~0)
        # must add positive BCE on top of the base loss.
        confidence_logits = torch.full((1, 4), 20.0)
        loss_conf, _ = compute_metrics(
            logits,
            targets,
            confidence_logits,
            loss_mask,
            block_size=2,
            loss_config=_DEFAULT_LOSS,
            confidence_head_alpha=1.0,
        )
        assert float(loss_conf) > float(loss_no_conf)

    def test_confidence_cumprod_bias_sign(self):
        # Draft != target so accept rate is ~0; an over-confident head (predicts
        # accept ~1) must show a positive cumulative-product calibration bias.
        # sample_from_anchor=False: position 0 is the anchor (masked).
        ids = torch.tensor([[0, 1, 0, 2]])
        logits = _ids_to_logits(ids, 8)
        targets = _ids_to_logits(torch.tensor([[0, 3, 0, 4]]), 8)
        loss_mask = torch.tensor([[0, 1, 0, 1]], dtype=torch.float32)
        confidence_logits = torch.full((1, 4), 20.0)  # sigmoid ~ 1.0
        _, metrics = compute_metrics(
            logits,
            targets,
            confidence_logits,
            loss_mask,
            block_size=2,
            loss_config=_DEFAULT_LOSS,
            sample_from_anchor=False,
        )
        bias = (
            metrics["confidence_cumprod_bias_sum"]
            / metrics["confidence_cumprod_bias_total"]
        )
        assert float(bias) > 0.5

    def test_alpha_weighting(self):
        ids = torch.tensor([[0, 1, 0, 2]])
        logits = _ids_to_logits(ids, 8)
        targets = _ids_to_logits(torch.tensor([[0, 3, 0, 4]]), 8)
        loss_mask = torch.tensor([[0, 1, 0, 1]], dtype=torch.float32)
        loss_small, _ = compute_metrics(
            logits,
            targets,
            None,
            loss_mask,
            block_size=2,
            loss_config=resolve_loss_config('{"tv": 0.1}', "eager"),
        )
        loss_large, _ = compute_metrics(
            logits,
            targets,
            None,
            loss_mask,
            block_size=2,
            loss_config=resolve_loss_config('{"tv": 1.0}', "eager"),
        )
        assert float(loss_large) > float(loss_small)

    def test_metric_keys_present(self):
        ids = torch.tensor([[0, 1, 0, 2]])
        logits = _ids_to_logits(ids, 8)
        targets = logits.clone()
        loss_mask = torch.tensor([[0, 1, 0, 1]], dtype=torch.float32)
        _, metrics = compute_metrics(
            logits,
            targets,
            torch.zeros(1, 4),
            loss_mask,
            block_size=2,
            loss_config=_DEFAULT_LOSS,
        )
        for key in (
            "loss_sum",
            "loss_total",
            "ce_loss_sum",
            "tv_loss_sum",
            "confidence_cumprod_bias_sum",
        ):
            assert key in metrics
        # all metric values must be tensors (so dist.reduce works in the trainer)
        assert all(torch.is_tensor(v) for v in metrics.values())
