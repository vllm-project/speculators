"""Device-to-host readback of the per-step metric scalars."""

from unittest.mock import patch

import torch

from speculators.train.trainer import _metrics_to_host


def _metrics() -> dict[str, torch.Tensor]:
    return {
        "loss": torch.tensor(1.25),
        "loss_total": torch.tensor(1.0),
        "acc_0": torch.tensor(0.1),
        "count": torch.tensor(7, dtype=torch.int64),
        "halved": torch.tensor([0.5]),
    }


def test_matches_per_entry_item():
    metrics = _metrics()

    assert _metrics_to_host(metrics) == {
        k: v.float().reshape(()).item() for k, v in metrics.items()
    }


def test_preserves_key_order():
    metrics = _metrics()

    assert list(_metrics_to_host(metrics)) == list(metrics)


def test_empty_metrics_read_back_as_empty():
    assert _metrics_to_host({}) == {}


def test_reads_back_without_a_sync_per_entry():
    """One transfer for the whole dict, not one per metric.

    ``.item()`` blocks on the device each time it is called, and a logged step
    carries a few dozen metrics; this guards the regression back to a
    per-entry comprehension.
    """
    with patch.object(
        torch.Tensor, "item", side_effect=AssertionError("per-entry sync")
    ):
        assert _metrics_to_host(_metrics())["loss"] == 1.25
