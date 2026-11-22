"""val_epoch must not average padding-only (all-failed) batches into the metrics."""

from functools import partial
from types import SimpleNamespace

import pytest
import torch

from speculators.losses import resolve_loss_config
from speculators.models.dspark.metrics import compute_metrics
from speculators.train.trainer import Trainer

torch.manual_seed(0)
_METRICS_FN = partial(
    compute_metrics,
    block_size=2,
    loss_config=resolve_loss_config('{"ce": 0.1, "tv": 0.9}', "eager"),
)
_LOGITS = torch.randn(1, 8, 16)
_TARGETS = torch.randn(1, 8, 16)
_CONF = torch.randn(1, 8)


class _Model:
    def __call__(self, loss_mask, **_):
        # Same draft outputs for every batch; only the loss mask differs.
        loss, metrics = _METRICS_FN(_LOGITS, _TARGETS, _CONF, loss_mask.float())
        return None, loss, metrics

    def eval(self):
        pass


class _Loader(list):
    batch_sampler = None


def _real():
    return {"loss_mask": torch.ones(1, 8, dtype=torch.bool)}


def _empty():
    # What collate builds when every sample in the batch failed.
    return {"loss_mask": torch.zeros(1, 8, dtype=torch.bool)}


def _val_metrics(batches):
    trainer = SimpleNamespace(
        val_loader=_Loader(batches),
        model=_Model(),
        rank=1,
        local_rank="cpu",
        device_type="cpu",
        is_distributed=False,
        global_step=0,
        config=SimpleNamespace(
            hidden_states_dtype=torch.bfloat16, val_call_kwargs=None
        ),
        _maybe_val_sync=lambda i: None,
    )
    return Trainer.val_epoch(trainer, epoch=0)


def test_empty_batches_do_not_change_loss_epoch():
    clean = _val_metrics([_real(), _real()])
    mixed = _val_metrics([_real(), _empty(), _real(), _empty(), _empty()])
    assert clean["loss_epoch"] > 0
    assert mixed.keys() == clean.keys()
    for k in clean:
        assert mixed[k] == pytest.approx(clean[k]), k


def test_all_empty_epoch_reports_no_loss():
    metrics = _val_metrics([_empty(), _empty()])
    assert "loss_epoch" not in metrics
