"""Validation must not average in batches whose samples all failed generation."""

import inspect
from collections.abc import Callable
from functools import partial
from types import SimpleNamespace

import pytest
import torch

from speculators.losses import resolve_loss_config
from speculators.models.dspark.metrics import compute_metrics
from speculators.train import trainer as trainer_module
from speculators.train.recovery import RECOVERY_METADATA_KEY, RecoveryMetadata
from speculators.train.trainer import Trainer

SEQ_LEN, VOCAB = 8, 16

_metrics_fn = partial(
    compute_metrics,
    block_size=2,
    loss_config=resolve_loss_config('{"ce": 0.1, "tv": 0.9}', "eager"),
)
torch.manual_seed(0)
_LOGITS = torch.randn(1, SEQ_LEN, VOCAB)
_TARGETS = torch.randn(1, SEQ_LEN, VOCAB)
_CONFIDENCE = torch.randn(1, SEQ_LEN)


class _Model:
    def __init__(self, forward: Callable):
        self.forward = forward

    def __call__(self, loss_mask, **_kwargs):
        return self.forward(loss_mask)

    def eval(self):
        pass


def _dspark_forward(loss_mask):
    # Same draft outputs for every batch; only the loss mask differs.
    loss, metrics = _metrics_fn(_LOGITS, _TARGETS, _CONFIDENCE, loss_mask.float())
    return None, loss, metrics


_model = _Model(_dspark_forward)


class _Loader(list):
    batch_sampler = None


def _real_batch():
    return {"loss_mask": torch.ones(1, SEQ_LEN, dtype=torch.bool)}


def _empty_batch():
    # What collate returns when every sample in the batch failed generation.
    return {
        "loss_mask": torch.zeros(1, SEQ_LEN, dtype=torch.bool),
        RECOVERY_METADATA_KEY: RecoveryMetadata(failure_count=1, locally_empty=True),
    }


def _val_epoch(batches, model=_model, is_distributed=False):
    trainer = SimpleNamespace(
        val_loader=_Loader(batches),
        model=model,
        rank=1,  # skip the rank-0 progress bar
        local_rank="cpu",
        device_type="cpu",
        is_distributed=is_distributed,
        global_step=0,
        config=SimpleNamespace(
            hidden_states_dtype=torch.bfloat16, val_call_kwargs=None
        ),
        _maybe_val_sync=lambda _index: None,
    )
    # Unwrap torch.no_grad; gradients are irrelevant here.
    return inspect.unwrap(Trainer.val_epoch)(trainer, 0)


def test_empty_batches_do_not_lower_validation_loss():
    real_only = _val_epoch([_real_batch() for _ in range(4)])
    with_empty = _val_epoch(
        [_real_batch() for _ in range(4)] + [_empty_batch() for _ in range(4)]
    )

    assert real_only["loss_epoch"] > 0
    assert with_empty["loss_epoch"] == pytest.approx(real_only["loss_epoch"])


def test_validation_with_no_data_returns_none():
    assert _val_epoch([_empty_batch() for _ in range(4)]) is None


def test_unpaired_metric_is_averaged_over_non_empty_batches():
    # MTP reports loss_step_{step} without a _sum/_total pair.
    def forward(loss_mask):
        value = torch.tensor(2.0 if loss_mask.any() else 0.0)
        return None, value, {"loss_step_0": value}

    metrics = _val_epoch([_real_batch(), _empty_batch()], model=_Model(forward))

    assert metrics["loss_step_0_epoch"] == pytest.approx(2.0)


def test_no_data_on_every_rank_returns_none_on_each(monkeypatch):
    # The decision uses the reduced count, so every rank returns the same thing
    # and nobody skips a collective. Simulate two ranks with no data each.
    def all_reduce(tensor, op):
        tensor.mul_(2)

    monkeypatch.setattr(trainer_module.dist, "all_reduce", all_reduce)
    monkeypatch.setattr(trainer_module.dist, "get_world_size", lambda: 2)

    assert _val_epoch([_empty_batch()], is_distributed=True) is None


def test_other_rank_data_is_counted_when_this_rank_is_empty(monkeypatch):
    # This rank has only empty batches; the other rank contributed one real
    # batch, so the reduced count is 1 and metrics are returned.
    real_metrics = _dspark_forward(_real_batch()["loss_mask"])[2]

    def all_reduce(tensor, op):
        # Add the other rank's contribution: its real-batch metrics, count 1.
        keys = [*real_metrics, trainer_module._COUNTED_BATCHES_KEY]
        other = torch.stack(
            [real_metrics[k].float().reshape(()) for k in keys[:-1]]
            + [torch.tensor(1.0)]
        )
        tensor.add_(other)

    monkeypatch.setattr(trainer_module.dist, "all_reduce", all_reduce)
    monkeypatch.setattr(trainer_module.dist, "get_world_size", lambda: 2)

    metrics = _val_epoch([_empty_batch()], is_distributed=True)
    expected = _val_epoch([_real_batch()])

    assert metrics["loss_epoch"] == pytest.approx(expected["loss_epoch"])
