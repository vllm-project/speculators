"""Focused optimizer execution-backend and fused clipping tests."""

from types import SimpleNamespace

import pytest
import torch

from speculators.train.optimizers import build_optimizers, restore_adamw_backend
from speculators.train.trainer import Trainer, TrainerConfig


def _optimizer_config(**overrides):
    values = {
        "optimizer": "adamw",
        "adamw_backend": "auto",
        "lr": 1e-3,
        "weight_decay": 0.01,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_foreach_adamw_backend_is_explicit():
    model = torch.nn.Linear(4, 3)
    (optimizer,) = build_optimizers(model, _optimizer_config(adamw_backend="foreach"))

    assert optimizer.param_groups[0]["foreach"] is True
    assert optimizer.param_groups[0]["fused"] is False


def test_fused_adamw_rejects_cpu_parameters():
    with pytest.raises(ValueError, match="CUDA parameters"):
        build_optimizers(
            torch.nn.Linear(4, 3), _optimizer_config(adamw_backend="fused")
        )


@pytest.mark.parametrize(
    ("backend", "expected_foreach", "expected_fused"),
    [
        ("auto", None, None),
        ("foreach", True, False),
        ("fused", False, True),
    ],
)
def test_restore_adamw_backend_overrides_checkpoint_param_groups(
    backend, expected_foreach, expected_fused
):
    parameter = torch.nn.Parameter(torch.ones(2))
    checkpoint_optimizer = torch.optim.AdamW([parameter], foreach=True, fused=False)
    resumed_optimizer = torch.optim.AdamW([parameter])
    resumed_optimizer.load_state_dict(checkpoint_optimizer.state_dict())

    assert resumed_optimizer.param_groups[0]["foreach"] is True
    assert resumed_optimizer.param_groups[0]["fused"] is False

    restore_adamw_backend([resumed_optimizer], backend)

    assert resumed_optimizer.param_groups[0]["foreach"] is expected_foreach
    assert resumed_optimizer.param_groups[0]["fused"] is expected_fused


def test_trainer_restores_requested_backend_after_checkpoint_load():
    class _Checkpoint:
        previous_epoch = 3

        @staticmethod
        def load_optimizer_state_dict(model, optimizers):
            del model
            optimizers[0].param_groups[0]["foreach"] = None
            optimizers[0].param_groups[0]["fused"] = None

    trainer = Trainer.__new__(Trainer)
    trainer.model = torch.nn.Linear(4, 3)
    trainer.config = TrainerConfig(
        lr=1e-3,
        num_epochs=1,
        save_path="unused",
        resume_from_checkpoint=True,
        adamw_backend="foreach",
        scheduler_type="none",
    )
    trainer.resume_from_checkpoint = True
    trainer.checkpointer = _Checkpoint()

    trainer.setup_optimizer()

    assert trainer.optimizers[0].param_groups[0]["foreach"] is True
    assert trainer.optimizers[0].param_groups[0]["fused"] is False


class _RecordingFusedOptimizer:
    def __init__(self, fail: bool = False):
        self.param_groups = [{"foreach": False, "fused": True}]
        self.fail = fail
        self.saw_grad_scale = False

    def step(self):
        self.saw_grad_scale = hasattr(self, "grad_scale")
        if self.fail:
            raise RuntimeError("step failed")


def _fused_clip_trainer(optimizer) -> Trainer:
    trainer = Trainer.__new__(Trainer)
    trainer.model = torch.nn.Linear(3, 2, bias=False)
    for parameter in trainer.model.parameters():
        parameter.grad = torch.full_like(parameter, 2.0)
    trainer.config = TrainerConfig(
        lr=1e-3,
        num_epochs=1,
        save_path="unused",
        optimizer="adamw",
        adamw_backend="fused",
        gradient_clip_backend="fused_adamw",
        max_grad_norm=0.5,
    )
    trainer.optimizers = [optimizer]
    trainer.local_rank = torch.device("cpu")
    return trainer


def test_fused_clip_sets_scale_and_cleans_it_after_step():
    optimizer = _RecordingFusedOptimizer()
    trainer = _fused_clip_trainer(optimizer)

    norm = trainer._clip_gradients()
    expected_scale = ((norm + 1e-6) / trainer.config.max_grad_norm).clamp(min=1)
    torch.testing.assert_close(optimizer.grad_scale, expected_scale)
    trainer._optimizers_step()

    assert optimizer.saw_grad_scale
    assert not hasattr(optimizer, "grad_scale")


def test_fused_clip_cleans_scale_when_optimizer_step_fails():
    optimizer = _RecordingFusedOptimizer(fail=True)
    trainer = _fused_clip_trainer(optimizer)
    trainer._clip_gradients()

    with pytest.raises(RuntimeError, match="step failed"):
        trainer._optimizers_step()

    assert not hasattr(optimizer, "grad_scale")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_fused_adamw_clip_matches_explicit_clipping_on_cuda():
    initial = torch.randn(19, device="cuda")
    reference = torch.nn.Parameter(initial.clone())
    fused = torch.nn.Parameter(initial.clone())
    gradient = torch.linspace(-3, 4, initial.numel(), device="cuda")
    reference.grad = gradient.clone()
    fused.grad = gradient.clone()
    reference_optimizer = torch.optim.AdamW([reference], lr=3e-4, fused=True)
    fused_optimizer = torch.optim.AdamW([fused], lr=3e-4, fused=True)

    torch.nn.utils.clip_grad_norm_([reference], 0.7)
    reference_optimizer.step()
    grad_norm = torch.nn.utils.get_total_norm([fused.grad], foreach=True)
    fused_optimizer.grad_scale = ((grad_norm + 1e-6) / 0.7).clamp(min=1)
    try:
        fused_optimizer.step()
    finally:
        del fused_optimizer.grad_scale

    torch.testing.assert_close(fused, reference, rtol=1e-6, atol=1e-7)
    torch.testing.assert_close(
        fused_optimizer.state[fused]["exp_avg"],
        reference_optimizer.state[reference]["exp_avg"],
    )
    torch.testing.assert_close(
        fused_optimizer.state[fused]["exp_avg_sq"],
        reference_optimizer.state[reference]["exp_avg_sq"],
    )
