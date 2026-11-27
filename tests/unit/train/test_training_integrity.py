"""CPU unit tests for training-run integrity guards."""

import json
import logging
import weakref
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest
import torch
import torch.distributed as dist
from torch import nn
from torch.distributed.fsdp import FSDPModule
from torch.utils.data import DataLoader

import speculators.train.trainer as trainer_module
from speculators.model import SpeculatorModel
from speculators.train.checkpointer import SingleGPUCheckpointer
from speculators.train.trainer import Trainer, TrainerConfig


class _TinyLinearSpeculator(nn.Module):
    def __init__(self, initial_weight: float = 1.0):
        super().__init__()
        self.proj = nn.Linear(1, 1, bias=False)
        with torch.no_grad():
            self.proj.weight.fill_(initial_weight)
        self.calls = 0

    def forward(
        self,
        target_scale: torch.Tensor,
        document_ids: torch.Tensor,
        error_records: int = 0,
        loss_override: str = "normal",
    ):
        del document_ids, error_records
        self.calls += 1
        pred = self.proj(torch.ones((1, 1)))
        if loss_override == "nan":
            loss = pred.sum() * float("nan")
        elif loss_override == "inf":
            loss = pred.sum() * float("inf")
        else:
            loss = ((pred - target_scale) ** 2).sum()
        return None, loss, {"loss": loss.detach()}


def _make_cpu_trainer(
    model: nn.Module,
    loader: object,
    tmp_path: Path,
    *,
    lr: float = 0.1,
    accum: int = 1,
    log_freq: int = 1,
    checkpoint_freq: float = 1,
    save_best: bool = False,
    max_steps: int | None = None,
) -> Trainer:
    trainer = Trainer.__new__(Trainer)
    trainer.model = cast("SpeculatorModel", model)
    trainer.config = TrainerConfig(
        lr=lr,
        num_epochs=1,
        save_path=str(tmp_path),
        gradient_accumulation_steps=accum,
        log_freq=log_freq,
        checkpoint_freq=checkpoint_freq,
        save_best=save_best,
        max_steps=max_steps,
        scheduler_type="none",
        hidden_states_dtype=torch.float32,
    )
    trainer.train_loader = cast("DataLoader", loader)
    trainer.val_loader = None
    trainer.is_distributed = False
    trainer.rank = 0
    trainer.local_rank = "cpu"  # type: ignore[assignment]
    trainer.device_type = "cpu"
    trainer.global_step = 0
    trainer.current_epoch = 0
    trainer.best_val_loss = float("inf")
    trainer.checkpointer = SingleGPUCheckpointer(str(tmp_path))
    trainer.optimizers = [torch.optim.SGD(model.parameters(), lr=lr)]
    trainer.schedulers = []
    return trainer


def _patch_cpu_timer(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.accelerator, "synchronize", lambda: None)
    monkeypatch.setattr(torch.accelerator, "is_available", lambda: False)


def test_nan_loss_batch_is_skipped_and_run_continues(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    _patch_cpu_timer(monkeypatch)
    logged: list[dict] = []
    monkeypatch.setattr(
        trainer_module.metric_logger,
        "info",
        lambda record, **kwargs: logged.append(record),
    )

    items = [
        {
            "target_scale": torch.tensor([[2.0]]),
            "document_ids": torch.tensor([0, 1]),
            "error_records": 0,
            "loss_override": "nan",
        },
        {
            "target_scale": torch.tensor([[2.0]]),
            "document_ids": torch.tensor([0, 1]),
            "error_records": 0,
            "loss_override": "normal",
        },
    ]

    class _Loader:
        batch_sampler = None

        def __len__(self) -> int:
            return len(items)

        def __iter__(self):
            return iter(items)

    model = _TinyLinearSpeculator(initial_weight=1.0)
    trainer = _make_cpu_trainer(model, _Loader(), tmp_path, lr=0.1)

    with caplog.at_level(logging.WARNING, logger="speculators"):
        trainer.train_epoch(0)

    assert trainer.global_step == 2
    assert torch.isfinite(model.proj.weight).all()
    assert model.proj.weight.item() == pytest.approx(1.1)
    assert any("non-finite loss" in rec.message.lower() for rec in caplog.records)
    assert len(logged) == 1
    assert logged[0]["global_step"] == 1
    assert "grad_norm" in logged[0]["train"]


def test_inf_loss_with_gradient_accumulation_clears_partial_grads(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _patch_cpu_timer(monkeypatch)
    logged: list[dict] = []
    monkeypatch.setattr(
        trainer_module.metric_logger,
        "info",
        lambda record, **kwargs: logged.append(record),
    )

    items = [
        {
            "target_scale": torch.tensor([[5.0]]),
            "document_ids": torch.tensor([0]),
            "error_records": 0,
            "loss_override": "normal",
        },
        {
            "target_scale": torch.tensor([[5.0]]),
            "document_ids": torch.tensor([0]),
            "error_records": 0,
            "loss_override": "inf",
        },
        {
            "target_scale": torch.tensor([[2.0]]),
            "document_ids": torch.tensor([0]),
            "error_records": 0,
            "loss_override": "normal",
        },
        {
            "target_scale": torch.tensor([[2.0]]),
            "document_ids": torch.tensor([0]),
            "error_records": 0,
            "loss_override": "normal",
        },
    ]

    class _Loader:
        batch_sampler = None

        def __len__(self) -> int:
            return len(items)

        def __iter__(self):
            return iter(items)

    model = _TinyLinearSpeculator(initial_weight=1.0)
    trainer = _make_cpu_trainer(model, _Loader(), tmp_path, lr=0.1, accum=2)
    trainer.train_epoch(0)

    assert trainer.global_step == 2
    assert torch.isfinite(model.proj.weight).all()
    # Clean window (weight=1, target=2): grad -2.0 clipped to -1.0 -> +0.1
    assert model.proj.weight.item() == pytest.approx(1.1)
    assert len(logged) == 1
    assert logged[0]["global_step"] == 1


def test_distributed_nonfinite_all_reduce_skips_when_remote_rank_flags(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _patch_cpu_timer(monkeypatch)

    class _Loader:
        batch_sampler = None

        def __len__(self) -> int:
            return 1

        def __iter__(self):
            return iter(
                [
                    {
                        "target_scale": torch.tensor([[3.0]]),
                        "document_ids": torch.tensor([0]),
                        "error_records": 0,
                        "loss_override": "normal",
                    }
                ]
            )

    model = _TinyLinearSpeculator(initial_weight=1.0)
    trainer = _make_cpu_trainer(model, _Loader(), tmp_path, lr=0.1)
    trainer.is_distributed = True

    def _remote_bad_all_reduce(tensor: torch.Tensor, op: object = None, **_kwargs):
        if op == dist.ReduceOp.MAX:
            tensor.fill_(1)

    monkeypatch.setattr(dist, "all_reduce", _remote_bad_all_reduce)
    trainer.train_epoch(0)

    assert trainer.global_step == 1
    assert model.proj.weight.item() == pytest.approx(1.0)


def test_grad_norm_logged_on_normal_steps_and_matches_pre_clip_norm(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _patch_cpu_timer(monkeypatch)
    logged: list[dict] = []
    monkeypatch.setattr(
        trainer_module.metric_logger,
        "info",
        lambda record, **kwargs: logged.append(record),
    )

    class _Loader:
        batch_sampler = None

        def __len__(self) -> int:
            return 1

        def __iter__(self):
            return iter(
                [
                    {
                        "target_scale": torch.tensor([[4.0]]),
                        "document_ids": torch.tensor([0]),
                        "error_records": 0,
                        "loss_override": "normal",
                    }
                ]
            )

    # weight=1.0, target=4.0 => loss=(1-4)^2=9 => pre-clip grad=2*(1-4)=-6 => norm=6.0
    model = _TinyLinearSpeculator(initial_weight=1.0)
    trainer = _make_cpu_trainer(model, _Loader(), tmp_path, lr=0.1)
    trainer.train_epoch(0)

    assert len(logged) == 1
    assert logged[0]["train"]["grad_norm"] == pytest.approx(6.0)
    # After clipping to 1.0 with SGD lr=0.1, weight moves by +0.1 (from 1.0 -> 1.1)
    assert model.proj.weight.item() == pytest.approx(1.1)


def test_finite_loss_with_nonfinite_gradient_raises_at_clip(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _patch_cpu_timer(monkeypatch)

    class _BadGradModel(_TinyLinearSpeculator):
        def forward(
            self,
            target_scale: torch.Tensor,
            document_ids: torch.Tensor,
            error_records: int = 0,
            loss_override: str = "normal",
        ):
            _, loss, metrics = super().forward(
                target_scale, document_ids, error_records, loss_override
            )
            loss.register_hook(lambda grad: grad * float("nan"))
            return None, loss, metrics

    class _Loader:
        batch_sampler = None

        def __len__(self) -> int:
            return 1

        def __iter__(self):
            return iter(
                [
                    {
                        "target_scale": torch.tensor([[2.0]]),
                        "document_ids": torch.tensor([0]),
                        "error_records": 0,
                    }
                ]
            )

    model = _BadGradModel(initial_weight=1.0)
    trainer = _make_cpu_trainer(model, _Loader(), tmp_path, lr=0.1)

    with pytest.raises(RuntimeError, match="non-finite"):
        trainer.train_epoch(0)

    assert model.proj.weight.item() == pytest.approx(1.0)


def _make_checkpoint_test_trainer(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *,
    num_epochs: int,
    checkpoint_freq: int,
    save_best: bool,
) -> tuple[Trainer, list[int]]:
    trainer = Trainer.__new__(Trainer)
    trainer.config = TrainerConfig(
        lr=1e-3,
        num_epochs=num_epochs,
        save_path=str(tmp_path),
        resume_from_checkpoint=False,
        checkpoint_freq=checkpoint_freq,
        save_best=save_best,
    )
    trainer.best_val_loss = float("inf")
    trainer.current_epoch = 0
    trainer.global_step = 42
    trainer.is_distributed = False
    trainer.rank = 0
    trainer.local_rank = 0
    trainer.resume_from_checkpoint = False
    trainer.train_loader = cast("DataLoader", [])
    trainer.val_loader = None
    trainer.checkpointer = SingleGPUCheckpointer(str(tmp_path))
    trainer.model = cast("SpeculatorModel", nn.Identity())
    trainer.optimizers = cast("list[torch.optim.Optimizer]", [object()])
    trainer.schedulers = []

    saved_epochs: list[int] = []

    def _fake_save_checkpoint(_model, _opt, epoch: int):
        saved_epochs.append(epoch)
        (tmp_path / str(epoch)).mkdir(exist_ok=True)

    monkeypatch.setattr(trainer, "train_epoch", lambda *_a, **_kw: None)
    monkeypatch.setattr(trainer.checkpointer, "save_checkpoint", _fake_save_checkpoint)
    monkeypatch.setattr(
        trainer.checkpointer,
        "save_scheduler_state_dict",
        lambda *_a, **_kw: None,
    )
    return trainer, saved_epochs


def test_final_checkpoint_forced_when_checkpoint_freq_skips_last_epoch(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    trainer, saved_epochs = _make_checkpoint_test_trainer(
        monkeypatch, tmp_path, num_epochs=7, checkpoint_freq=3, save_best=False
    )
    trainer.run_training()

    assert saved_epochs == [0, 2, 5, 6]
    state_file = tmp_path / "6" / "training_state.json"
    assert state_file.exists()
    state = json.loads(state_file.read_text())
    assert state["epoch"] == 6
    assert state["local_step"] == 0
    assert state["global_step"] == 42
    link = tmp_path / "epoch6_end"
    assert link.is_symlink()
    assert link.resolve() == (tmp_path / "6").resolve()


def test_final_checkpoint_not_duplicated_when_freq_covers_last_epoch(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    trainer, saved_epochs = _make_checkpoint_test_trainer(
        monkeypatch, tmp_path, num_epochs=6, checkpoint_freq=3, save_best=False
    )
    trainer.run_training()

    assert saved_epochs == [0, 2, 5]


def test_final_checkpoint_not_forced_when_save_best_enabled(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    trainer, saved_epochs = _make_checkpoint_test_trainer(
        monkeypatch, tmp_path, num_epochs=4, checkpoint_freq=3, save_best=True
    )
    trainer.run_training()

    assert saved_epochs == []


def test_forced_final_checkpoint_eligible_for_checkpoint_best(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    trainer, saved_epochs = _make_checkpoint_test_trainer(
        monkeypatch, tmp_path, num_epochs=4, checkpoint_freq=3, save_best=False
    )
    val_losses = [0.9, 0.8, 0.85, 0.7]
    trainer.val_loader = cast("DataLoader", [object()])
    monkeypatch.setattr(
        trainer,
        "val_epoch",
        lambda epoch: {"loss_epoch": val_losses[epoch]},
    )

    trainer.run_training()

    assert saved_epochs == [0, 2, 3]
    best_link = tmp_path / "checkpoint_best"
    assert best_link.is_symlink()
    assert best_link.resolve() == (tmp_path / "3").resolve()
    val_metrics = json.loads((tmp_path / "3" / "val_metrics.json").read_text())
    assert val_metrics["loss_epoch"] == pytest.approx(0.7)


def test_mid_window_nonfinite_skips_remaining_microbatch_and_releases_outputs(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _patch_cpu_timer(monkeypatch)
    logged: list[dict] = []
    monkeypatch.setattr(
        trainer_module.metric_logger,
        "info",
        lambda record, **kwargs: logged.append(record),
    )

    class _Sentinel:
        pass

    rejected_ref: list[weakref.ref[_Sentinel]] = []
    alive_at_next_forward: list[bool] = []

    class _TrackingSpeculator(_TinyLinearSpeculator):
        def forward(
            self,
            target_scale: torch.Tensor,
            document_ids: torch.Tensor,
            error_records: int = 0,
            loss_override: str = "normal",
        ):
            if self.calls == 2 and rejected_ref:
                alive_at_next_forward.append(rejected_ref[0]() is not None)
            _, loss, metrics = super().forward(
                target_scale, document_ids, error_records, loss_override
            )
            token = _Sentinel()
            if loss_override == "nan":
                rejected_ref.append(weakref.ref(token))
            return token, loss, metrics

    items = [
        {
            "target_scale": torch.tensor([[5.0]]),
            "document_ids": torch.tensor([0]),
            "error_records": 0,
            "loss_override": "normal",
        },
        {
            "target_scale": torch.tensor([[5.0]]),
            "document_ids": torch.tensor([0]),
            "error_records": 0,
            "loss_override": "nan",
        },
        {
            "target_scale": torch.tensor([[9.0]]),
            "document_ids": torch.tensor([0]),
            "error_records": 0,
            "loss_override": "normal",
        },
        {
            "target_scale": torch.tensor([[2.0]]),
            "document_ids": torch.tensor([0]),
            "error_records": 0,
            "loss_override": "normal",
        },
        {
            "target_scale": torch.tensor([[2.0]]),
            "document_ids": torch.tensor([0]),
            "error_records": 0,
            "loss_override": "normal",
        },
        {
            "target_scale": torch.tensor([[2.0]]),
            "document_ids": torch.tensor([0]),
            "error_records": 0,
            "loss_override": "normal",
        },
    ]

    class _Loader:
        batch_sampler = None

        def __len__(self) -> int:
            return len(items)

        def __iter__(self):
            return iter(items)

    model = _TrackingSpeculator(initial_weight=1.0)
    trainer = _make_cpu_trainer(model, _Loader(), tmp_path, lr=0.1, accum=3)
    trainer.train_epoch(0)

    # Microbatch 3 in window 1 must be skipped without calling model.forward.
    assert model.calls == 5
    assert trainer.global_step == 2
    assert alive_at_next_forward == [False]
    assert model.proj.weight.item() == pytest.approx(1.1)
    assert len(logged) == 1
    assert logged[0]["global_step"] == 1


def test_discard_window_gradients_clears_fsdp_state(tmp_path: Path) -> None:
    class _FakeFSDPWithReset(FSDPModule, nn.Linear):
        def __new__(cls, *args, **kwargs):
            return nn.Module.__new__(cls)

        def __init__(self) -> None:
            super().__init__(1, 1, bias=False)
            self.reset_calls = 0

        def reset_iter_state(self) -> None:
            self.reset_calls += 1

    model_reset = _FakeFSDPWithReset()
    trainer_reset = _make_cpu_trainer(model_reset, [], tmp_path)
    trainer_reset._discard_window_gradients()
    assert model_reset.reset_calls == 1

    class _FakeFSDPLegacy(FSDPModule, nn.Linear):
        def __new__(cls, *args, **kwargs):
            return nn.Module.__new__(cls)

        def __init__(self) -> None:
            super().__init__(1, 1, bias=False)
            self.fake_param = SimpleNamespace(
                unsharded_accumulated_grad=torch.ones(1),
                _unsharded_param=SimpleNamespace(grad=torch.ones(1)),
                unsharded_param=None,
            )
            self.fake_param.unsharded_param = self.fake_param._unsharded_param

        def reset_iter_state(self) -> None:
            raise AssertionError("should not be called when absent")

        def _get_fsdp_state(self):
            group = SimpleNamespace(fsdp_params=[self.fake_param])
            return SimpleNamespace(_fsdp_param_groups=[group])

    model_legacy = _FakeFSDPLegacy()
    # Hide reset_iter_state on this instance's class so fallback path runs.
    model_legacy.__dict__["reset_iter_state"] = None
    trainer_legacy = _make_cpu_trainer(model_legacy, [], tmp_path)
    trainer_legacy._discard_window_gradients()
    assert model_legacy.fake_param.unsharded_accumulated_grad is None
    assert model_legacy.fake_param.unsharded_param.grad is None
