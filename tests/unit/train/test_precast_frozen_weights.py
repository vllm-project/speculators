"""Frozen linear weights are stored in the autocast dtype."""

from types import SimpleNamespace
from typing import cast
from unittest.mock import Mock

import torch
from torch import nn

from speculators.train import trainer as trainer_module
from speculators.train.trainer import Trainer


class _Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.trainable = nn.Linear(4, 4)
        self.frozen = nn.Linear(4, 8, bias=False)
        self.frozen_with_bias = nn.Linear(4, 4)
        self.frozen_embedding = nn.Embedding(4, 4)
        self.frozen.requires_grad_(False)
        self.frozen_with_bias.requires_grad_(False)
        self.frozen_embedding.requires_grad_(False)


def _precast(
    model: nn.Module,
    dtype: torch.dtype = torch.bfloat16,
    *,
    is_distributed: bool = False,
    fsdp_shard: bool = False,
) -> None:
    stand_in = cast(
        "Trainer",
        SimpleNamespace(
            model=model,
            is_distributed=is_distributed,
            config=SimpleNamespace(
                hidden_states_dtype=dtype,
                fsdp_shard=fsdp_shard,
            ),
        ),
    )
    Trainer._precast_frozen_linear_weights(stand_in)


def test_frozen_linear_weights_are_cast():
    model = _Model()

    _precast(model)

    assert model.frozen.weight.dtype == torch.bfloat16
    assert model.frozen_with_bias.weight.dtype == torch.bfloat16
    assert model.frozen_with_bias.bias.dtype == torch.bfloat16


def test_trainable_weights_keep_their_fp32_master_copy():
    """The optimizer updates this tensor in place."""
    model = _Model()

    _precast(model)

    assert model.trainable.weight.dtype == torch.float32
    assert model.trainable.bias.dtype == torch.float32


def test_only_linear_modules_are_touched():
    """Autocast does not cast embeddings either."""
    model = _Model()

    _precast(model)

    assert model.frozen_embedding.weight.dtype == torch.float32


def test_values_survive_the_cast():
    model = _Model()
    before = model.frozen.weight.detach().clone()

    _precast(model)

    assert torch.equal(model.frozen.weight.float(), before.bfloat16().float())
    assert model.frozen.weight.requires_grad is False


def test_an_fp32_run_leaves_everything_alone():
    model = _Model()

    _precast(model, dtype=torch.float32)

    assert all(p.dtype == torch.float32 for p in model.parameters())


def test_a_sharded_run_is_left_alone():
    """FSDP2 needs one original dtype per parameter group."""
    model = _Model()

    _precast(model, is_distributed=True, fsdp_shard=True)

    assert all(p.dtype == torch.float32 for p in model.parameters())


def test_a_single_process_run_still_casts_with_fsdp_shard_set():
    """fsdp_shard only takes effect once there is more than one process."""
    model = _Model()

    _precast(model, is_distributed=False, fsdp_shard=True)

    assert model.frozen.weight.dtype == torch.bfloat16


def test_a_precast_model_no_longer_reports_one_dtype():
    """Why the loads below cannot fall back to `model.dtype`: it returns the
    first floating parameter's dtype, and that may now be a cast frozen one."""
    model = _Model()

    _precast(model)

    assert {p.dtype for p in model.parameters()} == {torch.float32, torch.bfloat16}


def test_resume_loads_the_model_as_fp32(monkeypatch):
    """A bf16 load would round the fp32 master weights on resume."""
    monkeypatch.setattr(
        trainer_module.SpeculatorModel, "verify_training_compatible", lambda _m: None
    )
    model = _Model()
    checkpointer = Mock(previous_epoch=0)
    stand_in = cast(
        "Trainer",
        SimpleNamespace(
            model=model,
            is_distributed=False,
            local_rank="cpu",
            resume_from_checkpoint=True,
            checkpointer=checkpointer,
            config=SimpleNamespace(gradient_checkpointing=False),
            _precast_frozen_linear_weights=lambda: None,
        ),
    )

    Trainer.setup_model(stand_in)

    checkpointer.load_model_state_dict.assert_called_once_with(
        model, float_dtype=torch.float32
    )


def test_resume_loads_the_optimizer_state_as_fp32(monkeypatch):
    """Momentum buffers and Adam moments are fp32 too."""
    monkeypatch.setattr(trainer_module, "build_optimizers", lambda _m, _c: [])
    model = _Model()
    checkpointer = Mock(previous_epoch=0)
    stand_in = cast(
        "Trainer",
        SimpleNamespace(
            model=model,
            resume_from_checkpoint=True,
            checkpointer=checkpointer,
            config=SimpleNamespace(scheduler_type="none"),
        ),
    )

    Trainer.setup_optimizer(stand_in)

    checkpointer.load_optimizer_state_dict.assert_called_once_with(
        model, [], float_dtype=torch.float32
    )
