"""Tests for fp32 master weight training.

Covers:
- FP32MasterOptimizer accumulates sub-bf16-ulp updates that pure-bf16 loses
- param_groups sharing with the inner optimizer (LR scheduler compatibility)
- zero_grad clears model param gradients
- state_dict round-trip preserves fp32 masters bit-exactly
- legacy (bare-optimizer) state dict loading
- build_optimizers wrapping for adamw and muon modes
- Trainer._fp32_master_mode gating
- SingleGPUCheckpointer round-trip keeps optimizer state in fp32
"""

from pathlib import Path

import pytest
import torch
from torch import nn
from transformers import PretrainedConfig, PreTrainedModel
from transformers.optimization import get_linear_schedule_with_warmup

from speculators.train.checkpointer import SingleGPUCheckpointer
from speculators.train.optimizers import FP32MasterOptimizer, build_optimizers
from speculators.train.trainer import Trainer, TrainerConfig

# bf16 has 8 mantissa bits: the spacing between representable values at 1.0.
BF16_ULP_AT_ONE = 2**-8


def make_config(**overrides) -> TrainerConfig:
    return TrainerConfig(
        lr=1e-4,
        num_epochs=1,
        save_path="checkpoint",
        **overrides,
    )


def adamw_factory(named_params):
    # weight_decay=0 keeps the constant-gradient update math exact: for a
    # constant gradient, bias-corrected Adam moves each weight by exactly lr.
    return torch.optim.AdamW(named_params, lr=1e-5, weight_decay=0.0)


def make_bf16_linear(out_features: int = 4) -> nn.Linear:
    torch.manual_seed(0)
    linear = nn.Linear(4, out_features, bias=False).to(torch.bfloat16)
    with torch.no_grad():
        linear.weight.fill_(1.0)
    return linear


def run_constant_grad_steps(model: nn.Module, opt, num_steps: int):
    for _ in range(num_steps):
        opt.zero_grad()
        for param in model.parameters():
            param.grad = torch.ones_like(param)
        opt.step()


def test_fp32_masters_accumulate_sub_ulp_updates():
    # Each AdamW step moves the weights by lr=1e-5, far below the bf16 ulp at
    # 1.0 (~3.9e-3): a bare bf16 optimizer rounds every update away.
    num_steps = 300
    expected = 1.0 - num_steps * 1e-5  # 0.997

    bare_model = make_bf16_linear()
    bare_opt = adamw_factory(list(bare_model.named_parameters()))
    run_constant_grad_steps(bare_model, bare_opt, num_steps)
    assert torch.all(bare_model.weight == 1.0), (
        "expected pure-bf16 AdamW to lose all sub-ulp updates"
    )

    master_model = make_bf16_linear()
    master_opt = FP32MasterOptimizer(
        list(master_model.named_parameters()), adamw_factory
    )
    run_constant_grad_steps(master_model, master_opt, num_steps)

    masters = master_opt.state_dict()["fp32_masters"]
    assert all(m.dtype == torch.float32 for m in masters)
    assert torch.allclose(masters[0], torch.full_like(masters[0], expected), atol=1e-4)
    # The bf16 params track the rounded masters and have visibly moved.
    assert torch.all(
        master_model.weight == torch.tensor(expected, dtype=torch.bfloat16)
    )


def test_wrapper_shares_param_groups_with_inner_for_schedulers():
    model = make_bf16_linear()
    opt = FP32MasterOptimizer(list(model.named_parameters()), adamw_factory)

    assert isinstance(opt, torch.optim.Optimizer)
    assert opt.param_groups is opt.inner.param_groups

    scheduler = get_linear_schedule_with_warmup(
        opt, num_warmup_steps=2, num_training_steps=10
    )
    # Warmup step 0: lr is scaled to 0; the inner optimizer must see it too.
    assert opt.inner.param_groups[0]["lr"] == 0.0
    run_constant_grad_steps(model, opt, 1)
    scheduler.step()
    assert opt.inner.param_groups[0]["lr"] == pytest.approx(1e-5 / 2)


def test_zero_grad_clears_model_param_grads():
    model = make_bf16_linear()
    opt = FP32MasterOptimizer(list(model.named_parameters()), adamw_factory)

    model.weight.grad = torch.ones_like(model.weight)
    opt.zero_grad()
    assert model.weight.grad is None


def test_skips_frozen_params_and_requires_a_trainable_one():
    model = nn.ModuleDict(
        {"a": nn.Linear(4, 4, bias=False), "b": nn.Linear(4, 4, bias=False)}
    ).to(torch.bfloat16)
    model["b"].weight.requires_grad_(False)

    opt = FP32MasterOptimizer(list(model.named_parameters()), adamw_factory)
    assert len(opt.state_dict()["fp32_masters"]) == 1

    model["a"].weight.requires_grad_(False)
    with pytest.raises(ValueError, match="No trainable parameters"):
        FP32MasterOptimizer(list(model.named_parameters()), adamw_factory)


def test_state_dict_round_trip_preserves_masters_bit_exactly(tmp_path: Path):
    model = make_bf16_linear()
    opt = FP32MasterOptimizer(list(model.named_parameters()), adamw_factory)
    run_constant_grad_steps(model, opt, 17)

    payload_path = tmp_path / "opt.pt"
    torch.save(opt.state_dict(), payload_path)
    loaded = torch.load(payload_path, weights_only=True)

    fresh_model = make_bf16_linear()
    fresh_opt = FP32MasterOptimizer(list(fresh_model.named_parameters()), adamw_factory)
    fresh_opt.load_state_dict(loaded)

    original_masters = opt.state_dict()["fp32_masters"]
    restored_masters = fresh_opt.state_dict()["fp32_masters"]
    for original, restored in zip(original_masters, restored_masters, strict=True):
        assert restored.dtype == torch.float32
        assert torch.equal(original, restored)
    # Model params are re-synced from the restored masters.
    assert torch.equal(fresh_model.weight, model.weight)
    # Inner AdamW moments are fp32 and restored.
    exp_avg = next(iter(fresh_opt.inner.state.values()))["exp_avg"]
    assert exp_avg.dtype == torch.float32
    # param_groups stay shared after load_state_dict rebinds them.
    assert fresh_opt.param_groups is fresh_opt.inner.param_groups


def test_legacy_bare_optimizer_state_dict_loads():
    model = make_bf16_linear()
    bare_opt = adamw_factory(list(model.named_parameters()))
    run_constant_grad_steps(model, bare_opt, 3)
    legacy_state = bare_opt.state_dict()

    fresh_model = make_bf16_linear()
    wrapper = FP32MasterOptimizer(list(fresh_model.named_parameters()), adamw_factory)
    wrapper.load_state_dict(legacy_state)

    # torch casts the legacy (bf16) moments up to the masters' fp32.
    exp_avg = next(iter(wrapper.inner.state.values()))["exp_avg"]
    assert exp_avg.dtype == torch.float32


def test_legacy_state_dict_mismatch_raises_actionable_error():
    model = nn.ModuleDict(
        {"a": nn.Linear(4, 4, bias=False), "b": nn.Linear(4, 4, bias=False)}
    ).to(torch.bfloat16)
    bare_opt = adamw_factory(list(model.named_parameters()))
    legacy_state = bare_opt.state_dict()

    model["b"].weight.requires_grad_(False)  # wrapper keeps only 1 of 2 params
    wrapper = FP32MasterOptimizer(list(model.named_parameters()), adamw_factory)
    with pytest.raises(ValueError, match="optimizer_state_dict.pt"):
        wrapper.load_state_dict(legacy_state)


def test_build_optimizers_wraps_adamw():
    model = make_bf16_linear()
    optimizers = build_optimizers(
        model, make_config(optimizer="adamw"), fp32_masters=True
    )
    assert len(optimizers) == 1
    assert isinstance(optimizers[0], FP32MasterOptimizer)
    assert isinstance(optimizers[0].inner, torch.optim.AdamW)


@pytest.mark.skipif(
    not hasattr(torch.optim, "Muon"), reason="torch.optim.Muon not available"
)
def test_build_optimizers_wraps_muon_and_adamw_groups():
    model = nn.ModuleDict(
        {
            "matrix": nn.Linear(4, 4, bias=True),  # 2D weight -> Muon, bias -> AdamW
            "embed_tokens": nn.Embedding(8, 4),  # 2D but excluded from Muon
        }
    ).to(torch.bfloat16)
    optimizers = build_optimizers(
        model, make_config(optimizer="muon", muon_lr=1e-3), fp32_masters=True
    )

    assert len(optimizers) == 2
    muon_opt, adamw_opt = optimizers
    assert isinstance(muon_opt, FP32MasterOptimizer)
    assert isinstance(muon_opt.inner, torch.optim.Muon)
    assert len(muon_opt.state_dict()["fp32_masters"]) == 1
    assert isinstance(adamw_opt, FP32MasterOptimizer)
    assert isinstance(adamw_opt.inner, torch.optim.AdamW)
    assert len(adamw_opt.state_dict()["fp32_masters"]) == 2  # bias + embedding


@pytest.mark.parametrize(
    ("hidden_states_dtype", "expected"),
    [
        (torch.bfloat16, True),
        (torch.float16, True),
        (torch.float32, False),  # already training in fp32; masters redundant
    ],
)
def test_trainer_fp32_master_mode_gating(
    hidden_states_dtype: torch.dtype, expected: bool
):
    trainer = Trainer.__new__(Trainer)
    trainer.config = make_config(hidden_states_dtype=hidden_states_dtype)
    assert trainer._fp32_master_mode is expected


def test_missing_optimizer_state_resumes_fresh(tmp_path: Path):
    # The legacy-mismatch error advises deleting optimizer_state_dict.pt; the
    # loader must then no-op instead of crashing on the missing file.
    (tmp_path / "0").mkdir()
    checkpointer = SingleGPUCheckpointer(tmp_path)
    checkpointer.previous_epoch = 0

    model = make_bf16_linear()
    opt = FP32MasterOptimizer(list(model.named_parameters()), adamw_factory)
    checkpointer.load_optimizer_state_dict(model, [opt])  # no exception


class TinyHFModel(PreTrainedModel):
    config_class = PretrainedConfig

    def __init__(self, config: PretrainedConfig):
        super().__init__(config)
        self.lin = nn.Linear(4, 4, bias=False)


def test_single_gpu_checkpointer_round_trip_keeps_fp32_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    # get_current_device() falls back to "cuda:0" on accelerator-less machines.
    monkeypatch.setattr(
        "speculators.train.checkpointer.get_current_device", lambda: "cpu"
    )
    model = TinyHFModel(PretrainedConfig()).to(torch.bfloat16)
    opt = FP32MasterOptimizer(list(model.named_parameters()), adamw_factory)
    run_constant_grad_steps(model, opt, 5)
    saved_masters = [m.clone() for m in opt.state_dict()["fp32_masters"]]

    checkpointer = SingleGPUCheckpointer(tmp_path)
    checkpointer.save_checkpoint(model, [opt], epoch=0)

    # The optimizer payload must keep the masters in fp32 (no bf16 downcast).
    payload = torch.load(checkpointer.optimizer_path(0), weights_only=True)
    assert all(m.dtype == torch.float32 for m in payload["fp32_masters"])

    fresh_model = TinyHFModel(PretrainedConfig()).to(torch.bfloat16)
    fresh_opt = FP32MasterOptimizer(list(fresh_model.named_parameters()), adamw_factory)
    checkpointer.previous_epoch = 0
    checkpointer.load_optimizer_state_dict(fresh_model, [fresh_opt])

    restored_masters = fresh_opt.state_dict()["fp32_masters"]
    for saved, restored in zip(saved_masters, restored_masters, strict=True):
        assert torch.equal(saved, restored)
