"""Tests for fp32 master weights."""

from types import SimpleNamespace

import torch
from torch import nn

from speculators.train.optimizers import build_optimizers, make_fp32_masters


def _model() -> tuple[nn.Module, nn.Linear]:
    """Returns the model and its first layer, so a test can freeze that layer."""
    torch.manual_seed(0)
    first = nn.Linear(8, 8, bias=False)
    model = nn.Sequential(first, nn.Linear(8, 8, bias=False))
    model.to(torch.bfloat16)
    return model, first


def _config(fp32: bool) -> SimpleNamespace:
    return SimpleNamespace(
        optimizer="adamw", lr=1e-4, weight_decay=0.0, fp32_master_weights=fp32
    )


def test_masters_are_fp32_copies_of_the_bf16_parameters():
    model, _ = _model()

    masters = make_fp32_masters(model)

    assert set(masters) == {name for name, _ in model.named_parameters()}
    for name, param in model.named_parameters():
        assert masters[name].dtype == torch.float32
        assert masters[name].requires_grad
        torch.testing.assert_close(masters[name], param.float())


def test_frozen_parameters_get_no_master():
    """A frozen bf16 parameter in a group of fp32 masters mixes dtypes.

    torch's grouped Adam step requires one dtype per group, so a frozen weight
    left in place raises rather than being harmlessly ignored.
    """
    model, first = _model()
    first.weight.requires_grad_(False)

    masters = make_fp32_masters(model)

    assert "0.weight" not in masters
    assert "1.weight" in masters


def test_small_updates_accumulate_in_fp32_but_are_lost_in_bf16():
    """The reason the option exists: bf16 has ~8 bits of mantissa.

    Near 1.0 the smallest representable bf16 step is 2**-8 ~ 0.0039. An update
    below that -- where a decayed LR schedule ends up -- is rounded away every
    single time it is applied to the bf16 weight, so it never accumulates. An
    fp32 master accumulates the same updates and crosses the threshold.
    """
    start = torch.full((4,), 1.0, dtype=torch.bfloat16)
    tiny = 1e-4
    steps = 100

    direct = start.clone()
    master = start.float()
    for _ in range(steps):
        direct += torch.full_like(direct, tiny)
        master += tiny

    assert torch.equal(direct, start), "expected bf16 to swallow every update"
    assert not torch.equal(master.to(torch.bfloat16), start)


def test_build_optimizers_steps_the_masters_and_leaves_params_untouched():
    model, _ = _model()

    optimizers, pairs = build_optimizers(model, _config(fp32=True))

    assert len(optimizers) == 1
    assert len(pairs) == len(list(model.named_parameters()))
    stepped = {id(p) for group in optimizers[0].param_groups for p in group["params"]}
    assert stepped == {id(master) for _, master in pairs}
    assert stepped.isdisjoint({id(p) for p in model.parameters()})


def test_build_optimizers_without_the_flag_is_unchanged():
    model, _ = _model()

    optimizers, pairs = build_optimizers(model, _config(fp32=False))

    assert pairs == []
    stepped = {id(p) for group in optimizers[0].param_groups for p in group["params"]}
    assert stepped == {id(p) for p in model.parameters()}
