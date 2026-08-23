"""Dion3 is an opt-in substitute for Muon over the identical parameter split.

These tests pin the parts that are easy to get wrong and cheap to check without a
GPU or the optional ``dion`` install: that the option is accepted by the config,
that the parameter split is the same one Muon uses, that the learning-rate
convention is matched rather than left at dion's default, and that a missing
``dion`` install fails with an actionable message rather than an ImportError from
somewhere deep inside the trainer.
"""

import sys
from types import SimpleNamespace
from unittest import mock

import pytest
import torch
from torch import nn

from speculators.train.optimizers import (
    build_optimizers,
    split_named_params_for_muon,
)


class _Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(8, 16, bias=False)
        self.norm = nn.LayerNorm(8)
        self.embed_tokens = nn.Embedding(32, 8)


def _config(**over):
    base = {
        "optimizer": "dion3",
        "lr": 1e-4,
        "weight_decay": 0.01,
        "muon_lr": 1e-3,
        "muon_momentum": 0.95,
        "muon_weight_decay": 0.1,
        "muon_ns_steps": 5,
        "muon_adjust_lr_fn": "match_rms_adamw",
        "dion_fraction": 0.25,
        "dion_selection_scope": "global",
    }
    base.update(over)
    return SimpleNamespace(**base)


def test_config_accepts_dion3():
    from speculators.train.config.schema import OptimizerArgs  # noqa: PLC0415

    args = OptimizerArgs(optimizer="dion3", dion_fraction=0.5)
    assert args.optimizer == "dion3"
    assert args.dion_fraction == 0.5
    # fraction is a proportion; 0 and >1 are meaningless
    with pytest.raises(ValueError):
        OptimizerArgs(optimizer="dion3", dion_fraction=0.0)
    with pytest.raises(ValueError):
        OptimizerArgs(optimizer="dion3", dion_fraction=1.5)


def test_missing_dion_is_an_actionable_error():
    """dion is not on PyPI, so this is the common failure and must explain itself."""
    with (
        mock.patch.dict(sys.modules, {"dion": None}),
        pytest.raises(ImportError, match="github.com/microsoft/dion"),
    ):
        build_optimizers(_Tiny(), _config())


def test_dion3_reuses_the_muon_parameter_split():
    """The split must be Muon's, so the two optimizers stay comparable."""
    model = _Tiny()
    matrix, scalar = split_named_params_for_muon(model)
    matrix_names = {n for n, _ in matrix}
    scalar_names = {n for n, _ in scalar}

    assert "proj.weight" in matrix_names
    # embeddings and norms are excluded from the orthogonalized group
    assert "embed_tokens.weight" in scalar_names
    assert "norm.weight" in scalar_names
    assert not matrix_names & scalar_names


def test_dion3_is_constructed_with_matched_optimizer_conventions():
    """Dion's defaults differ from the existing Muon and scalar AdamW paths.

    ``rms_norm`` is ``0.2*sqrt(max(fan_out, fan_in))``, the same expression as
    torch Muon's ``match_rms_adamw``. Getting this wrong silently changes the
    effective learning rate, which would make any Muon-vs-Dion3 comparison
    meaningless. Dion also defaults AdamW beta2 to 0.95, while the existing
    scalar optimizer uses torch's 0.999 default.
    """
    pytest.importorskip("dion", reason="optional dependency, install from git")
    captured = {}

    import dion  # noqa: PLC0415

    class _Spy(dion.Dion3):
        def __init__(self, param_groups, **kwargs):
            captured.update(kwargs)
            captured["groups"] = [g.get("algorithm") for g in param_groups]
            super().__init__(param_groups, **kwargs)

    with mock.patch.object(dion, "Dion3", _Spy):
        optimizers = build_optimizers(_Tiny(), _config())

    assert len(optimizers) == 1
    assert captured["adjust_lr"] == "rms_norm"
    assert captured["betas"] == (0.9, 0.999)
    assert captured["fraction"] == 0.25
    assert captured["selection_scope"] == "global"
    assert captured["groups"] == ["nordion2", "adamw"]
    adamw_group = optimizers[0].param_groups[1]
    assert (adamw_group["beta1"], adamw_group["beta2"]) == (0.9, 0.999)


def test_step_is_wrapped_for_static_shapes():
    """The workaround must be scoped, not process-global, and version-gated.

    torch 2.13 regressed inductor here; 2.12.x is unaffected and should not pay
    the extra recompiles that pinning shapes static costs.
    """
    pytest.importorskip("dion", reason="optional dependency, install from git")
    dynamo_config = torch._dynamo.config
    before = dynamo_config.automatic_dynamic_shapes
    optimizers = build_optimizers(_Tiny(), _config())
    # constructing must not touch the global flag
    assert dynamo_config.automatic_dynamic_shapes is before
    # and step must be the instance-level wrapper on the affected torch versions
    from torch.torch_version import TorchVersion  # noqa: PLC0415

    affected = TorchVersion(torch.__version__) >= (2, 13)
    assert ("step" in optimizers[0].__dict__) is affected


def test_muon_path_is_unchanged():
    """The default path must not be perturbed by the new branch."""
    optimizers = build_optimizers(_Tiny(), _config(optimizer="muon"))
    assert [type(o).__name__ for o in optimizers] == ["Muon", "AdamW"]
    # the dion3 step wrapper must not be applied to the muon path
    assert "step" not in optimizers[0].__dict__
