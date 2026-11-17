"""Optimizer resume preserves tensor identity across backbone composition."""

import copy
import json
from typing import Any, cast

import pytest
import torch
from torch.distributed.checkpoint.state_dict import (
    StateDictOptions,
    get_optimizer_state_dict,
)

from speculators.models.dflash.checkpoint import legacy_key
from speculators.train import checkpointer as checkpointer_module
from speculators.train.checkpointer import SingleGPUCheckpointer
from speculators.train.optimizer_checkpoint import (
    optimizer_metadata,
    prepare_optimizer_state,
    save_optimizer_metadata,
)
from tests.unit.models.test_dflash_family_contracts import make_contract_model


def _step(model, optimizers):
    for index, parameter in enumerate(model.parameters()):
        if parameter.requires_grad:
            parameter.grad = torch.full_like(parameter, (index + 1) * 0.01)
    for optimizer in optimizers:
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)


@pytest.mark.parametrize("named", [False, True])
@pytest.mark.parametrize("multiple", [False, True])
def test_resume_next_step_matches_uninterrupted(named, multiple, tmp_path, monkeypatch):
    monkeypatch.setattr(checkpointer_module, "get_current_device", lambda: "cpu")
    original = make_contract_model("dspark", True)
    parameters = list(original.named_parameters() if named else original.parameters())
    groups = [parameters[::2], parameters[1::2]] if multiple else [parameters]
    optimizers: list[torch.optim.Optimizer] = [
        torch.optim.AdamW(group, lr=1e-3) for group in groups
    ]
    _step(original, optimizers)
    SingleGPUCheckpointer(tmp_path).save_checkpoint(
        original, optimizers, epoch=0, float_dtype=torch.float32
    )
    resumed = make_contract_model("dspark", True)
    parameters = list(resumed.named_parameters() if named else resumed.parameters())
    groups = [parameters[::2], parameters[1::2]] if multiple else [parameters]
    restored_optimizers: list[torch.optim.Optimizer] = [
        torch.optim.AdamW(group, lr=1e-3) for group in groups
    ]
    checkpointer = SingleGPUCheckpointer(tmp_path)
    checkpointer.load_model_state_dict(resumed)
    checkpointer.load_optimizer_state_dict(resumed, restored_optimizers)
    _step(original, optimizers)
    _step(resumed, restored_optimizers)
    for (name, parameter), (other_name, other) in zip(
        original.named_parameters(), resumed.named_parameters(), strict=True
    ):
        assert name == other_name
        torch.testing.assert_close(parameter, other, atol=0, rtol=0)


def test_named_legacy_optimizer_is_mapped_without_tensor_copy(tmp_path):
    model = make_contract_model("dflash", False)
    optimizer = torch.optim.AdamW(model.named_parameters())
    _step(model, [optimizer])
    state = optimizer.state_dict()
    for group in state["param_groups"]:
        group["param_names"] = [
            "module." + legacy_key(name) for name in group["param_names"]
        ]
    prepared = prepare_optimizer_state(
        tmp_path / "missing.json", model, [optimizer], state, distributed=False
    )
    assert prepared["param_groups"][0]["param_names"] == [
        name for name, _ in model.named_parameters()
    ]
    for key in state["state"]:
        assert prepared["state"][key]["exp_avg"] is state["state"][key]["exp_avg"]


def test_legacy_distributed_optimizer_names_are_mapped(tmp_path):
    model = make_contract_model("dflash", False)
    optimizer = torch.optim.AdamW(model.named_parameters())
    _step(model, [optimizer])
    state = cast(
        "dict[str, Any]",
        get_optimizer_state_dict(
            model, optimizer, options=StateDictOptions(full_state_dict=True)
        ),
    )
    legacy = dict(
        state,
        state={legacy_key(name): value for name, value in state["state"].items()},
        param_groups=[
            dict(group, params=[legacy_key(name) for name in group["params"]])
            for group in state["param_groups"]
        ],
    )
    prepared = prepare_optimizer_state(
        tmp_path / "missing.json", model, [optimizer], legacy, distributed=True
    )
    assert prepared["state"].keys() == state["state"].keys()
    assert prepared["param_groups"] == state["param_groups"]


def test_unnamed_legacy_optimizer_is_rejected(tmp_path):
    model = make_contract_model("dflash", False)
    optimizer = torch.optim.AdamW(model.parameters())
    with pytest.raises(ValueError, match="unnamed legacy.*optimizer"):
        prepare_optimizer_state(
            tmp_path / "missing.json",
            model,
            [optimizer],
            optimizer.state_dict(),
            distributed=False,
        )


@pytest.mark.parametrize("change", ["version", "order", "optimizer"])
def test_changed_optimizer_identity_is_rejected(change, tmp_path):
    model = make_contract_model("dflash", False)
    optimizer = torch.optim.AdamW(model.parameters())
    metadata = optimizer_metadata(model, [optimizer], distributed=False)
    if change == "version":
        metadata["version"] = 2
    elif change == "order":
        metadata["parameter_groups"][0][0].reverse()
    else:
        metadata["optimizer_types"] = ["different.Optimizer"]
    path = tmp_path / "optimizer_metadata.json"
    path.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="metadata does not match"):
        prepare_optimizer_state(
            path, model, [optimizer], optimizer.state_dict(), distributed=False
        )


def test_legacy_names_cannot_hide_reordered_parameters(tmp_path):
    model = make_contract_model("dflash", False)
    optimizer = torch.optim.AdamW(model.named_parameters())
    state = copy.deepcopy(optimizer.state_dict())
    state["param_groups"][0]["param_names"].reverse()
    with pytest.raises(ValueError, match="identities or ordering differ"):
        prepare_optimizer_state(
            tmp_path / "missing.json", model, [optimizer], state, distributed=False
        )


def test_metadata_validates_distributed_optimizer_topology(tmp_path):
    model = make_contract_model("dflash", False)
    optimizer = torch.optim.AdamW(model.parameters())
    path = tmp_path / "optimizer_metadata.json"
    save_optimizer_metadata(path, model, [optimizer], distributed=True)
    with pytest.raises(ValueError, match="metadata does not match"):
        prepare_optimizer_state(
            path, model, [optimizer], optimizer.state_dict(), distributed=False
        )
