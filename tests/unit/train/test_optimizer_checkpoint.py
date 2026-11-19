"""Same-version optimizer resume preserves the next update."""

import pytest
import torch

from speculators.models.modular import ModularDraftModel
from speculators.train import checkpointer as checkpointer_module
from speculators.train.checkpointer import SingleGPUCheckpointer
from tests.unit.models.test_dflash_family_contracts import make_contract_model
from tests.unit.models.test_modular_model import modular_config


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


def test_mixed_model_checkpoint_resumes_next_update(tmp_path, monkeypatch):
    monkeypatch.setattr(checkpointer_module, "get_current_device", lambda: "cpu")
    model = ModularDraftModel(modular_config(confidence=True))
    optimizer = torch.optim.AdamW(model.parameters())
    _step(model, [optimizer])
    SingleGPUCheckpointer(tmp_path).save_checkpoint(
        model, optimizer, epoch=0, float_dtype=torch.float32
    )
    restored = ModularDraftModel(modular_config(confidence=True))
    restored_optimizer = torch.optim.AdamW(restored.parameters())
    checkpointer = SingleGPUCheckpointer(tmp_path)
    checkpointer.load_model_state_dict(restored)
    checkpointer.load_optimizer_state_dict(restored, restored_optimizer)
    _step(model, [optimizer])
    _step(restored, [restored_optimizer])
    for (name, parameter), (other_name, other) in zip(
        model.named_parameters(), restored.named_parameters(), strict=True
    ):
        assert name == other_name
        torch.testing.assert_close(parameter, other, atol=0, rtol=0, equal_nan=True)
