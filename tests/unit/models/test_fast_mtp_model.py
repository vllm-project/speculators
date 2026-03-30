"""Tests for FastMTPSpeculator."""

import pytest
import torch
from torch import nn
from transformers.models.qwen3_next.configuration_qwen3_next import Qwen3NextConfig

from speculators.models.fast_mtp import FastMTPSpeculator


@pytest.fixture(scope="module")
def tiny_config():
    return Qwen3NextConfig(
        hidden_size=64,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        intermediate_size=128,
        vocab_size=100,
        head_dim=32,
        full_attention_interval=1,
    )


@pytest.fixture(scope="module")
def model(tiny_config):
    return FastMTPSpeculator.from_training_args(
        verifier_config=tiny_config,
        num_speculative_steps=3,
        verifier_name_or_path=None,
    )


@pytest.fixture
def batch():
    return {
        "input_ids": torch.randint(0, 100, (2, 8)),
        "hidden_states": torch.randn(2, 8, 64),
    }


def test_forward_output_shapes(model, batch) -> None:
    logits_list, loss, metrics = model(**batch)
    B, L = batch["input_ids"].shape

    assert isinstance(logits_list, list)
    assert len(logits_list) == 3
    for k, logits in enumerate(logits_list):
        assert logits.shape == (B, L - k - 1, 100)

    assert loss.ndim == 0
    assert torch.isfinite(loss)
    for step in range(3):
        assert f"loss_step_{step}" in metrics


def test_state_dict_keys(model) -> None:
    sd = model.state_dict()
    assert not any(k.startswith("mtp_layers.") for k in sd)
    for key in (
        "mtp.pre_fc_norm_hidden.weight",
        "mtp.pre_fc_norm_embedding.weight",
        "mtp.fc.weight",
        "mtp.norm.weight",
    ):
        assert key in sd
    assert any(k.startswith("mtp.layers.0.") for k in sd)


def test_layers_property(model) -> None:
    assert isinstance(model.layers, nn.ModuleList)
    assert len(model.layers) == 1


def test_batch_size_mismatch_raises(model) -> None:
    with pytest.raises(ValueError, match="does not match"):
        model(
            input_ids=torch.randint(0, 100, (2, 8)), hidden_states=torch.randn(3, 8, 64)
        )


def test_step_weights_wrong_length_raises(model, batch) -> None:
    with pytest.raises(ValueError, match="step_weights"):
        model(**batch, step_weights=[0.5, 0.5])


def test_get_trainer_kwargs_default_weights() -> None:
    train_kw, val_kw = FastMTPSpeculator.get_trainer_kwargs()
    assert len(train_kw["step_weights"]) == 3
    assert sum(train_kw["step_weights"]) == pytest.approx(1.0, abs=0.01)


def test_get_trainer_kwargs_custom_weights() -> None:
    weights = [0.6, 0.3, 0.1]
    train_kw, val_kw = FastMTPSpeculator.get_trainer_kwargs(step_weights=weights)
    assert train_kw["step_weights"] == weights
    assert val_kw["step_weights"] == weights
