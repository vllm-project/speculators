"""Unit tests for FastMTPSpeculator forward pass structure."""

import pytest
import torch
from transformers.models.qwen3_next.configuration_qwen3_next import Qwen3NextConfig

from speculators.models.fast_mtp import FastMTPSpeculator

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def tiny_qwen3_next_config():
    """Minimal Qwen3-Next config for fast testing (hidden_size=64, vocab=100)."""
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
def model(tiny_qwen3_next_config):
    return FastMTPSpeculator.from_training_args(
        verifier_config=tiny_qwen3_next_config,
        num_speculative_steps=3,
        verifier_name_or_path=None,
    )


@pytest.fixture
def batch():
    B, L, H = 2, 8, 64
    return {
        "input_ids": torch.randint(0, 100, (B, L)),
        "hidden_states": torch.randn(B, L, H),
    }


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------


def test_fast_mtp_speculator_registered_as_mtp() -> None:
    """FastMTPSpeculator must declare 'mtp' as its speculators_model_type."""
    # The registry key is the value of speculators_model_type on the config class.
    assert (
        FastMTPSpeculator.config_class.model_fields["speculators_model_type"].default
        == "mtp"
    )  # noqa: E501


# ---------------------------------------------------------------------------
# Forward output structure
# ---------------------------------------------------------------------------


def test_forward_returns_three_tuple(model, batch) -> None:
    result = model(**batch)
    assert isinstance(result, tuple)
    assert len(result) == 3


def test_forward_logits_is_list(model, batch) -> None:
    logits_list, _, _ = model(**batch)
    assert isinstance(logits_list, list)


def test_forward_logits_list_has_k_elements(model, batch) -> None:
    logits_list, _, _ = model(**batch)
    assert len(logits_list) == 3  # num_speculative_steps=3


def test_forward_each_logits_shape_is_blv(model, batch) -> None:
    B = batch["input_ids"].shape[0]
    logits_list, _, _ = model(**batch)
    for step, logits in enumerate(logits_list):
        expected_len = batch["input_ids"].shape[1] - step - 1
        assert logits.shape == (B, expected_len, 100), (
            f"Step {step}: expected ({B}, {expected_len}, 100), got {logits.shape}"
        )


def test_forward_loss_is_scalar_tensor(model, batch) -> None:
    _, loss, _ = model(**batch)
    assert isinstance(loss, torch.Tensor)
    assert loss.ndim == 0


def test_forward_loss_is_finite(model, batch) -> None:
    _, loss, _ = model(**batch)
    assert torch.isfinite(loss)


def test_forward_metrics_are_tensors(model, batch) -> None:
    _, _, metrics = model(**batch)
    assert isinstance(metrics, dict)
    for key, val in metrics.items():
        assert isinstance(val, torch.Tensor), f"metrics[{key!r}] is not a tensor"


def test_forward_metrics_has_step_keys(model, batch) -> None:
    _, _, metrics = model(**batch)
    for step in range(3):
        assert f"loss_step_{step}" in metrics


# ---------------------------------------------------------------------------
# Frozen weights
# ---------------------------------------------------------------------------


def test_embed_tokens_trainable_without_verifier(model) -> None:
    """Without a verifier path, embed_tokens has random init and is trainable."""
    # The model fixture uses verifier_name_or_path=None, so no verifier weights
    # are loaded — embed_tokens.weight.requires_grad is True (trainable).
    # When a verifier IS loaded, _setup_embeddings_and_lm_head sets requires_grad=False.
    assert model.embed_tokens.weight.requires_grad


def test_lm_head_trainable_without_verifier(model) -> None:
    """Without a verifier path, lm_head has random init and is trainable."""
    assert model.lm_head.weight.requires_grad


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def test_shape_mismatch_raises(model) -> None:
    input_ids = torch.randint(0, 100, (2, 8))
    hidden_states = torch.randn(3, 8, 64)  # batch mismatch
    with pytest.raises(ValueError, match="does not match"):
        model(input_ids=input_ids, hidden_states=hidden_states)


def test_step_weights_wrong_length_raises(model, batch) -> None:
    with pytest.raises(ValueError, match="step_weights"):
        model(**batch, step_weights=[0.5, 0.5])  # need 3 weights, got 2


# ---------------------------------------------------------------------------
# step_weights effect on loss
# ---------------------------------------------------------------------------


def test_step_weights_change_loss_value(model, batch) -> None:
    torch.manual_seed(0)
    _, loss_default, _ = model(**batch)
    _, loss_custom, _ = model(**batch, step_weights=[0.8, 0.1, 0.1])
    # Different weights should produce different total loss
    assert not torch.allclose(loss_default, loss_custom)


# ---------------------------------------------------------------------------
# loss_mask effect
# ---------------------------------------------------------------------------


def test_loss_mask_zeros_reduce_loss(model, batch) -> None:
    torch.manual_seed(1)
    _, loss_all_ones, _ = model(**batch, loss_mask=torch.ones_like(batch["input_ids"]))
    _, loss_some_zeros, _ = model(
        **batch,
        loss_mask=torch.cat(
            [
                torch.zeros_like(batch["input_ids"][:, :4]),
                torch.ones_like(batch["input_ids"][:, 4:]),
            ],
            dim=1,
        ),
    )
    # More positions excluded → fewer non-(-100) targets → different loss value
    assert not torch.allclose(loss_all_ones, loss_some_zeros)


# ---------------------------------------------------------------------------
# get_trainer_kwargs
# ---------------------------------------------------------------------------


def test_get_trainer_kwargs_default_weights() -> None:
    train_kw, val_kw = FastMTPSpeculator.get_trainer_kwargs()
    assert "step_weights" in train_kw
    assert len(train_kw["step_weights"]) == 3
    assert sum(train_kw["step_weights"]) == pytest.approx(1.0, abs=0.01)


def test_get_trainer_kwargs_custom_weights() -> None:
    weights = [0.6, 0.3, 0.1]
    train_kw, val_kw = FastMTPSpeculator.get_trainer_kwargs(step_weights=weights)
    assert train_kw["step_weights"] == weights
    assert val_kw["step_weights"] == weights
