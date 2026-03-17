"""Unit tests for FastMTPSpeculator model."""

import pytest
import torch
from torch import nn
from transformers.models.qwen2.configuration_qwen2 import Qwen2Config

from speculators import SpeculatorModel, SpeculatorsConfig, VerifierConfig
from speculators.models.fast_mtp import FastMTPConfig, FastMTPSpeculator
from speculators.proposals import GreedyTokenProposalConfig

_NO_VERIFIER = VerifierConfig(name_or_path=None, architectures=[])

HIDDEN_SIZE = 64
VOCAB_SIZE = 256
NUM_STEPS = 3
BATCH_SIZE = 2
SEQ_LEN = 12  # must satisfy SEQ_LEN > NUM_STEPS + 2


@pytest.fixture
def tiny_tc():
    return Qwen2Config(
        hidden_size=HIDDEN_SIZE,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        intermediate_size=128,
        vocab_size=VOCAB_SIZE,
        max_position_embeddings=64,
    )


@pytest.fixture
def fast_mtp_config(tiny_tc):
    return FastMTPConfig(
        transformer_layer_config=tiny_tc,
        speculators_config=SpeculatorsConfig(
            algorithm="mtp",
            proposal_methods=[GreedyTokenProposalConfig(speculative_tokens=NUM_STEPS)],
            default_proposal_method="greedy",
            verifier=_NO_VERIFIER,
        ),
    )


@pytest.fixture
def model(fast_mtp_config):
    m = FastMTPSpeculator(fast_mtp_config)
    m.eval()
    return m


@pytest.fixture
def inputs():
    torch.manual_seed(42)
    input_ids = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, SEQ_LEN))
    hidden_states = torch.randn(BATCH_SIZE, SEQ_LEN, HIDDEN_SIZE)
    return input_ids, hidden_states


@pytest.mark.smoke
def test_embed_tokens_always_initialized(model):
    assert isinstance(model.embed_tokens, nn.Embedding)
    assert model.embed_tokens.weight.shape == (VOCAB_SIZE, HIDDEN_SIZE)


@pytest.mark.smoke
def test_lm_head_always_initialized(model):
    assert isinstance(model.lm_head, nn.Linear)
    assert model.lm_head.weight.shape == (VOCAB_SIZE, HIDDEN_SIZE)
    assert model.lm_head.bias is None


@pytest.mark.smoke
def test_mtp_layers_initialized(model):
    assert len(model.mtp_layers) == 1


@pytest.mark.smoke
def test_embed_tokens_and_lm_head_frozen(model):
    """embed_tokens and lm_head must always be frozen; only mtp_layers are trained."""
    assert not model.embed_tokens.weight.requires_grad
    assert not model.lm_head.weight.requires_grad
    trainable = [n for n, p in model.named_parameters() if p.requires_grad]
    assert all("mtp_layers" in n for n in trainable)


@pytest.mark.smoke
def test_registry():
    assert SpeculatorModel.registry is not None
    assert "mtp" in SpeculatorModel.registry
    assert SpeculatorModel.registry["mtp"] is FastMTPSpeculator


@pytest.mark.smoke
def test_output_shapes_no_labels(model, inputs):
    # At step k: valid_len = seq_len - k - 2 (one for embed offset, one for label).
    input_ids, hidden_states = inputs
    with torch.no_grad():
        out = model(input_ids=input_ids, hidden_states=hidden_states)

    assert "logits_list" in out
    assert len(out["logits_list"]) == NUM_STEPS
    assert out["loss"] is None

    for step, logits in enumerate(out["logits_list"]):
        expected_len = SEQ_LEN - step - 2
        assert logits.shape == (BATCH_SIZE, expected_len, VOCAB_SIZE), (
            f"step {step}: got {list(logits.shape)}, "
            f"expected [{BATCH_SIZE}, {expected_len}, {VOCAB_SIZE}]"
        )


@pytest.mark.smoke
def test_no_nan_inf_in_logits(model, inputs):
    input_ids, hidden_states = inputs
    with torch.no_grad():
        out = model(input_ids=input_ids, hidden_states=hidden_states)
    for step, logits in enumerate(out["logits_list"]):
        assert not torch.isnan(logits).any(), f"NaN in step {step} logits"
        assert not torch.isinf(logits).any(), f"Inf in step {step} logits"


@pytest.mark.smoke
def test_recursive_hidden_states(model, inputs):
    """Verify step 1's hidden input is step 0's MTP output, not the original hidden.

    FastMTP is recursive: each step conditions on the speculated future rather than
    the original verifier context. This test verifies that property is live.
    """
    input_ids, hidden_states = inputs

    layer_inputs: list[torch.Tensor] = []
    layer_outputs: list[torch.Tensor] = []
    original_forward = model.mtp_layers[0].forward

    def capturing_forward(*args, **kwargs):
        # hidden_states is always passed as a keyword argument from core.py
        hs = kwargs.get("hidden_states", args[0] if args else None)
        layer_inputs.append(hs.detach().clone())
        result = original_forward(*args, **kwargs)
        layer_outputs.append(result.detach().clone())
        return result

    model.mtp_layers[0].forward = capturing_forward
    try:
        with torch.no_grad():
            model(input_ids=input_ids, hidden_states=hidden_states)
    finally:
        model.mtp_layers[0].forward = original_forward

    assert len(layer_inputs) == NUM_STEPS

    # Step 0: input must be the original verifier hidden, trimmed to valid_len.
    valid_len_0 = SEQ_LEN - 2
    assert torch.allclose(layer_inputs[0], hidden_states[:, :valid_len_0])

    # Step 1: input must be step 0's MTP output, trimmed to the next valid_len.
    # This is the recursive property: mtp_output[step] feeds hidden[step+1].
    valid_len_1 = SEQ_LEN - 3
    assert torch.allclose(layer_inputs[1], layer_outputs[0][:, :valid_len_1])

    # Sanity: step 1's input must differ from the original hidden at the same positions,
    # confirming the MTP layer actually transforms its input.
    assert not torch.allclose(layer_inputs[1], hidden_states[:, :valid_len_1]), (
        "step 1 hidden equals original verifier hidden — recursive update is not live"
    )


@pytest.mark.smoke
def test_loss_is_finite_with_labels(model, inputs):
    input_ids, hidden_states = inputs
    labels = input_ids.clone()
    with torch.no_grad():
        out = model(input_ids=input_ids, hidden_states=hidden_states, labels=labels)
    assert out["loss"] is not None
    assert torch.isfinite(out["loss"])
    assert out["loss"].item() > 0


@pytest.mark.smoke
def test_loss_metrics_per_step(model, inputs):
    input_ids, hidden_states = inputs
    labels = input_ids.clone()
    with torch.no_grad():
        out = model(input_ids=input_ids, hidden_states=hidden_states, labels=labels)
    for step in range(NUM_STEPS):
        key = f"loss_step_{step}"
        assert key in out["metrics"], f"Missing {key} in metrics"
        assert torch.isfinite(torch.tensor(out["metrics"][key]))


@pytest.mark.smoke
def test_loss_mask_all_ones_equals_no_mask(model, inputs):
    """An all-ones mask must produce identical loss to no mask at all.

    This holds because no label positions are set to -100 when every mask bit is 1.
    """
    input_ids, hidden_states = inputs
    labels = input_ids.clone()

    with torch.no_grad():
        out_no_mask = model(
            input_ids=input_ids, hidden_states=hidden_states, labels=labels
        )
        out_ones_mask = model(
            input_ids=input_ids,
            hidden_states=hidden_states,
            labels=labels,
            loss_mask=torch.ones_like(labels),
        )

    assert torch.allclose(out_no_mask["loss"], out_ones_mask["loss"], atol=1e-6)


@pytest.mark.smoke
def test_loss_mask_partial_differs_from_full(model, inputs):
    """Masking out half the sequence positions must change the loss.

    This verifies that masked positions are actually excluded, not just weighted down.
    """
    input_ids, hidden_states = inputs
    labels = input_ids.clone()

    half_mask = torch.zeros_like(labels)
    half_mask[:, SEQ_LEN // 2 :] = 1

    with torch.no_grad():
        out_full = model(
            input_ids=input_ids, hidden_states=hidden_states, labels=labels
        )
        out_partial = model(
            input_ids=input_ids,
            hidden_states=hidden_states,
            labels=labels,
            loss_mask=half_mask,
        )

    assert not torch.allclose(out_full["loss"], out_partial["loss"], atol=1e-6), (
        "partial loss_mask produced identical loss to no mask — masking has no effect"
    )


@pytest.mark.smoke
def test_get_trainer_kwargs_uses_provided_weights():
    weights = [0.6, 0.3, 0.1]
    train_kw, val_kw = FastMTPSpeculator.get_trainer_kwargs(step_weights=weights)
    assert train_kw["step_weights"] == weights
    assert val_kw["step_weights"] == weights


@pytest.mark.smoke
def test_get_trainer_kwargs_falls_back_to_paper_defaults():
    train_kw, val_kw = FastMTPSpeculator.get_trainer_kwargs()
    assert train_kw["step_weights"] == [0.51, 0.31, 0.18]
    assert val_kw["step_weights"] == [0.51, 0.31, 0.18]
