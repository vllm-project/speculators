"""
Unit tests for FastMTPDraftModel in the Speculators library.
"""

from unittest.mock import patch

import pytest
import torch
from torch import nn
from transformers.configuration_utils import PretrainedConfig
from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.models.llama.modeling_llama import (
    LlamaDecoderLayer,
    LlamaRMSNorm,
    LlamaRotaryEmbedding,
)

from speculators import (
    SpeculatorModel,
    SpeculatorsConfig,
    VerifierConfig,
)
from speculators.models import FastMTPDraftModel, FastMTPSpeculatorConfig
from speculators.proposals import GreedyTokenProposalConfig

HIDDEN_SIZE = 64
VOCAB_SIZE = 256
DRAFT_VOCAB_SIZE = 256
SEQ_LEN = 32
NUM_HEADS = 4
NUM_KV_HEADS = 2
INTERMEDIATE_SIZE = 128
HEAD_DIM = HIDDEN_SIZE // NUM_HEADS


@pytest.fixture
def small_llama_config():
    return LlamaConfig(
        vocab_size=VOCAB_SIZE,
        hidden_size=HIDDEN_SIZE,
        intermediate_size=INTERMEDIATE_SIZE,
        num_hidden_layers=1,
        num_attention_heads=NUM_HEADS,
        num_key_value_heads=NUM_KV_HEADS,
        head_dim=HEAD_DIM,
        max_position_embeddings=512,
        rms_norm_eps=1e-5,
        attn_implementation="eager",
    )


@pytest.fixture
def sample_verifier_config():
    return VerifierConfig(
        name_or_path="test/verifier",
        architectures=["LlamaForCausalLM"],
    )


@pytest.fixture
def sample_speculators_config(sample_verifier_config):
    return SpeculatorsConfig(
        algorithm="fastmtp",
        proposal_methods=[GreedyTokenProposalConfig()],
        default_proposal_method="greedy",
        verifier=sample_verifier_config,
    )


@pytest.fixture
def fastmtp_config(small_llama_config, sample_speculators_config):
    return FastMTPSpeculatorConfig(
        transformer_layer_config=small_llama_config,
        draft_vocab_size=DRAFT_VOCAB_SIZE,
        num_speculative_steps=3,
        speculators_config=sample_speculators_config,
    )


def _mock_load_model_layers(layer_names, model_path):  # noqa: ARG001
    """Mock load_model_layers to return synthetic weights."""
    result = {}
    for name in layer_names:
        if "embed_tokens" in name:
            result[name] = torch.randn(VOCAB_SIZE, HIDDEN_SIZE)
        elif "lm_head" in name:
            result[name] = torch.randn(DRAFT_VOCAB_SIZE, HIDDEN_SIZE)
    return result


def _mock_auto_config(*args, **kwargs):
    """Mock AutoConfig.from_pretrained to return a LlamaConfig."""
    return LlamaConfig(
        vocab_size=VOCAB_SIZE,
        hidden_size=HIDDEN_SIZE,
        intermediate_size=INTERMEDIATE_SIZE,
        num_hidden_layers=1,
        num_attention_heads=NUM_HEADS,
        num_key_value_heads=NUM_KV_HEADS,
        head_dim=HEAD_DIM,
        max_position_embeddings=512,
    )


@pytest.fixture
def mock_verifier_deps():
    """Provide mock patches for verifier weight loading."""
    with (
        patch(
            "speculators.models.fastmtp.core.load_model_layers",
            side_effect=_mock_load_model_layers,
        ),
        patch(
            "speculators.models.fastmtp.core.AutoConfig.from_pretrained",
            side_effect=_mock_auto_config,
        ),
    ):
        yield


@pytest.fixture
def fastmtp_model(fastmtp_config, mock_verifier_deps):
    return FastMTPDraftModel(fastmtp_config)


@pytest.mark.smoke
def test_fastmtp_class_attributes():
    """Test FastMTPDraftModel class attributes."""
    assert FastMTPDraftModel.auto_package == "speculators.models"
    assert FastMTPDraftModel.registry_auto_discovery is True
    assert FastMTPDraftModel.config_class == FastMTPSpeculatorConfig
    assert FastMTPDraftModel.base_model_prefix == "model"


@pytest.mark.smoke
def test_fastmtp_registry():
    """Test FastMTPDraftModel is registered in SpeculatorModel registry."""
    assert SpeculatorModel.registry is not None
    assert "fastmtp" in SpeculatorModel.registry
    assert SpeculatorModel.registry["fastmtp"] == FastMTPDraftModel


@pytest.mark.smoke
def test_fastmtp_registered_model_class_from_config(fastmtp_config):
    """Test resolving FastMTPDraftModel from config."""
    model_class = SpeculatorModel.registered_model_class_from_config(fastmtp_config)
    assert model_class == FastMTPDraftModel


@pytest.mark.smoke
def test_fastmtp_initialization(fastmtp_model, fastmtp_config):
    """Test basic initialization of FastMTPDraftModel."""
    model = fastmtp_model

    assert isinstance(model, FastMTPDraftModel)
    assert model.hidden_size == HIDDEN_SIZE
    assert model.draft_vocab_size == DRAFT_VOCAB_SIZE
    assert model.num_speculative_steps == 3
    assert model.verifier_attachment_mode == "train_only"


@pytest.mark.smoke
def test_fastmtp_architecture_components(fastmtp_model):
    """Test that all MTP head components are properly initialized."""
    model = fastmtp_model

    # MTP head components
    assert isinstance(model.token_layernorm, LlamaRMSNorm)
    assert isinstance(model.hidden_layernorm, LlamaRMSNorm)
    assert isinstance(model.input_proj, nn.Linear)
    assert model.input_proj.weight.shape == (HIDDEN_SIZE, 2 * HIDDEN_SIZE)
    assert model.input_proj.bias is None
    assert isinstance(model.decoder_layer, LlamaDecoderLayer)
    assert isinstance(model.final_layernorm, LlamaRMSNorm)

    # Rotary embedding
    assert isinstance(model.rotary_emb, LlamaRotaryEmbedding)

    # Embeddings and LM heads
    assert isinstance(model.embed_tokens, nn.Embedding)
    assert model.embed_tokens.weight.shape == (VOCAB_SIZE, HIDDEN_SIZE)
    assert model.embed_tokens.weight.requires_grad is False

    assert isinstance(model.lm_head, nn.Linear)
    assert model.lm_head.weight.shape == (DRAFT_VOCAB_SIZE, HIDDEN_SIZE)
    assert model.lm_head.weight.requires_grad is True

    assert isinstance(model.verifier_lm_head, nn.Linear)
    assert model.verifier_lm_head.weight.shape == (DRAFT_VOCAB_SIZE, HIDDEN_SIZE)
    assert model.verifier_lm_head.weight.requires_grad is False


@pytest.mark.smoke
def test_fastmtp_initialization_no_vocab_mapping(fastmtp_model):
    """Test that vocab mapping buffers are None when not provided."""
    model = fastmtp_model
    assert model.t2d is None
    assert model.d2t is None


@pytest.mark.smoke
def test_fastmtp_initialization_mismatched_vocab_mapping(
    fastmtp_config, mock_verifier_deps
):
    """Test that providing only one of t2d/d2t raises an error."""
    t2d = torch.ones(VOCAB_SIZE, dtype=torch.bool)
    with pytest.raises(ValueError, match="Both t2d and d2t must be provided"):
        FastMTPDraftModel(fastmtp_config, t2d=t2d, d2t=None)


@pytest.mark.smoke
def test_fastmtp_forward_with_loss(fastmtp_model):
    """Test forward pass returns (draft_tokens, loss, metrics).

    When verifier hidden states are provided.
    """
    model = fastmtp_model
    model.eval()

    batch_size = 1
    hidden_states = torch.randn(batch_size, SEQ_LEN, HIDDEN_SIZE)
    input_ids = torch.randint(0, VOCAB_SIZE, (batch_size, SEQ_LEN))
    verifier_last_hidden_states = torch.randn(batch_size, SEQ_LEN, HIDDEN_SIZE)
    lengths = torch.tensor([SEQ_LEN], dtype=torch.long)
    loss_mask = torch.ones(batch_size, SEQ_LEN)

    with torch.no_grad():
        result = model(
            hidden_states=hidden_states,
            input_ids=input_ids,
            lengths=lengths,
            loss_mask=loss_mask,
            verifier_last_hidden_states=verifier_last_hidden_states,
            ttt_steps=3,
            ttt_step_loss_decay=1.0,
        )

    draft_tokens, loss, metrics = result

    # Check draft tokens
    assert len(draft_tokens) == 3
    for dt in draft_tokens:
        assert dt.shape == (batch_size, SEQ_LEN)
        assert dt.dtype == torch.long

    # Check loss
    assert loss.dim() == 0  # scalar
    assert torch.isfinite(loss)

    # Check metrics
    assert "loss" in metrics
    assert "loss_0" in metrics
    assert "loss_1" in metrics
    assert "loss_2" in metrics
    assert "full_acc_0" in metrics
    assert "cond_acc_0" in metrics


@pytest.mark.smoke
def test_fastmtp_forward_without_loss(fastmtp_model):
    """Test forward pass returns only draft_tokens.

    When verifier hidden states are not provided.
    """
    model = fastmtp_model
    model.eval()

    batch_size = 1
    hidden_states = torch.randn(batch_size, SEQ_LEN, HIDDEN_SIZE)
    input_ids = torch.randint(0, VOCAB_SIZE, (batch_size, SEQ_LEN))

    with torch.no_grad():
        result = model(
            hidden_states=hidden_states,
            input_ids=input_ids,
            ttt_steps=2,
        )

    assert isinstance(result, list)
    assert len(result) == 2
    for dt in result:
        assert dt.shape == (batch_size, SEQ_LEN)


@pytest.mark.smoke
def test_fastmtp_forward_variable_steps(fastmtp_model):
    """Test forward pass with different number of ttt_steps."""
    model = fastmtp_model
    model.eval()

    batch_size = 1
    hidden_states = torch.randn(batch_size, SEQ_LEN, HIDDEN_SIZE)
    input_ids = torch.randint(0, VOCAB_SIZE, (batch_size, SEQ_LEN))
    verifier_last_hidden_states = torch.randn(batch_size, SEQ_LEN, HIDDEN_SIZE)

    for num_steps in [1, 2, 5]:
        with torch.no_grad():
            result = model(
                hidden_states=hidden_states,
                input_ids=input_ids,
                verifier_last_hidden_states=verifier_last_hidden_states,
                ttt_steps=num_steps,
            )
        draft_tokens, loss, metrics = result
        assert len(draft_tokens) == num_steps


@pytest.mark.smoke
def test_fastmtp_forward_off_policy(fastmtp_model):
    """Test forward pass with use_off_policy_tokens=True."""
    model = fastmtp_model
    model.eval()

    batch_size = 1
    hidden_states = torch.randn(batch_size, SEQ_LEN, HIDDEN_SIZE)
    input_ids = torch.randint(0, VOCAB_SIZE, (batch_size, SEQ_LEN))
    verifier_last_hidden_states = torch.randn(batch_size, SEQ_LEN, HIDDEN_SIZE)

    with torch.no_grad():
        result = model(
            hidden_states=hidden_states,
            input_ids=input_ids,
            verifier_last_hidden_states=verifier_last_hidden_states,
            ttt_steps=3,
            use_off_policy_tokens=True,
        )

    draft_tokens, loss, metrics = result
    assert len(draft_tokens) == 3
    assert torch.isfinite(loss)


@pytest.mark.smoke
def test_fastmtp_state_dict_keys(fastmtp_model):
    """Test that state dict contains expected keys and excludes verifier weights."""
    state_dict = fastmtp_model.state_dict()

    # Should contain MTP head components
    assert any("token_layernorm" in k for k in state_dict)
    assert any("hidden_layernorm" in k for k in state_dict)
    assert any("input_proj" in k for k in state_dict)
    assert any("decoder_layer" in k for k in state_dict)
    assert any("final_layernorm" in k for k in state_dict)

    # Should contain trainable lm_head
    assert any("lm_head" in k for k in state_dict)

    # Should contain embed_tokens (saved for completeness)
    assert any("embed_tokens" in k for k in state_dict)


@pytest.mark.smoke
def test_fastmtp_import_model_classes(fastmtp_model):
    """Test that model classes are correctly imported."""
    config = fastmtp_model.config
    decoder_cls, norm_cls, rotary_cls = fastmtp_model._import_model_classes(
        config.transformer_layer_config
    )

    assert decoder_cls == LlamaDecoderLayer
    assert norm_cls == LlamaRMSNorm
    assert rotary_cls == LlamaRotaryEmbedding


@pytest.mark.smoke
def test_fastmtp_import_model_classes_invalid_config():
    """Test that invalid config raises TypeError."""

    class CustomConfig(PretrainedConfig):
        model_type = "custom_invalid"

    invalid_config = CustomConfig()

    with pytest.raises(TypeError, match="is not a valid causal language model"):
        FastMTPDraftModel._import_model_classes(FastMTPDraftModel, invalid_config)
