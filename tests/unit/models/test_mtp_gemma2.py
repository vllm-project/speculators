"""Unit tests for Gemma2 MTP Query-Only Attention."""

import pytest
import torch
from transformers.models.gemma2.configuration_gemma2 import Gemma2Config

from speculators.config import SpeculatorsConfig, VerifierConfig
from speculators.models.base_components import model_classes
from speculators.models.mtp.config import MTPSpeculatorConfig
from speculators.models.mtp.core import MTPDraftModel
from speculators.proposals.greedy import GreedyTokenProposalConfig

if "gemma2" not in model_classes:
    pytest.skip(
        "transformers < 4.42 installed, skipping gemma2 tests", allow_module_level=True
    )

from speculators.models.mtp.model_definitions import QueryOnlyGemma2Attention


@pytest.fixture
def gemma2_config():
    config = Gemma2Config(
        hidden_size=64,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        attention_bias=False,
        attention_dropout=0.0,
    )
    config._attn_implementation = "eager"
    config.sliding_window = 4096
    config.query_pre_attn_scalar = 224
    config.attn_logit_softcapping = 50.0
    config.layer_types = ["sliding_attention"]
    return config


def test_query_only_gemma2_attention_local_kv(gemma2_config):
    """Test that local KV cache is used when sliding_window > 0."""
    attn = QueryOnlyGemma2Attention(gemma2_config, layer_idx=0)

    batch_sz, seq_len = 2, 5
    hidden_states = torch.randn(batch_sz, seq_len, 64)

    local_kv = torch.randn(batch_sz, seq_len, 2, 2, 16)
    global_kv = torch.randn(batch_sz, seq_len, 2, 2, 16)

    cos = torch.randn(batch_sz, seq_len, 16)
    sin = torch.randn(batch_sz, seq_len, 16)

    output1, _ = attn(
        hidden_states=hidden_states,
        position_embeddings=(cos, sin),
        attention_mask=None,
        verifier_kv_last_local=local_kv,
        verifier_kv_last_global=global_kv,
    )
    assert output1.shape == (batch_sz, seq_len, 64)

    global_kv_changed = torch.randn_like(global_kv)
    output2, _ = attn(
        hidden_states=hidden_states,
        position_embeddings=(cos, sin),
        attention_mask=None,
        verifier_kv_last_local=local_kv,
        verifier_kv_last_global=global_kv_changed,
    )
    assert torch.equal(output1, output2)

    local_kv_changed = torch.randn_like(local_kv)
    output3, _ = attn(
        hidden_states=hidden_states,
        position_embeddings=(cos, sin),
        attention_mask=None,
        verifier_kv_last_local=local_kv_changed,
        verifier_kv_last_global=global_kv,
    )
    assert not torch.equal(output1, output3)


def test_query_only_gemma2_attention_global_kv(gemma2_config):
    """Test that global KV cache is used when layer_types is full_attention."""
    gemma2_config.layer_types = ["full_attention"]
    gemma2_config.sliding_window = None
    attn = QueryOnlyGemma2Attention(gemma2_config, layer_idx=0)

    batch_sz, seq_len = 2, 5
    hidden_states = torch.randn(batch_sz, seq_len, 64)

    local_kv = torch.randn(batch_sz, seq_len, 2, 2, 16)
    global_kv = torch.randn(batch_sz, seq_len, 2, 2, 16)

    cos = torch.randn(batch_sz, seq_len, 16)
    sin = torch.randn(batch_sz, seq_len, 16)

    output1, _ = attn(
        hidden_states=hidden_states,
        position_embeddings=(cos, sin),
        attention_mask=None,
        verifier_kv_last_local=local_kv,
        verifier_kv_last_global=global_kv,
    )
    assert output1.shape == (batch_sz, seq_len, 64)

    local_kv_changed = torch.randn_like(local_kv)
    output2, _ = attn(
        hidden_states=hidden_states,
        position_embeddings=(cos, sin),
        attention_mask=None,
        verifier_kv_last_local=local_kv_changed,
        verifier_kv_last_global=global_kv,
    )
    assert torch.equal(output1, output2)

    global_kv_changed = torch.randn_like(global_kv)
    output3, _ = attn(
        hidden_states=hidden_states,
        position_embeddings=(cos, sin),
        attention_mask=None,
        verifier_kv_last_local=local_kv,
        verifier_kv_last_global=global_kv_changed,
    )
    assert not torch.equal(output1, output3)


def test_query_only_gemma2_attention_sliding_window_mask(gemma2_config):
    """Test sliding window masking restricts attention when mask is provided."""
    gemma2_config.sliding_window = 2
    attn = QueryOnlyGemma2Attention(gemma2_config, layer_idx=0)

    batch_sz, seq_len = 1, 5
    hidden_states = torch.randn(batch_sz, seq_len, 64)

    local_kv = torch.randn(batch_sz, seq_len, 2, 2, 16)
    global_kv = torch.randn(batch_sz, seq_len, 2, 2, 16)

    cos = torch.randn(batch_sz, seq_len, 16)
    sin = torch.randn(batch_sz, seq_len, 16)

    attention_mask = torch.zeros(batch_sz, 1, seq_len, seq_len)
    causal_mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool))
    attention_mask = attention_mask.masked_fill(
        ~causal_mask.view(1, 1, seq_len, seq_len), torch.finfo(hidden_states.dtype).min
    )

    output1, _ = attn(
        hidden_states=hidden_states,
        position_embeddings=(cos, sin),
        attention_mask=attention_mask,
        verifier_kv_last_local=local_kv,
        verifier_kv_last_global=global_kv,
    )

    local_kv_changed = local_kv.clone()
    local_kv_changed[:, 0, ...] = torch.randn_like(local_kv[:, 0, ...])

    output2, _ = attn(
        hidden_states=hidden_states,
        position_embeddings=(cos, sin),
        attention_mask=attention_mask,
        verifier_kv_last_local=local_kv_changed,
        verifier_kv_last_global=global_kv,
    )

    assert torch.equal(output1[:, 4], output2[:, 4])
    assert not torch.equal(output1[:, 0], output2[:, 0])


def test_mtp_draft_model_gemma2_kv_slicing():
    """Test MTPDraftModel.forward slices verifier KV caches to valid_len."""
    tc = Gemma2Config(
        hidden_size=16,
        vocab_size=256,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        head_dim=8,
    )
    tc._attn_implementation = "eager"
    config = MTPSpeculatorConfig(
        transformer_layer_config=tc,
        speculators_config=SpeculatorsConfig(
            algorithm="mtp",
            proposal_methods=[GreedyTokenProposalConfig(speculative_tokens=2)],
            default_proposal_method="greedy",
            verifier=VerifierConfig(
                architectures=["Gemma2ForCausalLM"], name_or_path="dummy"
            ),
        ),
    )

    model = MTPDraftModel(config)
    with torch.no_grad():
        model.embed_tokens.weight.normal_(0, 0.02)
        model.lm_head.weight.normal_(0, 0.02)

    valid_len = 6
    kv_len = 10
    input_ids = torch.randint(0, 256, (1, valid_len))
    hidden_states = torch.randn(1, valid_len, 16)
    local_kv = torch.randn(1, kv_len, 2, 2, 8)
    global_kv = torch.randn(1, kv_len, 2, 2, 8)

    draft_tokens, loss, _ = model(
        input_ids=input_ids,
        hidden_states=hidden_states,
        verifier_kv_last_local=local_kv,
        verifier_kv_last_global=global_kv,
    )
    assert len(draft_tokens) == 2
    assert torch.isfinite(loss)
