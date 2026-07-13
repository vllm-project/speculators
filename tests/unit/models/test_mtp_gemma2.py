"""Unit tests for Gemma2 MTP Query-Only Attention."""

import pytest
import torch
from transformers.models.gemma2.configuration_gemma2 import Gemma2Config

from speculators.models.base_components import model_classes
from speculators.models.mtp.model_definitions import mtp_model_classes

if "gemma2" not in model_classes:
    pytest.skip("transformers < 4.42 installed, skipping gemma2 tests", allow_module_level=True)

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
    # Mock some Gemma2 specific attrs that we override
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
    
    # [batch, seq_len, 2, num_kv_heads, head_dim]
    local_kv = torch.randn(batch_sz, seq_len, 2, 2, 16)
    global_kv = torch.randn(batch_sz, seq_len, 2, 2, 16)
    
    # Mock position embeddings (cos, sin) - shape [batch, seq_len, head_dim]
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
    
    # Change global_kv (the inactive cache)
    global_kv_changed = torch.randn_like(global_kv)
    output2, _ = attn(
        hidden_states=hidden_states,
        position_embeddings=(cos, sin),
        attention_mask=None,
        verifier_kv_last_local=local_kv,
        verifier_kv_last_global=global_kv_changed,
    )
    assert torch.equal(output1, output2)
    
    # Change local_kv (the active cache)
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
    
    # Change local_kv (the inactive cache)
    local_kv_changed = torch.randn_like(local_kv)
    output2, _ = attn(
        hidden_states=hidden_states,
        position_embeddings=(cos, sin),
        attention_mask=None,
        verifier_kv_last_local=local_kv_changed,
        verifier_kv_last_global=global_kv,
    )
    assert torch.equal(output1, output2)
    
    # Change global_kv (the active cache)
    global_kv_changed = torch.randn_like(global_kv)
    output3, _ = attn(
        hidden_states=hidden_states,
        position_embeddings=(cos, sin),
        attention_mask=None,
        verifier_kv_last_local=local_kv,
        verifier_kv_last_global=global_kv_changed,
    )
    assert not torch.equal(output1, output3)
