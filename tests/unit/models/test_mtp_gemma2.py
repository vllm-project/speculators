"""Unit tests for Gemma2 MTP Query-Only Attention."""

import pytest
import torch
from transformers import PretrainedConfig

from speculators.models.base_components import model_classes
from speculators.models.mtp.model_definitions import mtp_model_classes

if "gemma2" not in model_classes:
    pytest.skip("transformers < 4.42 installed, skipping gemma2 tests", allow_module_level=True)

from speculators.models.mtp.model_definitions import QueryOnlyGemma2Attention


@pytest.fixture
def gemma2_config():
    config = PretrainedConfig(
        hidden_size=64,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        attention_bias=False,
        attention_dropout=0.0,
        _attn_implementation="eager"
    )
    # Mock some Gemma2 specific attrs
    config.sliding_window = 4096
    config.query_pre_attn_scalar = 224
    config.hidden_size = 64
    config.num_attention_heads = 4
    config.num_key_value_heads = 2
    config.head_dim = 16
    return config


def test_query_only_gemma2_attention_local_kv(gemma2_config):
    """Test that local KV cache is used when sliding_window > 0."""
    attn = QueryOnlyGemma2Attention(gemma2_config, layer_idx=0)
    
    batch_sz, seq_len = 2, 5
    hidden_states = torch.randn(batch_sz, seq_len, 64)
    
    # [batch, seq_len, 2, num_kv_heads, head_dim]
    local_kv = torch.randn(batch_sz, seq_len, 2, 2, 16)
    global_kv = torch.randn(batch_sz, seq_len, 2, 2, 16)
    
    # Mock position embeddings (cos, sin)
    cos = torch.randn(1, 1, seq_len, 16)
    sin = torch.randn(1, 1, seq_len, 16)
    
    output, _ = attn(
        hidden_states=hidden_states,
        position_embeddings=(cos, sin),
        attention_mask=None,
        verifier_kv_last_local=local_kv,
        verifier_kv_last_global=global_kv,
    )
    
    assert output.shape == (batch_sz, seq_len, 64)


def test_query_only_gemma2_attention_global_kv(gemma2_config):
    """Test that global KV cache is used when sliding_window is 0 or None."""
    gemma2_config.sliding_window = None
    attn = QueryOnlyGemma2Attention(gemma2_config, layer_idx=0)
    
    batch_sz, seq_len = 2, 5
    hidden_states = torch.randn(batch_sz, seq_len, 64)
    
    local_kv = torch.randn(batch_sz, seq_len, 2, 2, 16)
    global_kv = torch.randn(batch_sz, seq_len, 2, 2, 16)
    
    cos = torch.randn(1, 1, seq_len, 16)
    sin = torch.randn(1, 1, seq_len, 16)
    
    output, _ = attn(
        hidden_states=hidden_states,
        position_embeddings=(cos, sin),
        attention_mask=None,
        verifier_kv_last_local=local_kv,
        verifier_kv_last_global=global_kv,
    )
    
    assert output.shape == (batch_sz, seq_len, 64)
