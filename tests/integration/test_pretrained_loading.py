"""Integration tests for pretrained speculator model loading from Hugging Face Hub.

These tests use EAGLE3 as the reference implementation, but the loading utilities
are designed to work with any registered speculator algorithm.
"""

import pytest
import torch

from speculators.config import SpeculatorModelConfig
from speculators.model import SpeculatorModel
from speculators.utils.loading import extract_vocab_mappings, load_full_state_dict

# Real model from HF Hub for integration testing (using EAGLE3 as reference)
# This model uses Qwen/Qwen3-8B as verifier, which is publicly accessible
TEST_MODEL_ID = "RedHatAI/Qwen3-8B-speculator.eagle3"


@pytest.mark.integration
def test_load_full_state_dict_from_hub():
    """Test loading complete state dict from HF Hub and expected key structure."""
    state_dict = load_full_state_dict(TEST_MODEL_ID)

    assert isinstance(state_dict, dict)
    assert len(state_dict) > 0
    expected_patterns = ["d2t", "t2d", "layers"]
    for pattern in expected_patterns:
        assert any(
            pattern in k.lower() for k in state_dict
        ), f"Missing pattern: {pattern}"
    for key, tensor in state_dict.items():
        assert tensor.device.type == "cpu", f"Tensor {key} not on CPU: {tensor.device}"


@pytest.mark.integration
def test_extract_vocab_mappings_from_real_model():
    """Test extracting d2t/t2d from real speculator model and in-place removal."""
    state_dict = load_full_state_dict(TEST_MODEL_ID)
    keys_before = set(state_dict.keys())
    device = torch.device("cpu")
    d2t, t2d = extract_vocab_mappings(state_dict, device)

    assert d2t.dim() == 1
    assert t2d.dim() == 1
    assert d2t.device == device
    assert t2d.device == device
    assert d2t.shape[0] > 0
    assert not any("d2t" in k.lower() for k in state_dict)
    assert not any("t2d" in k.lower() for k in state_dict)
    assert len(keys_before - set(state_dict.keys())) == 2


@pytest.mark.integration
def test_load_config_from_pretrained():
    """Test loading SpeculatorModelConfig from pretrained."""
    config = SpeculatorModelConfig.from_pretrained(TEST_MODEL_ID)
    assert config.draft_vocab_size > 0
    assert config.transformer_layer_config is not None
    tc = config.transformer_layer_config
    for attr in ("hidden_size", "num_hidden_layers", "num_attention_heads"):
        assert hasattr(tc, attr)
        assert getattr(tc, attr) > 0


@pytest.mark.integration
@pytest.mark.slow
def test_end_to_end_pretrained_loading():
    """Test full workflow: load state dict → extract mappings → create model → load."""
    state_dict = load_full_state_dict(TEST_MODEL_ID)
    device = torch.device("cpu")
    d2t, t2d = extract_vocab_mappings(state_dict, device)
    config = SpeculatorModelConfig.from_pretrained(TEST_MODEL_ID)

    assert config.draft_vocab_size == d2t.shape[0]
    model_class = SpeculatorModel.registered_model_class_from_config(config)
    model = model_class(config=config, t2d=t2d, d2t=d2t)
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)

    expected_missing = {"t2d", "d2t", "verifier_lm_head.weight"}
    assert [k for k in missing_keys if k not in expected_missing] == []
    assert len(unexpected_keys) == 0
    model.eval()
    assert hasattr(model, "layers")
    assert hasattr(model, "d2t")
    assert hasattr(model, "t2d")


