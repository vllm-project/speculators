"""Integration tests for pretrained EAGLE3 model loading from Hugging Face Hub."""

import pytest
import torch

from speculators.config import SpeculatorModelConfig
from speculators.models.eagle3 import Eagle3DraftModel
from speculators.utils.loading import extract_vocab_mappings, load_full_state_dict

# Real model from HF Hub for integration testing
EAGLE3_MODEL_ID = "RedHatAI/Llama-3.1-8B-Instruct-speculator.eagle3"


@pytest.mark.integration
def test_load_full_state_dict_from_hub():
    """Test loading complete state dict from HF Hub."""
    state_dict = load_full_state_dict(EAGLE3_MODEL_ID)

    # Verify structure
    assert isinstance(state_dict, dict)
    assert len(state_dict) > 0

    # Check for EAGLE3 specific keys
    assert any("d2t" in k.lower() for k in state_dict), "d2t mapping not found"
    assert any("t2d" in k.lower() for k in state_dict), "t2d mapping not found"
    assert any("layers" in k for k in state_dict), "layers not found"

    # All tensors should be on CPU
    for key, tensor in state_dict.items():
        assert tensor.device.type == "cpu", f"Tensor {key} not on CPU: {tensor.device}"


@pytest.mark.integration
def test_extract_vocab_mappings_from_real_model():
    """Test extracting d2t/t2d from real EAGLE3 model."""
    state_dict = load_full_state_dict(EAGLE3_MODEL_ID)

    # Count keys before extraction
    keys_before = set(state_dict.keys())

    device = torch.device("cpu")
    d2t, t2d = extract_vocab_mappings(state_dict, device)

    # Validate extraction
    assert d2t.dim() == 1, f"d2t should be 1D, got {d2t.dim()}D"
    assert t2d.dim() == 1, f"t2d should be 1D, got {t2d.dim()}D"
    assert d2t.device == device
    assert t2d.device == device

    # Verify they were removed from state dict
    assert not any("d2t" in k.lower() for k in state_dict), (
        "d2t should be removed after extraction"
    )
    assert not any("t2d" in k.lower() for k in state_dict), (
        "t2d should be removed after extraction"
    )

    # Verify only d2t/t2d were removed
    keys_after = set(state_dict.keys())
    removed_keys = keys_before - keys_after
    assert len(removed_keys) == 2, (
        f"Expected 2 keys removed (d2t, t2d), got {len(removed_keys)}"
    )

    # Vocab size consistency check
    draft_vocab_size = d2t.shape[0]

    assert draft_vocab_size > 0, "draft_vocab_size should be positive"


@pytest.mark.integration
def test_load_config_from_pretrained():
    """Test loading SpeculatorModelConfig from pretrained."""
    config = SpeculatorModelConfig.from_pretrained(EAGLE3_MODEL_ID)

    # Verify config structure
    assert config.draft_vocab_size > 0, "draft_vocab_size should be positive"
    assert config.transformer_layer_config is not None, (
        "transformer_layer_config missing"
    )

    # Check transformer config attributes
    transformer_config = config.transformer_layer_config
    assert hasattr(transformer_config, "hidden_size"), (
        "hidden_size missing from transformer_config"
    )
    assert hasattr(transformer_config, "num_hidden_layers"), (
        "num_hidden_layers missing from transformer_config"
    )
    assert hasattr(transformer_config, "num_attention_heads"), (
        "num_attention_heads missing from transformer_config"
    )

    # Verify values are reasonable
    assert transformer_config.hidden_size > 0
    assert transformer_config.num_hidden_layers > 0
    assert transformer_config.num_attention_heads > 0


@pytest.mark.integration
@pytest.mark.slow
def test_end_to_end_pretrained_loading():
    """Test complete workflow: load state dict → extract mappings → create model."""
    # Load state dict
    state_dict = load_full_state_dict(EAGLE3_MODEL_ID)

    # Extract vocab mappings
    device = torch.device("cpu")
    d2t, t2d = extract_vocab_mappings(state_dict, device)

    # Load config
    config = SpeculatorModelConfig.from_pretrained(EAGLE3_MODEL_ID)

    # Verify draft_vocab_size consistency
    assert config.draft_vocab_size == d2t.shape[0], (
        f"Config vocab size {config.draft_vocab_size} != d2t shape {d2t.shape[0]}"
    )

    # Create model with extracted mappings
    model = Eagle3DraftModel(config=config, t2d=t2d, d2t=d2t)

    # Load weights (should work without errors)
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)

    # Filter expected missing keys
    expected_missing = {"t2d", "d2t", "verifier_lm_head.weight"}
    unexpected_missing = [k for k in missing_keys if k not in expected_missing]

    # Assertions
    assert len(unexpected_missing) == 0, (
        f"Unexpected missing keys: {unexpected_missing}"
    )
    assert len(unexpected_keys) == 0, f"Unexpected keys: {unexpected_keys}"

    # Model should be in eval mode and ready to use
    model.eval()
    assert model.config.draft_vocab_size == d2t.shape[0]

    # Verify model has expected attributes
    assert hasattr(model, "layers")
    assert hasattr(model, "d2t")
    assert hasattr(model, "t2d")


@pytest.mark.integration
def test_pretrained_model_key_structure():
    """Test that pretrained model has expected key structure."""
    state_dict = load_full_state_dict(EAGLE3_MODEL_ID)

    # Expected key patterns
    expected_patterns = [
        "d2t",
        "t2d",
        "layers",
    ]

    found_patterns = dict.fromkeys(expected_patterns, False)

    for key in state_dict:
        for pattern in expected_patterns:
            if pattern in key.lower():
                found_patterns[pattern] = True

    # Check all patterns found
    missing_patterns = [p for p, found in found_patterns.items() if not found]
    assert len(missing_patterns) == 0, (
        f"Missing expected patterns in state dict: {missing_patterns}"
    )
