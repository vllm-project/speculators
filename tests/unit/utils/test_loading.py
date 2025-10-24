"""
Unit tests for the loading module in the Speculators library.
"""

import pytest
import torch
from transformers import AutoModelForCausalLM

from speculators.utils.loading import _resolve_file, load_model_layers

# Test model from HuggingFace
TEST_MODEL_REPO = "nm-testing/tiny-testing-random-weights"

# _resolve_file Tests


@pytest.mark.regression
def test_resolve_file_hub_download():
    """Test resolving a file from HuggingFace Hub using real model."""
    result = _resolve_file(TEST_MODEL_REPO, "config.json")

    assert result.exists()
    assert result.name == "config.json"


# load_model_layers Tests


@pytest.mark.regression
def test_load_model_layers_multiple_shards():
    """Test loading layers from multiple safetensors shards."""
    # embed_tokens is in shard 1, lm_head is in shard 2
    result = load_model_layers(
        ["model.embed_tokens.weight", "lm_head.weight"],
        TEST_MODEL_REPO,
    )

    assert len(result) == 2
    assert "model.embed_tokens.weight" in result
    assert "lm_head.weight" in result
    assert isinstance(result["model.embed_tokens.weight"], torch.Tensor)
    assert isinstance(result["lm_head.weight"], torch.Tensor)
    # Both should have same vocab dimension
    assert (
        result["model.embed_tokens.weight"].shape[0]
        == result["lm_head.weight"].shape[0]
    )
    # Verify CPU device
    assert result["model.embed_tokens.weight"].device.type == "cpu"


@pytest.mark.regression
def test_load_model_layers_matches_full_model():
    """Test that tensors loaded via utility match those from full model loading."""
    # Load full model
    full_model = AutoModelForCausalLM.from_pretrained(
        TEST_MODEL_REPO,
        torch_dtype="auto",
    )

    # Get state dict from full model
    state_dict = full_model.state_dict()

    # Load specific layers using our utility
    layer_names = [
        "model.embed_tokens.weight",
        "lm_head.weight",
        "model.norm.weight",
        "model.layers.0.input_layernorm.weight",
        "model.layers.0.mlp.gate_proj.weight",
        "model.layers.1.mlp.down_proj.weight",
    ]

    loaded_tensors = load_model_layers(layer_names, TEST_MODEL_REPO)

    # Compare each tensor
    for layer_name in layer_names:
        assert layer_name in loaded_tensors, f"Layer {layer_name} not loaded"
        assert layer_name in state_dict, f"Layer {layer_name} not in state_dict"

        util_tensor = loaded_tensors[layer_name]
        model_tensor = state_dict[layer_name]

        # Check dtype matches
        assert util_tensor.dtype == model_tensor.dtype, (
            f"Dtype mismatch for {layer_name}: "
            f"{util_tensor.dtype} vs {model_tensor.dtype}"
        )

        # Check shape matches
        assert util_tensor.shape == model_tensor.shape, (
            f"Shape mismatch for {layer_name}: "
            f"{util_tensor.shape} vs {model_tensor.shape}"
        )

        # Check values are identical
        assert torch.equal(util_tensor, model_tensor), (
            f"Tensor values don't match for {layer_name}"
        )

    # Clean up
    del full_model
    del state_dict
