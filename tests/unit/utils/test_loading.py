"""
Unit tests for the loading module in the Speculators library.
"""

import json
from unittest.mock import MagicMock

import pytest
import torch
from safetensors.torch import save_file
from transformers import AutoModelForCausalLM

from speculators.utils.loading import (
    _find_exact_key,
    _resolve_file,
    _validate_mapping_tensor,
    extract_vocab_mappings,
    load_full_state_dict,
    load_model_layers,
    load_pretrained_weights,
)

# Test model from HuggingFace
TEST_MODEL_REPO = "nm-testing/tiny-testing-random-weights"
SMALL_MODEL_REPO = "nm-testing/tinysmokellama-3.2"

# _resolve_file Tests


@pytest.mark.sanity
def test_resolve_file_hub_download():
    """Test resolving a file from HuggingFace Hub using real model."""
    result = _resolve_file(TEST_MODEL_REPO, "config.json")

    assert result.exists()
    assert result.name == "config.json"


# load_model_layers Tests


@pytest.mark.sanity
@pytest.mark.parametrize(
    "test_model_repo",
    [
        TEST_MODEL_REPO,  # Multi-shard model
        SMALL_MODEL_REPO,  # Single-shard model
    ],
)
def test_load_model(test_model_repo: str):
    """Test loading layers from a model repository."""
    result = load_model_layers(
        ["model.embed_tokens.weight", "lm_head.weight"],
        test_model_repo,
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


@pytest.mark.sanity
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


# extract_vocab_mappings Tests


def test_extract_vocab_mappings_success():
    """Test successful extraction of vocabulary mappings."""
    state_dict = {
        "model.layer1.weight": torch.randn(10, 10),
        "d2t": torch.randn(100),
        "t2d": torch.randn(128),
    }

    device = torch.device("cpu")
    d2t, t2d = extract_vocab_mappings(state_dict, device)

    # Verify extraction
    assert d2t.shape == (100,)
    assert t2d.shape == (128,)
    assert d2t.device == device
    assert t2d.device == device
    # Should be removed from state dict
    assert "d2t" not in state_dict
    assert "t2d" not in state_dict
    # Other keys should remain
    assert "model.layer1.weight" in state_dict


def test_extract_vocab_mappings_missing_d2t():
    """Test error handling when d2t mapping is missing."""
    state_dict = {
        "model.layer1.weight": torch.randn(10, 10),
        "t2d": torch.randn(100),
    }

    with pytest.raises(ValueError, match="Key 'd2t' not found"):
        extract_vocab_mappings(state_dict, torch.device("cpu"))


def test_extract_vocab_mappings_missing_t2d():
    """Test error handling when t2d mapping is missing."""
    state_dict = {
        "model.layer1.weight": torch.randn(10, 10),
        "d2t": torch.randn(100),
    }

    with pytest.raises(ValueError, match="Key 't2d' not found"):
        extract_vocab_mappings(state_dict, torch.device("cpu"))


def test_extract_vocab_mappings_invalid_shape():
    """Test error handling for invalid tensor shapes."""
    state_dict = {
        "d2t": torch.randn(10, 10, 10),  # 3D tensor (invalid)
        "t2d": torch.randn(128),
    }

    with pytest.raises(ValueError, match="Unexpected d2t shape"):
        extract_vocab_mappings(state_dict, torch.device("cpu"))


def test_extract_vocab_mappings_case_insensitive():
    """Test case-insensitive key matching."""
    state_dict = {
        "D2T": torch.randn(100),
        "T2D": torch.randn(128),
    }

    device = torch.device("cpu")
    d2t, t2d = extract_vocab_mappings(state_dict, device)

    assert d2t.shape == (100,)
    assert t2d.shape == (128,)
    # Keys should be removed (case-insensitive)
    assert "D2T" not in state_dict
    assert "T2D" not in state_dict


def test_extract_vocab_mappings_2d_tensors():
    """Test error handling for 2D vocab mapping tensors."""
    state_dict = {
        "d2t": torch.randn(100, 1),  # 2D tensor (invalid for 1D lookup)
        "t2d": torch.randn(128, 1),  # 2D tensor (invalid for 1D mask)
    }

    device = torch.device("cpu")
    with pytest.raises(ValueError, match="Unexpected d2t shape"):
        extract_vocab_mappings(state_dict, device)


# _find_exact_key Tests


def test_find_exact_key_exact_match():
    """Test finding exact key in state dict."""
    state_dict = {
        "d2t": torch.randn(100),
        "model.layer.weight": torch.randn(10, 10),
    }

    result = _find_exact_key(state_dict, "d2t")
    assert result == "d2t"


def test_find_exact_key_case_insensitive():
    """Test case-insensitive key matching."""
    state_dict = {
        "D2T": torch.randn(100),
        "model.layer.weight": torch.randn(10, 10),
    }

    result = _find_exact_key(state_dict, "d2t")
    assert result == "D2T"


def test_find_exact_key_not_found():
    """Test error when key not found."""
    state_dict = {
        "model.layer.weight": torch.randn(10, 10),
        "other.weight": torch.randn(5, 5),
    }

    with pytest.raises(ValueError, match="Key 'd2t' not found"):
        _find_exact_key(state_dict, "d2t")


def test_find_exact_key_error_message_quality():
    """Test that error message shows available keys."""
    state_dict = {
        "model.layer1.weight": torch.randn(10, 10),
        "model.layer2.weight": torch.randn(10, 10),
    }

    with pytest.raises(ValueError) as exc_info:
        _find_exact_key(state_dict, "missing_key")

    error_msg = str(exc_info.value)
    assert "missing_key" in error_msg
    assert "Available keys" in error_msg


# _validate_mapping_tensor Tests


def test_validate_mapping_tensor_1d():
    """Test validation of 1D tensors."""
    tensor = torch.randn(100)
    # Should not raise
    _validate_mapping_tensor(tensor, "test_tensor")


def test_validate_mapping_tensor_2d():
    """Test that 2D tensors are rejected."""
    tensor = torch.randn(100, 128)

    with pytest.raises(ValueError, match="Unexpected test_tensor shape"):
        _validate_mapping_tensor(tensor, "test_tensor")


def test_validate_mapping_tensor_3d_fails():
    """Test that 3D tensors are rejected."""
    tensor = torch.randn(10, 10, 10)

    with pytest.raises(ValueError, match="Unexpected test_tensor shape"):
        _validate_mapping_tensor(tensor, "test_tensor")


def test_validate_mapping_tensor_0d_fails():
    """Test that scalar tensors are rejected."""
    tensor = torch.tensor(5.0)

    with pytest.raises(ValueError, match="Unexpected test_tensor shape"):
        _validate_mapping_tensor(tensor, "test_tensor")


# load_full_state_dict Tests


def test_load_full_state_dict_single_file(tmp_path):
    """Test loading from single safetensors file."""
    # Create a mock safetensors file
    state_dict = {
        "layer1.weight": torch.randn(10, 10),
        "layer2.bias": torch.randn(10),
        "d2t": torch.randn(100),
        "t2d": torch.randn(128),
    }

    model_file = tmp_path / "model.safetensors"
    save_file(state_dict, str(model_file))

    # Load using our function
    loaded_dict = load_full_state_dict(str(tmp_path))

    # Verify all keys are present
    assert len(loaded_dict) == len(state_dict)
    for key, value in state_dict.items():
        assert key in loaded_dict
        assert torch.equal(loaded_dict[key], value)


def test_load_full_state_dict_sharded(tmp_path):
    """Test loading from sharded safetensors files."""
    # Create mock sharded files
    shard1_dict = {
        "layer1.weight": torch.randn(10, 10),
        "d2t": torch.randn(100),
    }
    shard2_dict = {
        "layer2.weight": torch.randn(20, 20),
        "t2d": torch.randn(128),
    }

    shard1_file = tmp_path / "model-00001-of-00002.safetensors"
    shard2_file = tmp_path / "model-00002-of-00002.safetensors"

    save_file(shard1_dict, str(shard1_file))
    save_file(shard2_dict, str(shard2_file))

    # Create index file
    index = {
        "metadata": {"total_size": 12345},
        "weight_map": {
            "layer1.weight": "model-00001-of-00002.safetensors",
            "d2t": "model-00001-of-00002.safetensors",
            "layer2.weight": "model-00002-of-00002.safetensors",
            "t2d": "model-00002-of-00002.safetensors",
        },
    }

    index_file = tmp_path / "model.safetensors.index.json"
    with index_file.open("w") as f:
        json.dump(index, f)

    # Load using our function
    loaded_dict = load_full_state_dict(str(tmp_path))

    # Verify all keys from both shards are present
    assert len(loaded_dict) == 4
    assert "layer1.weight" in loaded_dict
    assert "layer2.weight" in loaded_dict
    assert "d2t" in loaded_dict
    assert "t2d" in loaded_dict

    # Verify values
    assert torch.equal(loaded_dict["layer1.weight"], shard1_dict["layer1.weight"])
    assert torch.equal(loaded_dict["d2t"], shard1_dict["d2t"])
    assert torch.equal(loaded_dict["layer2.weight"], shard2_dict["layer2.weight"])
    assert torch.equal(loaded_dict["t2d"], shard2_dict["t2d"])


def test_load_full_state_dict_file_not_found(tmp_path):
    """Test error when no model files are present in an existing local directory."""
    empty_dir = tmp_path / "empty_model_dir"
    empty_dir.mkdir()

    with pytest.raises(FileNotFoundError):
        load_full_state_dict(str(empty_dir))


# load_pretrained_weights Tests


def test_load_pretrained_weights_success():
    """Test loading weights into model."""
    mock_model = MagicMock()
    mock_model.load_state_dict.return_value = ([], [])
    state_dict = {"layer.weight": torch.randn(10, 10)}

    load_pretrained_weights(mock_model, state_dict, "/path/to/model")
    mock_model.load_state_dict.assert_called_once()


def test_load_pretrained_weights_with_expected_missing():
    """Test loading with expected missing keys."""
    mock_model = MagicMock()
    expected_missing = ["d2t", "t2d", "verifier_lm_head.weight"]
    mock_model.load_state_dict.return_value = (expected_missing, [])

    load_pretrained_weights(mock_model, {"layer.weight": torch.randn(10, 10)}, "/test")
    # Should not raise
