"""Tests for loading pretrained EAGLE3 models."""

import pytest
import torch

from speculators.utils.loading import extract_vocab_mappings, load_full_state_dict


@pytest.mark.parametrize(
    "model_id",
    [
        "RedHatAI/Llama-3.1-8B-Instruct-speculator.eagle3",
    ],
)
def test_load_pretrained_from_hub(model_id):
    """Test loading a pretrained EAGLE3 model from HF Hub."""
    # Load state dict
    state_dict = load_full_state_dict(model_id)

    # Verify structure
    assert isinstance(state_dict, dict)
    assert len(state_dict) > 0

    # Check for key components
    d2t_keys = [k for k in state_dict if "d2t" in k.lower()]
    t2d_keys = [k for k in state_dict if "t2d" in k.lower()]

    assert len(d2t_keys) > 0, "d2t mapping not found"
    assert len(t2d_keys) > 0, "t2d mapping not found"

    # Extract mappings
    device = torch.device("cpu")
    d2t, t2d = extract_vocab_mappings(state_dict, device)

    # Validate shapes
    assert d2t.dim() in [1, 2]
    assert t2d.dim() in [1, 2]
    assert d2t.device == device
    assert t2d.device == device


def test_extract_vocab_mappings_missing_d2t():
    """Test error handling when d2t mapping is missing."""
    state_dict = {
        "model.layer1.weight": torch.randn(10, 10),
        "model.t2d": torch.randn(100),
    }

    with pytest.raises(ValueError, match="No 'd2t' key found"):
        extract_vocab_mappings(state_dict, torch.device("cpu"))


def test_extract_vocab_mappings_missing_t2d():
    """Test error handling when t2d mapping is missing."""
    state_dict = {
        "model.layer1.weight": torch.randn(10, 10),
        "model.d2t": torch.randn(100),
    }

    with pytest.raises(ValueError, match="No 't2d' key found"):
        extract_vocab_mappings(state_dict, torch.device("cpu"))


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
    assert "d2t" not in state_dict  # Should be removed
    assert "t2d" not in state_dict  # Should be removed


def test_extract_vocab_mappings_invalid_shape():
    """Test error handling for invalid tensor shapes."""
    state_dict = {
        "d2t": torch.randn(10, 10, 10),  # 3D tensor (invalid)
        "t2d": torch.randn(128),
    }

    with pytest.raises(ValueError, match="Unexpected d2t shape"):
        extract_vocab_mappings(state_dict, torch.device("cpu"))
