"""Unit tests for pretrained model loading functionality in train.py"""

import argparse
import logging
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from scripts.train import (
    initialize_vocab_config,
    load_model_weights,
    load_pretrained_model,
    load_vocab_mappings,
)

# Test load_vocab_mappings()


def test_load_vocab_mappings_success(tmp_path):
    """Test loading d2t/t2d from numpy files."""
    # Create temporary numpy files
    d2t_data = np.arange(100)
    t2d_data = np.arange(128)

    d2t_path = tmp_path / "d2t.npy"
    t2d_path = tmp_path / "t2d.npy"

    np.save(d2t_path, d2t_data)
    np.save(t2d_path, t2d_data)

    # Test loading
    device = torch.device("cpu")
    d2t, t2d, vocab_size = load_vocab_mappings(str(d2t_path), str(t2d_path), device)

    assert d2t.shape == (100,)
    assert t2d.shape == (128,)
    assert vocab_size == 100
    assert d2t.device == device
    assert t2d.device == device
    # Verify values match numpy arrays
    assert torch.equal(d2t, torch.from_numpy(d2t_data))
    assert torch.equal(t2d, torch.from_numpy(t2d_data))


def test_load_vocab_mappings_missing_d2t():
    """Test error when only t2d provided."""
    with pytest.raises(ValueError, match="Both d2t and t2d paths must be provided"):
        load_vocab_mappings(None, "/path/to/t2d.npy", torch.device("cpu"))


def test_load_vocab_mappings_missing_t2d():
    """Test error when only d2t provided."""
    with pytest.raises(ValueError, match="Both d2t and t2d paths must be provided"):
        load_vocab_mappings("/path/to/d2t.npy", None, torch.device("cpu"))


def test_load_vocab_mappings_both_missing():
    """Test error when both paths are None."""
    with pytest.raises(ValueError, match="Both d2t and t2d paths must be provided"):
        load_vocab_mappings(None, None, torch.device("cpu"))


def test_load_vocab_mappings_empty_strings():
    """Test error when empty strings are provided."""
    with pytest.raises(ValueError, match="Both d2t and t2d paths must be provided"):
        load_vocab_mappings("", "/path/to/t2d.npy", torch.device("cpu"))


def test_load_vocab_mappings_file_not_found(tmp_path):
    """Test error when numpy files don't exist."""
    d2t_path = tmp_path / "nonexistent_d2t.npy"
    t2d_path = tmp_path / "nonexistent_t2d.npy"

    with pytest.raises(FileNotFoundError):
        load_vocab_mappings(str(d2t_path), str(t2d_path), torch.device("cpu"))


def test_load_vocab_mappings_2d_arrays(tmp_path):
    """Test loading 2D vocab mapping arrays."""
    # Create 2D arrays
    d2t_data = np.arange(200).reshape(100, 2)
    t2d_data = np.arange(256).reshape(128, 2)

    d2t_path = tmp_path / "d2t.npy"
    t2d_path = tmp_path / "t2d.npy"

    np.save(d2t_path, d2t_data)
    np.save(t2d_path, t2d_data)

    # Test loading
    device = torch.device("cpu")
    d2t, t2d, vocab_size = load_vocab_mappings(str(d2t_path), str(t2d_path), device)

    assert d2t.shape == (100, 2)
    assert t2d.shape == (128, 2)
    assert vocab_size == 100  # First dimension is vocab size


# Test load_pretrained_model()


@patch("scripts.train.load_full_state_dict")
@patch("scripts.train.extract_vocab_mappings")
def test_load_pretrained_model_success(mock_extract, mock_load_state):
    """Test loading pretrained model."""
    # Mock state dict
    mock_state_dict = {
        "layer.weight": torch.randn(10, 10),
        "d2t": torch.randn(100),
        "t2d": torch.randn(128),
    }
    mock_load_state.return_value = mock_state_dict.copy()

    # Mock vocab mappings
    mock_d2t = torch.arange(100)
    mock_t2d = torch.arange(128)
    mock_extract.return_value = (mock_d2t, mock_t2d)

    # Test
    device = torch.device("cpu")
    state_dict, d2t, t2d, vocab_size = load_pretrained_model("/path/to/model", device)

    # Verify calls
    mock_load_state.assert_called_once_with("/path/to/model")
    mock_extract.assert_called_once()

    # Verify returns
    assert torch.equal(d2t, mock_d2t)
    assert torch.equal(t2d, mock_t2d)
    assert vocab_size == 100


@patch("scripts.train.load_full_state_dict")
@patch("scripts.train.extract_vocab_mappings")
def test_load_pretrained_model_vocab_size_derivation(mock_extract, mock_load_state):
    """Test that vocab_size is correctly derived from d2t shape."""
    mock_state_dict = {"layer.weight": torch.randn(10, 10)}
    mock_load_state.return_value = mock_state_dict

    # Test with different vocab sizes
    for vocab_size in [100, 256, 1024]:
        mock_d2t = torch.arange(vocab_size)
        mock_t2d = torch.arange(vocab_size * 2)
        mock_extract.return_value = (mock_d2t, mock_t2d)

        _, _, _, derived_vocab_size = load_pretrained_model(
            "/path/to/model", torch.device("cpu")
        )

        assert derived_vocab_size == vocab_size


# Test initialize_vocab_config()


def test_initialize_vocab_config_conflicting_args():
    """Test error when both pretrained and d2t/t2d provided."""
    args = argparse.Namespace(
        pretrained_model_path="/path/to/model",
        d2t_path="/path/to/d2t.npy",
        t2d_path="/path/to/t2d.npy",
        verifier_name_or_path="meta-llama/Llama-2-7b",
    )

    with pytest.raises(
        ValueError,
        match="--pretrained-model-path overrides --d2t-path",
    ):
        initialize_vocab_config(args, torch.device("cpu"))


def test_initialize_vocab_config_partial_conflict_d2t_only():
    """Test error when pretrained + only d2t provided."""
    args = argparse.Namespace(
        pretrained_model_path="/path/to/model",
        d2t_path="/path/to/d2t.npy",
        t2d_path=None,
        verifier_name_or_path="meta-llama/Llama-2-7b",
    )

    with pytest.raises(
        ValueError,
        match="--pretrained-model-path overrides --d2t-path",
    ):
        initialize_vocab_config(args, torch.device("cpu"))


def test_initialize_vocab_config_partial_conflict_t2d_only():
    """Test error when pretrained + only t2d provided."""
    args = argparse.Namespace(
        pretrained_model_path="/path/to/model",
        d2t_path=None,
        t2d_path="/path/to/t2d.npy",
        verifier_name_or_path="meta-llama/Llama-2-7b",
    )

    with pytest.raises(
        ValueError,
        match="--pretrained-model-path overrides --d2t-path",
    ):
        initialize_vocab_config(args, torch.device("cpu"))


@patch("scripts.train.load_pretrained_model")
def test_initialize_vocab_config_from_pretrained(mock_load):
    """Test initialization from pretrained model."""
    # Mock return
    mock_state = {"layer": torch.randn(10, 10)}
    mock_d2t = torch.arange(100)
    mock_t2d = torch.arange(128)
    mock_load.return_value = (mock_state, mock_d2t, mock_t2d, 100)

    args = argparse.Namespace(
        pretrained_model_path="/path/to/model",
        d2t_path=None,
        t2d_path=None,
        verifier_name_or_path="meta-llama/Llama-2-7b",
    )

    d2t, t2d, vocab_size, state_dict = initialize_vocab_config(
        args, torch.device("cpu")
    )

    assert vocab_size == 100
    assert state_dict == mock_state
    assert torch.equal(d2t, mock_d2t)
    assert torch.equal(t2d, mock_t2d)
    mock_load.assert_called_once_with("/path/to/model", torch.device("cpu"))


@patch("scripts.train.load_vocab_mappings")
def test_initialize_vocab_config_from_numpy(mock_load_vocab):
    """Test initialization from numpy files."""
    mock_d2t = torch.arange(100)
    mock_t2d = torch.arange(128)
    mock_load_vocab.return_value = (mock_d2t, mock_t2d, 100)

    args = argparse.Namespace(
        pretrained_model_path=None,
        d2t_path="/path/to/d2t.npy",
        t2d_path="/path/to/t2d.npy",
        verifier_name_or_path="meta-llama/Llama-2-7b",
    )

    d2t, t2d, vocab_size, state_dict = initialize_vocab_config(
        args, torch.device("cpu")
    )

    assert vocab_size == 100
    assert state_dict is None
    assert torch.equal(d2t, mock_d2t)
    assert torch.equal(t2d, mock_t2d)
    mock_load_vocab.assert_called_once_with(
        "/path/to/d2t.npy", "/path/to/t2d.npy", torch.device("cpu")
    )


@patch("scripts.train.AutoConfig")
def test_initialize_vocab_config_no_vocab_mapping(mock_config_class):
    """Test initialization without vocab mappings (uses verifier vocab size)."""
    # Mock verifier config without text_config attribute
    mock_verifier_config = MagicMock(spec=["vocab_size"])
    mock_verifier_config.vocab_size = 32000
    mock_config_class.from_pretrained.return_value = mock_verifier_config

    args = argparse.Namespace(
        pretrained_model_path=None,
        d2t_path=None,
        t2d_path=None,
        verifier_name_or_path="meta-llama/Llama-2-7b",
    )

    d2t, t2d, vocab_size, state_dict = initialize_vocab_config(
        args, torch.device("cpu")
    )

    assert d2t is None
    assert t2d is None
    assert vocab_size == 32000
    assert state_dict is None
    mock_config_class.from_pretrained.assert_called_once_with("meta-llama/Llama-2-7b")


@patch("scripts.train.AutoConfig")
def test_initialize_vocab_config_multimodal_verifier(mock_config_class):
    """Test initialization with multimodal verifier (has text_config)."""
    # Mock multimodal config with text_config
    mock_text_config = MagicMock()
    mock_text_config.vocab_size = 32000

    mock_verifier_config = MagicMock()
    mock_verifier_config.text_config = mock_text_config

    mock_config_class.from_pretrained.return_value = mock_verifier_config

    args = argparse.Namespace(
        pretrained_model_path=None,
        d2t_path=None,
        t2d_path=None,
        verifier_name_or_path="Qwen/Qwen2-VL-7B",
    )

    d2t, t2d, vocab_size, state_dict = initialize_vocab_config(
        args, torch.device("cpu")
    )

    assert vocab_size == 32000


# Test load_model_weights()


def test_load_model_weights_success(caplog):
    """Test loading weights into model."""
    # Set logging level to capture INFO logs
    caplog.set_level(logging.INFO)

    # Create mock model
    mock_model = MagicMock()
    mock_model.load_state_dict.return_value = ([], [])  # no missing/unexpected keys

    state_dict = {"layer.weight": torch.randn(10, 10)}

    load_model_weights(mock_model, state_dict, "/path/to/model")

    mock_model.load_state_dict.assert_called_once_with(state_dict, strict=False)
    assert "✓ Successfully loaded all weights" in caplog.text


def test_load_model_weights_with_expected_missing_keys(caplog):
    """Test loading with expected missing keys (d2t, t2d, lm_head)."""
    # Set logging level to capture INFO logs
    caplog.set_level(logging.INFO)

    mock_model = MagicMock()
    expected_missing = ["d2t", "t2d", "verifier_lm_head.weight"]
    mock_model.load_state_dict.return_value = (expected_missing, [])

    state_dict = {"layer.weight": torch.randn(10, 10)}

    load_model_weights(mock_model, state_dict, "/path/to/model")

    # Should not warn about expected missing keys
    assert "Unexpected missing keys" not in caplog.text
    assert "✓ Successfully loaded all weights" in caplog.text


def test_load_model_weights_unexpected_missing_keys(caplog):
    """Test warning for unexpected missing keys."""
    # Set logging level to capture WARNING logs
    caplog.set_level(logging.WARNING)

    mock_model = MagicMock()
    missing_keys = ["d2t", "unexpected_layer.weight"]
    mock_model.load_state_dict.return_value = (missing_keys, [])

    state_dict = {"layer.weight": torch.randn(10, 10)}

    load_model_weights(mock_model, state_dict, "/path/to/model")

    assert "Unexpected missing keys" in caplog.text
    assert "unexpected_layer.weight" in caplog.text


def test_load_model_weights_unexpected_keys(caplog):
    """Test warning for unexpected keys in state dict."""
    # Set logging level to capture WARNING logs
    caplog.set_level(logging.WARNING)

    mock_model = MagicMock()
    unexpected_keys = ["extra_layer.weight"]
    mock_model.load_state_dict.return_value = ([], unexpected_keys)

    state_dict = {"layer.weight": torch.randn(10, 10)}

    load_model_weights(mock_model, state_dict, "/path/to/model")

    assert "Unexpected keys" in caplog.text
    assert "extra_layer.weight" in caplog.text


def test_load_model_weights_both_issues(caplog):
    """Test warnings when both missing and unexpected keys present."""
    # Set logging level to capture WARNING logs
    caplog.set_level(logging.WARNING)

    mock_model = MagicMock()
    missing_keys = ["d2t", "missing_layer.weight"]
    unexpected_keys = ["extra_layer.weight"]
    mock_model.load_state_dict.return_value = (missing_keys, unexpected_keys)

    state_dict = {"layer.weight": torch.randn(10, 10)}

    load_model_weights(mock_model, state_dict, "/path/to/model")

    assert "Unexpected missing keys" in caplog.text
    assert "missing_layer.weight" in caplog.text
    assert "Unexpected keys" in caplog.text
    assert "extra_layer.weight" in caplog.text
    assert "completed with warnings" in caplog.text


def test_load_model_weights_logging_info(caplog):
    """Test that informational logging is present."""
    # Set logging level to capture INFO logs
    caplog.set_level(logging.INFO)

    mock_model = MagicMock()
    mock_model.load_state_dict.return_value = ([], [])

    state_dict = {
        "layer1.weight": torch.randn(10, 10),
        "layer2.weight": torch.randn(20, 20),
    }

    load_model_weights(mock_model, state_dict, "/path/to/model")

    assert "Loading pretrained weights from /path/to/model" in caplog.text
    assert "Parameters to load: 2" in caplog.text
    assert "Fine-tuning from pretrained weights" in caplog.text
    assert "Optimizer state starts fresh" in caplog.text
