"""Unit tests for pretrained model loading in scripts.train."""

import argparse
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from scripts.train import (
    initialize_vocab_config,
    load_pretrained_model,
    load_vocab_mappings,
)

# --- load_vocab_mappings ---


def test_load_vocab_mappings_success(tmp_path):
    """Load d2t/t2d from numpy files and check shapes, device, values."""
    d2t_path = tmp_path / "d2t.npy"
    t2d_path = tmp_path / "t2d.npy"
    np.save(d2t_path, np.arange(100))
    np.save(t2d_path, np.arange(128))
    device = torch.device("cpu")
    d2t, t2d, vocab_size = load_vocab_mappings(str(d2t_path), str(t2d_path), device)
    assert d2t.shape == (100,)
    assert t2d.shape == (128,)
    assert vocab_size == 100
    assert d2t.device == device
    assert t2d.device == device
    assert torch.equal(d2t, torch.arange(100))
    assert torch.equal(t2d, torch.arange(128))


@pytest.mark.parametrize(
    ("d2t_path", "t2d_path"),
    [
        (None, "/path/to/t2d.npy"),
        ("/path/to/d2t.npy", None),
        (None, None),
        ("", "/path/to/t2d.npy"),
    ],
    ids=["missing_d2t", "missing_t2d", "both_missing", "empty_d2t"],
)
def test_load_vocab_mappings_requires_both_paths(d2t_path, t2d_path):
    """Reject when d2t or t2d path is missing or empty."""
    with pytest.raises(ValueError, match="Both d2t and t2d paths must be provided"):
        load_vocab_mappings(d2t_path, t2d_path, torch.device("cpu"))  # type: ignore[arg-type]


def test_load_vocab_mappings_file_not_found(tmp_path):
    """Reject when numpy files do not exist."""
    with pytest.raises(FileNotFoundError):
        load_vocab_mappings(
            str(tmp_path / "no_d2t.npy"),
            str(tmp_path / "no_t2d.npy"),
            torch.device("cpu"),
        )


def test_load_vocab_mappings_2d_arrays(tmp_path):
    """Support 2D arrays; vocab_size is first dimension."""
    np.save(tmp_path / "d2t.npy", np.arange(200).reshape(100, 2))
    np.save(tmp_path / "t2d.npy", np.arange(256).reshape(128, 2))
    d2t, t2d, vocab_size = load_vocab_mappings(
        str(tmp_path / "d2t.npy"), str(tmp_path / "t2d.npy"), torch.device("cpu")
    )
    assert d2t.shape == (100, 2)
    assert t2d.shape == (128, 2)
    assert vocab_size == 100


# --- load_pretrained_model ---


@patch("scripts.train.load_full_state_dict")
@patch("scripts.train.extract_vocab_mappings")
def test_load_pretrained_model_success(mock_extract, mock_load_state):
    """Load pretrained model via mocks; returns state_dict, d2t, t2d, vocab_size."""
    mock_load_state.return_value = {"layer.weight": torch.randn(10, 10)}
    mock_d2t, mock_t2d = torch.arange(100), torch.arange(128)
    mock_extract.return_value = (mock_d2t, mock_t2d)
    device = torch.device("cpu")
    state_dict, d2t, t2d, vocab_size = load_pretrained_model("/path/to/model", device)
    mock_load_state.assert_called_once_with("/path/to/model")
    mock_extract.assert_called_once()
    assert torch.equal(d2t, mock_d2t)
    assert torch.equal(t2d, mock_t2d)
    assert vocab_size == 100


@patch("scripts.train.load_full_state_dict")
@patch("scripts.train.extract_vocab_mappings")
def test_load_pretrained_model_vocab_size_from_d2t(mock_extract, mock_load_state):
    """vocab_size is derived from d2t.shape[0]."""
    mock_load_state.return_value = {}
    for size in [100, 256, 1024]:
        mock_extract.return_value = (torch.arange(size), torch.arange(size * 2))
        _, _, _, vocab_size = load_pretrained_model(
            "/path/to/model", torch.device("cpu")
        )
        assert vocab_size == size


# --- initialize_vocab_config ---


@pytest.mark.parametrize(
    ("d2t_path", "t2d_path"),
    [
        ("/path/to/d2t.npy", "/path/to/t2d.npy"),
        ("/path/to/d2t.npy", None),
        (None, "/path/to/t2d.npy"),
    ],
    ids=["both_paths", "d2t_only", "t2d_only"],
)
def test_initialize_vocab_config_rejects_pretrained_plus_paths(d2t_path, t2d_path):
    """Reject when pretrained_model_path and d2t/t2d paths are both set."""
    args = argparse.Namespace(
        pretrained_model_path="/path/to/model",
        d2t_path=d2t_path,
        t2d_path=t2d_path,
        verifier_name_or_path="meta-llama/Llama-2-7b",
    )
    with pytest.raises(ValueError, match="--pretrained-model-path overrides"):
        initialize_vocab_config(args, torch.device("cpu"))


@patch("scripts.train.load_pretrained_model")
def test_initialize_vocab_config_from_pretrained(mock_load):
    """Initialize from pretrained: returns d2t, t2d, vocab_size, state_dict."""
    mock_state = {"layer": torch.randn(10, 10)}
    mock_d2t, mock_t2d = torch.arange(100), torch.arange(128)
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
    """Initialize from numpy paths; state_dict is None."""
    mock_d2t, mock_t2d = torch.arange(100), torch.arange(128)
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
    """Without d2t/t2d, use verifier vocab_size; d2t/t2d and state_dict are None."""
    # Plain causal LM config (no text_config) so code uses .vocab_size directly
    mock_config_class.from_pretrained.return_value = type(
        "VerifierConfig", (), {"vocab_size": 32000}
    )()
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
    """Multimodal verifier: vocab_size from text_config.vocab_size."""
    mock_config_class.from_pretrained.return_value = MagicMock(
        text_config=MagicMock(vocab_size=32000)
    )
    args = argparse.Namespace(
        pretrained_model_path=None,
        d2t_path=None,
        t2d_path=None,
        verifier_name_or_path="Qwen/Qwen2-VL-7B",
    )
    _, _, vocab_size, _ = initialize_vocab_config(args, torch.device("cpu"))
    assert vocab_size == 32000
