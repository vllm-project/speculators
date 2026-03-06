"""Unit tests for pretrained model loading utilities in speculators.utils.loading."""

import json
from unittest.mock import MagicMock

import pytest
import torch
from safetensors.torch import save_file

from speculators.utils.loading import (
    _find_exact_key,
    _validate_mapping_tensor,
    extract_vocab_mappings,
    load_full_state_dict,
    load_pretrained_weights,
)


# --- extract_vocab_mappings ---


class TestExtractVocabMappings:
    def test_success(self):
        """Extracts d2t/t2d and removes them from state_dict."""
        state_dict = {
            "model.layer1.weight": torch.randn(10, 10),
            "d2t": torch.randn(100),
            "t2d": torch.randn(128),
        }

        device = torch.device("cpu")
        d2t, t2d = extract_vocab_mappings(state_dict, device)

        assert d2t.shape == (100,)
        assert t2d.shape == (128,)
        assert d2t.device == device
        assert t2d.device == device
        assert "d2t" not in state_dict
        assert "t2d" not in state_dict
        assert "model.layer1.weight" in state_dict

    @pytest.mark.parametrize(
        ("state_dict", "missing_key"),
        [
            ({"layer.weight": torch.randn(10, 10), "t2d": torch.randn(100)}, "d2t"),
            ({"layer.weight": torch.randn(10, 10), "d2t": torch.randn(100)}, "t2d"),
        ],
    )
    def test_missing_key(self, state_dict, missing_key):
        """Errors when d2t or t2d is missing."""
        with pytest.raises(ValueError, match=f"Key '{missing_key}' not found"):
            extract_vocab_mappings(state_dict, torch.device("cpu"))

    def test_invalid_shape_3d(self):
        """Errors on non-1D d2t."""
        state_dict = {"d2t": torch.randn(10, 10, 10), "t2d": torch.randn(128)}
        with pytest.raises(ValueError, match="Unexpected d2t shape"):
            extract_vocab_mappings(state_dict, torch.device("cpu"))

    def test_case_insensitive(self):
        """Matches keys case-insensitively."""
        state_dict = {"D2T": torch.randn(100), "T2D": torch.randn(128)}
        device = torch.device("cpu")
        d2t, t2d = extract_vocab_mappings(state_dict, device)

        assert d2t.shape == (100,)
        assert t2d.shape == (128,)
        assert "D2T" not in state_dict
        assert "T2D" not in state_dict


# --- _find_exact_key ---


class TestFindExactKey:
    @pytest.mark.parametrize(
        ("state_dict", "lookup_key", "expected_key"),
        [
            ({"d2t": torch.randn(10)}, "d2t", "d2t"),
            ({"D2T": torch.randn(10)}, "d2t", "D2T"),
        ],
    )
    def test_match(self, state_dict, lookup_key, expected_key):
        result = _find_exact_key(state_dict, lookup_key)
        assert result == expected_key

    def test_not_found(self):
        state_dict = {"model.layer.weight": torch.randn(10, 10)}
        with pytest.raises(ValueError, match="Key 'd2t' not found"):
            _find_exact_key(state_dict, "d2t")

    def test_error_shows_available_keys(self):
        state_dict = {
            "model.layer1.weight": torch.randn(10, 10),
            "model.layer2.weight": torch.randn(10, 10),
        }
        with pytest.raises(ValueError, match="Available keys"):
            _find_exact_key(state_dict, "missing_key")


# --- _validate_mapping_tensor ---


class TestValidateMappingTensor:
    def test_1d_valid(self):
        """1D tensors pass validation."""
        _validate_mapping_tensor(torch.randn(100), "test")

    @pytest.mark.parametrize(
        "tensor",
        [
            torch.randn(100, 128),  # 2D
            torch.randn(10, 10, 10),  # 3D
            torch.tensor(5.0),  # 0D scalar
        ],
    )
    def test_invalid_ndim(self, tensor):
        """Non-1D tensors are rejected."""
        with pytest.raises(ValueError, match="Unexpected test shape"):
            _validate_mapping_tensor(tensor, "test")


# --- load_full_state_dict ---


class TestLoadFullStateDict:
    def test_single_file(self, tmp_path):
        """Loads from a single model.safetensors file."""
        state_dict = {
            "layer1.weight": torch.randn(10, 10),
            "layer2.bias": torch.randn(10),
            "d2t": torch.randn(100),
            "t2d": torch.randn(128),
        }
        save_file(state_dict, str(tmp_path / "model.safetensors"))

        loaded = load_full_state_dict(str(tmp_path))
        assert len(loaded) == len(state_dict)
        for key, value in state_dict.items():
            assert key in loaded
            assert torch.equal(loaded[key], value)

    def test_sharded(self, tmp_path):
        """Loads from sharded safetensors files."""
        shard1 = {"layer1.weight": torch.randn(10, 10), "d2t": torch.randn(100)}
        shard2 = {"layer2.weight": torch.randn(20, 20), "t2d": torch.randn(128)}

        save_file(shard1, str(tmp_path / "model-00001-of-00002.safetensors"))
        save_file(shard2, str(tmp_path / "model-00002-of-00002.safetensors"))

        index = {
            "metadata": {"total_size": 12345},
            "weight_map": {
                "layer1.weight": "model-00001-of-00002.safetensors",
                "d2t": "model-00001-of-00002.safetensors",
                "layer2.weight": "model-00002-of-00002.safetensors",
                "t2d": "model-00002-of-00002.safetensors",
            },
        }
        with (tmp_path / "model.safetensors.index.json").open("w") as f:
            json.dump(index, f)

        loaded = load_full_state_dict(str(tmp_path))
        assert len(loaded) == 4
        assert torch.equal(loaded["layer1.weight"], shard1["layer1.weight"])
        assert torch.equal(loaded["d2t"], shard1["d2t"])
        assert torch.equal(loaded["layer2.weight"], shard2["layer2.weight"])
        assert torch.equal(loaded["t2d"], shard2["t2d"])

    def test_file_not_found(self, tmp_path):
        """Errors when no model files exist."""
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()
        with pytest.raises(FileNotFoundError):
            load_full_state_dict(str(empty_dir))


# --- load_pretrained_weights ---


class TestLoadPretrainedWeights:
    def test_success(self):
        """Loads weights into model via load_state_dict."""
        mock_model = MagicMock()
        mock_model.load_state_dict.return_value = ([], [])
        state_dict = {"layer.weight": torch.randn(10, 10)}

        load_pretrained_weights(mock_model, state_dict, "/path/to/model")
        mock_model.load_state_dict.assert_called_once()

    def test_with_expected_missing(self):
        """Expected missing keys (d2t, t2d, verifier_lm_head) don't warn."""
        mock_model = MagicMock()
        expected_missing = ["d2t", "t2d", "verifier_lm_head.weight"]
        mock_model.load_state_dict.return_value = (expected_missing, [])

        # Should not raise
        load_pretrained_weights(
            mock_model, {"layer.weight": torch.randn(10, 10)}, "/test"
        )

    def test_strict_false(self):
        """Passes strict=False to load_state_dict."""
        mock_model = MagicMock()
        mock_model.load_state_dict.return_value = ([], [])
        state_dict = {"layer.weight": torch.randn(10, 10)}

        load_pretrained_weights(mock_model, state_dict, "/path")

        call_kwargs = mock_model.load_state_dict.call_args
        assert call_kwargs[1].get("strict") is False or (
            len(call_kwargs[0]) >= 2 and call_kwargs[0][1] is False
        )
