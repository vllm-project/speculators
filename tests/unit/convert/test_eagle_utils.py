"""
Unit tests for Eagle converter utility functions.
"""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch

from speculators.convert.eagle.utils import (
    detect_fusion_bias_and_layernorms,
    download_checkpoint_from_hub,
    ensure_checkpoint_is_local,
    load_checkpoint_config,
    load_checkpoint_weights,
)


class TestDownloadCheckpointFromHub:
    """Test download_checkpoint_from_hub function."""

    @patch("speculators.convert.eagle.utils.snapshot_download")
    def test_successful_download(self, mock_snapshot_download, tmp_path):
        """Test successful checkpoint download."""
        mock_snapshot_download.return_value = str(tmp_path / "checkpoint")

        result = download_checkpoint_from_hub("test-model/checkpoint")

        assert isinstance(result, Path)
        assert str(result) == str(tmp_path / "checkpoint")
        mock_snapshot_download.assert_called_once_with(
            repo_id="test-model/checkpoint",
            allow_patterns=["*.json", "*.safetensors", "*.bin", "*.index.json"],
            cache_dir=None,
        )

    @patch("speculators.convert.eagle.utils.snapshot_download")
    def test_download_with_cache_dir(self, mock_snapshot_download, tmp_path):
        """Test download with custom cache directory."""
        cache_dir = tmp_path / "cache"
        mock_snapshot_download.return_value = str(tmp_path / "checkpoint")

        download_checkpoint_from_hub("test-model/checkpoint", cache_dir=str(cache_dir))

        mock_snapshot_download.assert_called_once_with(
            repo_id="test-model/checkpoint",
            allow_patterns=["*.json", "*.safetensors", "*.bin", "*.index.json"],
            cache_dir=str(cache_dir),
        )

    @patch("speculators.convert.eagle.utils.snapshot_download")
    def test_download_failure(self, mock_snapshot_download):
        """Test handling of download failures."""
        mock_snapshot_download.side_effect = Exception("Network error")

        with pytest.raises(FileNotFoundError, match="Checkpoint not found: test-model"):
            download_checkpoint_from_hub("test-model/checkpoint")


class TestEnsureCheckpointIsLocal:
    """Test ensure_checkpoint_is_local function."""

    def test_local_path_exists(self, tmp_path):
        """Test that existing local paths are returned as-is."""
        checkpoint_dir = tmp_path / "checkpoint"
        checkpoint_dir.mkdir()

        result = ensure_checkpoint_is_local(checkpoint_dir)

        assert result == checkpoint_dir

    @patch("speculators.convert.eagle.utils.download_checkpoint_from_hub")
    def test_download_when_not_local(self, mock_download, tmp_path):
        """Test downloading when path doesn't exist locally."""
        mock_download.return_value = tmp_path / "downloaded"

        result = ensure_checkpoint_is_local("test-model/checkpoint")

        assert result == tmp_path / "downloaded"
        mock_download.assert_called_once_with(
            model_id="test-model/checkpoint", cache_dir=None
        )

    @patch("speculators.convert.eagle.utils.download_checkpoint_from_hub")
    def test_download_with_cache_dir(self, mock_download, tmp_path):
        """Test downloading with cache directory."""
        cache_dir = tmp_path / "cache"
        mock_download.return_value = tmp_path / "downloaded"

        ensure_checkpoint_is_local("test-model/checkpoint", cache_dir=cache_dir)

        mock_download.assert_called_once_with(
            model_id="test-model/checkpoint", cache_dir=cache_dir
        )


class TestLoadCheckpointConfig:
    """Test load_checkpoint_config function."""

    def test_load_valid_config(self, tmp_path):
        """Test loading a valid config.json file."""
        config_data = {"model_type": "llama", "hidden_size": 4096, "num_layers": 32}

        checkpoint_dir = tmp_path / "checkpoint"
        checkpoint_dir.mkdir()
        config_path = checkpoint_dir / "config.json"
        config_path.write_text(json.dumps(config_data))

        result = load_checkpoint_config(checkpoint_dir)

        assert result == config_data

    def test_config_not_found(self, tmp_path):
        """Test error when config.json is missing."""
        checkpoint_dir = tmp_path / "checkpoint"
        checkpoint_dir.mkdir()

        with pytest.raises(FileNotFoundError, match="No config.json found"):
            load_checkpoint_config(checkpoint_dir)

    def test_invalid_json(self, tmp_path):
        """Test error when config.json contains invalid JSON."""
        checkpoint_dir = tmp_path / "checkpoint"
        checkpoint_dir.mkdir()
        config_path = checkpoint_dir / "config.json"
        config_path.write_text("invalid json {")

        with pytest.raises(json.JSONDecodeError):
            load_checkpoint_config(checkpoint_dir)


class TestLoadCheckpointWeights:
    """Test load_checkpoint_weights function."""

    @patch("speculators.convert.eagle.utils.safe_open")
    def test_load_safetensors_weights(self, mock_safe_open, tmp_path):
        """Test loading weights from safetensors format."""
        checkpoint_dir = tmp_path / "checkpoint"
        checkpoint_dir.mkdir()
        (checkpoint_dir / "model.safetensors").touch()

        # Mock safetensors file
        mock_file = MagicMock()
        mock_file.keys.return_value = ["weight1", "weight2"]
        mock_file.get_tensor.side_effect = lambda key: torch.randn(10, 10)
        mock_safe_open.return_value.__enter__.return_value = mock_file

        weights = load_checkpoint_weights(checkpoint_dir)

        assert len(weights) == 2
        assert "weight1" in weights
        assert "weight2" in weights
        assert all(isinstance(w, torch.Tensor) for w in weights.values())

    @patch("speculators.convert.eagle.utils.torch.load")
    def test_load_pytorch_weights(self, mock_torch_load, tmp_path):
        """Test loading weights from PyTorch bin format."""
        checkpoint_dir = tmp_path / "checkpoint"
        checkpoint_dir.mkdir()
        (checkpoint_dir / "pytorch_model.bin").touch()

        expected_weights = {
            "weight1": torch.randn(10, 10),
            "weight2": torch.randn(20, 20),
        }
        mock_torch_load.return_value = expected_weights

        weights = load_checkpoint_weights(checkpoint_dir)

        assert weights == expected_weights
        mock_torch_load.assert_called_once_with(
            checkpoint_dir / "pytorch_model.bin", map_location="cpu"
        )

    def test_no_weights_found(self, tmp_path):
        """Test error when no weights are found."""
        checkpoint_dir = tmp_path / "checkpoint"
        checkpoint_dir.mkdir()

        with pytest.raises(FileNotFoundError, match="No weights found"):
            load_checkpoint_weights(checkpoint_dir)

    def test_sharded_safetensors_not_supported(self, tmp_path):
        """Test error for sharded safetensors checkpoints."""
        checkpoint_dir = tmp_path / "checkpoint"
        checkpoint_dir.mkdir()
        (checkpoint_dir / "model.safetensors.index.json").touch()

        with pytest.raises(NotImplementedError, match="Sharded checkpoint detected"):
            load_checkpoint_weights(checkpoint_dir)

    def test_sharded_pytorch_not_supported(self, tmp_path):
        """Test error for sharded PyTorch checkpoints."""
        checkpoint_dir = tmp_path / "checkpoint"
        checkpoint_dir.mkdir()
        (checkpoint_dir / "pytorch_model.bin.index.json").touch()

        with pytest.raises(NotImplementedError, match="Sharded checkpoint detected"):
            load_checkpoint_weights(checkpoint_dir)

    def test_safetensors_takes_precedence(self, tmp_path):
        """Test that safetensors format takes precedence over PyTorch bin."""
        checkpoint_dir = tmp_path / "checkpoint"
        checkpoint_dir.mkdir()
        (checkpoint_dir / "model.safetensors").touch()
        (checkpoint_dir / "pytorch_model.bin").touch()

        with patch("speculators.convert.eagle.utils.safe_open") as mock_safe_open:
            mock_file = MagicMock()
            mock_file.keys.return_value = ["weight1"]
            mock_file.get_tensor.return_value = torch.randn(10, 10)
            mock_safe_open.return_value.__enter__.return_value = mock_file

            weights = load_checkpoint_weights(checkpoint_dir)

            assert len(weights) == 1
            mock_safe_open.assert_called_once()


class TestDetectFusionBiasAndLayernorms:
    """Test detect_fusion_bias_and_layernorms function."""

    def test_no_bias_no_layernorms(self):
        """Test detection when neither bias nor layernorms are present."""
        weights = {
            "fc.weight": torch.randn(4096, 8192),
            "layers.0.self_attn.q_proj.weight": torch.randn(4096, 4096),
        }

        has_bias, has_layernorms = detect_fusion_bias_and_layernorms(weights)

        assert not has_bias
        assert not has_layernorms

    def test_has_fusion_bias_only(self):
        """Test detection when only fusion bias is present."""
        weights = {
            "fc.weight": torch.randn(4096, 8192),
            "fc.bias": torch.randn(4096),
            "layers.0.self_attn.q_proj.weight": torch.randn(4096, 4096),
        }

        has_bias, has_layernorms = detect_fusion_bias_and_layernorms(weights)

        assert has_bias
        assert not has_layernorms

    def test_has_embed_layernorm_only(self):
        """Test detection when only embed_layernorm is present."""
        weights = {
            "fc.weight": torch.randn(4096, 8192),
            "embed_layernorm.weight": torch.randn(4096),
            "layers.0.self_attn.q_proj.weight": torch.randn(4096, 4096),
        }

        has_bias, has_layernorms = detect_fusion_bias_and_layernorms(weights)

        assert not has_bias
        assert has_layernorms

    def test_has_post_embedding_layernorm(self):
        """Test detection with post_embedding_layernorm."""
        weights = {
            "fc.weight": torch.randn(4096, 8192),
            "post_embedding_layernorm.weight": torch.randn(4096),
            "layers.0.self_attn.q_proj.weight": torch.randn(4096, 4096),
        }

        has_bias, has_layernorms = detect_fusion_bias_and_layernorms(weights)

        assert not has_bias
        assert has_layernorms

    def test_has_both_bias_and_layernorms(self):
        """Test detection when both bias and layernorms are present."""
        weights = {
            "fc.weight": torch.randn(4096, 8192),
            "fc.bias": torch.randn(4096),
            "embed_layernorm.weight": torch.randn(4096),
            "post_embedding_layernorm.weight": torch.randn(4096),
            "layers.0.self_attn.q_proj.weight": torch.randn(4096, 4096),
        }

        has_bias, has_layernorms = detect_fusion_bias_and_layernorms(weights)

        assert has_bias
        assert has_layernorms

    def test_empty_weights(self):
        """Test detection with empty weights dictionary."""
        weights = {}

        has_bias, has_layernorms = detect_fusion_bias_and_layernorms(weights)

        assert not has_bias
        assert not has_layernorms

    @patch("speculators.convert.eagle.utils.logger")
    def test_logging_messages(self, mock_logger):
        """Test that appropriate log messages are generated."""
        weights = {
            "fc.bias": torch.randn(4096),
            "embed_layernorm.weight": torch.randn(4096),
        }

        detect_fusion_bias_and_layernorms(weights)

        mock_logger.info.assert_any_call("Detected fusion bias in checkpoint")
        mock_logger.info.assert_any_call("Detected extra layernorms in checkpoint")
