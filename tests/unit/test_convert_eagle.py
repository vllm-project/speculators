"""
Unit tests for the simplified Eagle checkpoint converter.
"""

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import torch

from speculators.convert.eagle import EagleConverter
from speculators.convert.eagle.utils import (
    detect_fusion_bias_and_layernorms,
    download_checkpoint_from_hub,
    ensure_checkpoint_is_local,
    load_checkpoint_config,
    load_checkpoint_weights,
)


class TestEagleConverter:
    """Test the simplified Eagle converter."""

    @patch("speculators.convert.eagle.utils.snapshot_download")
    @patch("speculators.convert.eagle.utils.safe_open")
    @patch("safetensors.torch.save_file")
    def test_convert_standard_eagle(
        self, mock_save_file, mock_safe_open, mock_download
    ):
        """Test converting a standard Eagle checkpoint."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            input_path = tmpdir / "input"
            output_path = tmpdir / "output"

            # Setup mocks
            input_path.mkdir()

            # Mock config
            config = {
                "model_type": "llama",
                "vocab_size": 32000,
                "hidden_size": 4096,
                "intermediate_size": 11008,
                "num_hidden_layers": 32,
                "num_attention_heads": 32,
                "num_key_value_heads": 8,
                "bos_token_id": 1,
                "eos_token_id": 2,
            }
            (input_path / "config.json").write_text(json.dumps(config))

            # Mock weights
            weights = {
                "embed_tokens.weight": torch.randn(32000, 4096),
                "fc.weight": torch.randn(4096, 8192),
                "layers.0.self_attn.q_proj.weight": torch.randn(4096, 4096),
                "lm_head.weight": torch.randn(32000, 4096),
            }

            # Mock safetensors file
            (input_path / "model.safetensors").touch()
            mock_safe_open_instance = MagicMock()
            mock_safe_open_instance.keys.return_value = weights.keys()
            mock_safe_open_instance.get_tensor = lambda k: weights[k]
            mock_safe_open.return_value.__enter__.return_value = mock_safe_open_instance

            mock_download.return_value = input_path

            # Mock save_file to create the actual file and capture weights
            saved_weights_capture = []
            def mock_save_file_side_effect(weights_dict, path):
                saved_weights_capture.append(weights_dict)
                path.parent.mkdir(parents=True, exist_ok=True)
                path.touch()  # Create the file

            mock_save_file.side_effect = mock_save_file_side_effect

            # Run conversion
            converter = EagleConverter()
            converter.convert(
                input_path,
                output_path,
                base_model="meta-llama/Llama-3.1-8B",
                validate=False,  # Skip validation to avoid loading model
            )

            # Check output
            assert (output_path / "config.json").exists()
            assert (output_path / "model.safetensors").exists()

            # Check config
            saved_config = json.loads((output_path / "config.json").read_text())
            assert saved_config["speculators_model_type"] == "eagle"
            assert saved_config["layernorms"] is False
            assert saved_config["fusion_bias"] is False

            # Since we're using model.save_pretrained, the save_file mock won't be called
            # Instead, check that the model saved its files correctly
            assert (output_path / "eagle.py").exists()  # Auto-generated code
            # The actual weights are saved by the model's save_pretrained method

    def test_layernorm_weight_mapping(self):
        """Test that layernorm weights are mapped correctly."""
        converter = EagleConverter()
        
        # Test the mappings
        assert (
            converter.EAGLE_TO_SPECULATORS_LAYERNORM_MAPPINGS["embed_layernorm.weight"]
            == "embedding_layernorm.weight"
        )
        assert (
            converter.EAGLE_TO_SPECULATORS_LAYERNORM_MAPPINGS["lm_head_layernorm.weight"]
            == "pre_lm_head_layernorm.weight"
        )

    def test_weight_skipping_and_remapping(self):
        """Test weight skipping and remapping logic."""
        converter = EagleConverter()
        
        # Test embed_tokens skipping
        assert converter._should_skip_weight("embed_tokens.weight", has_layernorms=False) is True
        assert converter._should_skip_weight("embed_tokens.weight", has_layernorms=True) is True
        
        # Test hidden_layernorm skipping when layernorms disabled
        assert converter._should_skip_weight("hidden_layernorm.weight", has_layernorms=False) is True
        assert converter._should_skip_weight("hidden_layernorm.weight", has_layernorms=True) is False
        
        # Test fc weight remapping
        assert converter._remap_weight_name("fc.weight", has_layernorms=False) == "fusion_fc.weight"
        assert converter._remap_weight_name("fc.bias", has_layernorms=False) == "fusion_fc.bias"
        
        # Test transformer layer remapping
        assert converter._remap_weight_name("layers.0.self_attn.q_proj.weight", has_layernorms=False) == "transformer.self_attn.q_proj.weight"
        
        # Test hidden_layernorm remapping when layernorms enabled
        assert converter._remap_weight_name("hidden_layernorm.weight", has_layernorms=True) == "transformer.input_layernorm.weight"
        
        # Test layernorm mappings
        assert converter._remap_weight_name("embed_layernorm.weight", has_layernorms=True) == "embedding_layernorm.weight"
        assert converter._remap_weight_name("lm_head_layernorm.weight", has_layernorms=True) == "pre_lm_head_layernorm.weight"
        
        # Test unchanged names
        assert converter._remap_weight_name("lm_head.weight", has_layernorms=False) == "lm_head.weight"

    def test_process_checkpoint_weights(self):
        """Test processing weights with various configurations."""
        converter = EagleConverter()
        
        # Test fusion bias processing
        weights_with_bias = {"fc.bias": torch.randn(8192)}
        processed = converter._process_checkpoint_weights(weights_with_bias, has_layernorms=False)
        assert "fusion_fc.bias" in processed  # fc.bias is renamed to fusion_fc.bias

        # Test layernorm processing
        weights_with_layernorms = {
            "embed_layernorm.weight": torch.randn(4096),
            "lm_head_layernorm.weight": torch.randn(4096),
        }
        processed = converter._process_checkpoint_weights(weights_with_layernorms, has_layernorms=True)
        assert "embedding_layernorm.weight" in processed
        assert "pre_lm_head_layernorm.weight" in processed
        assert "embed_layernorm.weight" not in processed

    def test_detect_fusion_bias_and_layernorms(self):
        """Test automatic detection of fusion bias and layernorms."""
        # Test fusion bias detection
        weights = {"fc.bias": torch.randn(4096)}
        has_bias, has_ln = detect_fusion_bias_and_layernorms(weights)
        assert has_bias is True
        assert has_ln is False
        
        # Test layernorm detection
        weights = {"embed_layernorm.weight": torch.randn(4096)}
        has_bias, has_ln = detect_fusion_bias_and_layernorms(weights)
        assert has_bias is False
        assert has_ln is True
        
        # Test both
        weights = {
            "fc.bias": torch.randn(4096),
            "post_embedding_layernorm.weight": torch.randn(4096)
        }
        has_bias, has_ln = detect_fusion_bias_and_layernorms(weights)
        assert has_bias is True
        assert has_ln is True

    @patch("speculators.convert.eagle.utils.snapshot_download")
    def test_download_checkpoint_from_hub(self, mock_download):
        """Test downloading from HuggingFace Hub."""
        mock_download.return_value = "/tmp/downloaded"
        
        path = download_checkpoint_from_hub("test/model")
        assert path == Path("/tmp/downloaded")
        mock_download.assert_called_once_with(
            repo_id="test/model",
            allow_patterns=["*.json", "*.safetensors", "*.bin", "*.index.json"],
            cache_dir=None
        )

    def test_ensure_checkpoint_is_local(self):
        """Test ensuring checkpoint is local."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Test with existing local path
            local_path = Path(tmpdir) / "checkpoint"
            local_path.mkdir()
            
            result = ensure_checkpoint_is_local(local_path)
            assert result == local_path
            
            # Test with non-existent path (would trigger download)
            with patch("speculators.convert.eagle.utils.download_checkpoint_from_hub") as mock_download:
                mock_download.return_value = Path("/tmp/downloaded")
                
                result = ensure_checkpoint_is_local("non/existent")
                assert result == Path("/tmp/downloaded")
                mock_download.assert_called_once_with(
                    model_id="non/existent",
                    cache_dir=None
                )
