"""
Unit tests for the simplified Eagle checkpoint converter.
"""

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import torch

from speculators.convert.eagle import EagleConverter


class TestEagleConverter:
    """Test the simplified Eagle converter."""

    @patch("speculators.convert.eagle.eagle_converter.snapshot_download")
    @patch("speculators.convert.eagle.eagle_converter.safe_open")
    @patch("speculators.convert.eagle.eagle_converter.save_file")
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

            # Mock save_file to create the actual file
            def mock_save_file_side_effect(weights_dict, path):
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

            # Check that embed_tokens.weight was not saved (weight tying)
            saved_weights = mock_save_file.call_args[0][0]
            assert "embed_tokens.weight" not in saved_weights
            assert "lm_head.weight" in saved_weights
            assert (
                "fusion_fc.weight" in saved_weights
            )  # fc.weight is renamed to fusion_fc.weight

    def test_layernorm_weight_mapping(self):
        """Test that layernorm weights are mapped correctly."""
        converter = EagleConverter()

        # Test the mappings
        assert (
            converter.LAYERNORM_MAPPINGS["embed_layernorm.weight"]
            == "embedding_layernorm.weight"
        )
        assert (
            converter.LAYERNORM_MAPPINGS["lm_head_layernorm.weight"]
            == "pre_lm_head_layernorm.weight"
        )

    def test_feature_detection(self):
        """Test automatic feature detection from weights."""
        converter = EagleConverter()

        # Test fusion bias detection and mapping
        weights_with_bias = {"fc.bias": torch.randn(8192)}
        processed = converter._process_weights(weights_with_bias, layernorms=False)
        assert "fusion_fc.bias" in processed  # fc.bias is renamed to fusion_fc.bias

        # Test layernorm detection and mapping
        weights_with_layernorms = {
            "embed_layernorm.weight": torch.randn(4096),
            "lm_head_layernorm.weight": torch.randn(4096),
        }
        processed = converter._process_weights(weights_with_layernorms, layernorms=True)
        assert "embedding_layernorm.weight" in processed
        assert "pre_lm_head_layernorm.weight" in processed
        assert "embed_layernorm.weight" not in processed
