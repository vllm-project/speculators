"""
Unit tests for Eagle3 converter focusing on the specific fixes implemented.

Tests cover:
- Embeddings replacement with verifier embeddings
- Weight remapping from midlayer.* to layers.0.*
- Configuration compatibility (max_position_embeddings, rope_theta)
- Validation of the conversion process
"""

from pathlib import Path
from unittest.mock import patch

import pytest
import torch
from safetensors.torch import load_file

from speculators.convert.eagle.eagle3_converter import Eagle3Converter
from speculators.convert.eagle.eagle3_legacy_model import Eagle3Speculator
from speculators.convert.utils import load_checkpoint_config


class TestEagle3ConverterFixes:
    """Test the specific fixes implemented in Eagle3 converter."""

    @pytest.fixture
    def sample_eagle3_config(self):
        """Sample Eagle3 configuration for testing."""
        return {
            "target_vocab_size": 128000,
            "hidden_size": 4096,
            "intermediate_size": 11008,
            "num_attention_heads": 32,
            "num_key_value_heads": 8,
            "hidden_act": "silu",
            "max_position_embeddings": 2048,  # This will be maxed with verifier
            "initializer_range": 0.02,
            "rms_norm_eps": 1e-6,
            "attention_bias": False,
            "rope_theta": 10000.0,
            "mlp_bias": False,
        }

    @pytest.fixture
    def sample_eagle3_weights(self):
        """Sample Eagle3 weights with midlayer.* naming."""
        return {
            # Embeddings that should be replaced
            "embed_tokens.weight": torch.randn(128000, 4096),
            # Weights that should be remapped from midlayer.* to layers.0.*
            "midlayer.self_attn.q_proj.weight": torch.randn(4096, 4096),
            "midlayer.self_attn.k_proj.weight": torch.randn(1024, 4096),
            "midlayer.self_attn.v_proj.weight": torch.randn(1024, 4096),
            "midlayer.self_attn.o_proj.weight": torch.randn(4096, 4096),
            "midlayer.mlp.gate_proj.weight": torch.randn(11008, 4096),
            "midlayer.mlp.up_proj.weight": torch.randn(11008, 4096),
            "midlayer.mlp.down_proj.weight": torch.randn(4096, 11008),
            "midlayer.input_layernorm.weight": torch.randn(4096),
            "midlayer.post_attention_layernorm.weight": torch.randn(4096),
            # Other weights
            "t2d": torch.randn(128000, 4096),
        }

    @pytest.fixture
    def sample_verifier_config(self):
        """Sample verifier configuration."""
        return {
            "architectures": ["LlamaForCausalLM"],
            "max_position_embeddings": 131072,  # Should be maxed with Eagle3's 2048
            "rope_theta": 500000.0,
            "vocab_size": 128256,
        }

    @pytest.mark.sanity
    @patch(
        "speculators.convert.eagle.eagle3_converter.PretrainedConfig.get_config_dict"
    )
    def test_config_max_position_embeddings_logic(
        self, mock_get_config, sample_eagle3_config, sample_verifier_config
    ):
        """Test that max_position_embeddings uses max of Eagle3 and verifier."""
        mock_get_config.return_value = (sample_verifier_config, None)

        converter = Eagle3Converter()

        llama_config = converter._create_transformer_config_from_eagle(
            sample_eagle3_config, "meta-llama/Llama-3.1-8B-Instruct"
        )

        # Check that max_position_embeddings is the max of both values
        # Eagle3: 2048, Verifier: 131072, so should be 131072
        assert llama_config.max_position_embeddings == 131072

        # Check that other values come from Eagle3 config
        assert llama_config.hidden_size == 4096
        assert llama_config.num_attention_heads == 32
        # rope_theta comes from Eagle3 config, not verifier
        if hasattr(llama_config, "rope_parameters"):
            # Transformers v5
            assert llama_config.rope_parameters is not None
            assert llama_config.rope_parameters.get("rope_theta") == 10000.0
        else:
            assert llama_config.rope_theta == 10000.0

    @pytest.mark.sanity
    @patch(
        "speculators.convert.eagle.eagle3_converter.PretrainedConfig.get_config_dict"
    )
    def test_config_num_hidden_layers_from_config(
        self, mock_get_config, sample_eagle3_config
    ):
        """Test that num_hidden_layers is taken from eagle_config when present."""
        mock_get_config.return_value = ({}, None)
        converter = Eagle3Converter()

        # Add num_hidden_layers to the sample config
        sample_eagle3_config["num_hidden_layers"] = 3

        llama_config = converter._create_transformer_config_from_eagle(
            sample_eagle3_config, "meta-llama/Llama-3.1-8B-Instruct"
        )
        assert llama_config.num_hidden_layers == 3

    @pytest.mark.sanity
    @patch(
        "speculators.convert.eagle.eagle3_converter.PretrainedConfig.get_config_dict"
    )
    def test_config_num_hidden_layers_default(
        self, mock_get_config, sample_eagle3_config
    ):
        """Test that num_hidden_layers defaults to 1 when not in config."""
        mock_get_config.return_value = ({}, None)
        converter = Eagle3Converter()

        # Remove num_hidden_layers if present
        sample_eagle3_config.pop("num_hidden_layers", None)

        llama_config = converter._create_transformer_config_from_eagle(
            sample_eagle3_config, "meta-llama/Llama-3.1-8B-Instruct"
        )
        assert llama_config.num_hidden_layers == 1

    @pytest.mark.sanity
    @pytest.mark.parametrize("dtype_key", ["torch_dtype", "dtype"])
    @patch(
        "speculators.convert.eagle.eagle3_converter.PretrainedConfig.get_config_dict"
    )
    def test_config_dtype_from_either_key(
        self, mock_get_config, dtype_key, sample_eagle3_config
    ):
        """Checkpoints saved by newer transformers write ``dtype``, older ones
        ``torch_dtype``; both must set the dtype the weights are saved in."""
        mock_get_config.return_value = ({}, None)
        sample_eagle3_config[dtype_key] = "bfloat16"

        llama_config = Eagle3Converter()._create_transformer_config_from_eagle(
            sample_eagle3_config, "meta-llama/Llama-3.1-8B-Instruct"
        )
        # The dtype _save_converted_checkpoint casts the weights to.
        saved_dtype = getattr(llama_config, "dtype", None) or getattr(
            llama_config, "torch_dtype", None
        )
        assert saved_dtype == torch.bfloat16

    @pytest.mark.sanity
    @patch(
        "speculators.convert.eagle.eagle3_converter.PretrainedConfig.get_config_dict"
    )
    def test_config_fallback_when_verifier_unavailable(
        self, mock_get_config, sample_eagle3_config
    ):
        """Test fallback behavior when verifier config cannot be loaded."""
        mock_get_config.side_effect = Exception("Network error")

        converter = Eagle3Converter()

        # Should raise exception since method doesn't handle this gracefully
        with pytest.raises((RuntimeError, ValueError)):
            converter._create_transformer_config_from_eagle(
                sample_eagle3_config, "meta-llama/Llama-3.1-8B-Instruct"
            )

    @pytest.mark.sanity
    def test_converted_model_config_structure(self):
        """Test that the config structure created is valid for Eagle3Speculator."""
        converter = Eagle3Converter()

        eagle_config = {
            "target_vocab_size": 128000,
            "hidden_size": 4096,
            "draft_vocab_size": 32000,
        }

        with patch(
            "speculators.convert.eagle.eagle3_converter.PretrainedConfig.get_config_dict"
        ) as mock_config:
            mock_config.return_value = ({"max_position_embeddings": 131072}, None)

            config = converter._build_eagle3_speculator_config(
                eagle_config,
                "meta-llama/Llama-3.1-8B-Instruct",
                norm_before_residual=False,
            )

        # Verify config structure
        assert config.speculators_model_type == "eagle3"
        assert config.transformer_layer_config.hidden_size == 4096
        assert config.speculators_config.algorithm == "eagle3"
        assert (
            config.speculators_config.verifier.name_or_path
            == "meta-llama/Llama-3.1-8B-Instruct"
        )

    @pytest.mark.sanity
    def test_eagle_aux_hidden_state_layer_ids_parameter(self):
        """Test that eagle_aux_hidden_state_layer_ids parameter is properly handled."""
        converter = Eagle3Converter()

        eagle_config = {
            "target_vocab_size": 128000,
            "hidden_size": 4096,
            "draft_vocab_size": 32000,
        }

        with patch(
            "speculators.convert.eagle.eagle3_converter.PretrainedConfig.get_config_dict"
        ) as mock_config:
            mock_config.return_value = ({"max_position_embeddings": 131072}, None)

            # Test with None (default)
            config_none = converter._build_eagle3_speculator_config(
                eagle_config,
                "meta-llama/Llama-3.1-8B-Instruct",
                norm_before_residual=False,
            )
            assert config_none.eagle_aux_hidden_state_layer_ids is None

            # Test with specific layer IDs
            layer_ids = [1, 23, 44]
            config_with_ids = converter._build_eagle3_speculator_config(
                eagle_config,
                "meta-llama/Llama-3.1-8B-Instruct",
                norm_before_residual=False,
                eagle_aux_hidden_state_layer_ids=layer_ids,
            )
            assert config_with_ids.eagle_aux_hidden_state_layer_ids == layer_ids

    @pytest.mark.sanity
    def test_eagle_aux_hidden_state_layer_ids_in_config_serialization(self):
        """Test that eagle_aux_hidden_state_layer_ids is properly serialized."""
        converter = Eagle3Converter()

        eagle_config = {
            "target_vocab_size": 128000,
            "hidden_size": 4096,
            "draft_vocab_size": 32000,
        }

        with patch(
            "speculators.convert.eagle.eagle3_converter.PretrainedConfig.get_config_dict"
        ) as mock_config:
            mock_config.return_value = ({"max_position_embeddings": 131072}, None)

            layer_ids = [1, 23, 44]
            config = converter._build_eagle3_speculator_config(
                eagle_config,
                "meta-llama/Llama-3.1-8B-Instruct",
                norm_before_residual=False,
                eagle_aux_hidden_state_layer_ids=layer_ids,
            )

            # Test that the config can be serialized to dict
            config_dict = config.to_dict()
            assert "eagle_aux_hidden_state_layer_ids" in config_dict
            assert config_dict["eagle_aux_hidden_state_layer_ids"] == layer_ids

    @pytest.mark.sanity
    def test_nm_testing_2layer_eagle3_model_config(self, tmp_path):
        """Test that multi-layer conversion support."""
        converter = Eagle3Converter()

        # TODO: Use a real model in the future
        checkpoint_path = "nm-testing/testing-llama3.1.8b-2layer-eagle3"
        base_model = "RedHatAI/Meta-Llama-3.1-8B-Instruct-FP8-dynamic"

        local_checkpoint_path = tmp_path / checkpoint_path.rsplit("/", maxsplit=1)[-1]
        converter.convert(
            checkpoint_path,
            local_checkpoint_path,
            base_model,
            norm_before_residual=False,
        )

        config = load_checkpoint_config(local_checkpoint_path)

        # Verify that num_hidden_layers is correctly set to 2
        assert config["transformer_layer_config"]["num_hidden_layers"] == 2


class TestEagle3ConverterWeightValidation:
    """Test missing and colliding weight key detection during Eagle3 saving."""

    @staticmethod
    def _build_tiny_config_and_weights(
        *,
        reduce_vocab_size: bool = True,
        has_drafter_embedding: bool = True,
        norm_before_residual: bool = False,
        use_midlayer_prefix: bool = True,
    ):
        hidden_size = 32
        vocab_size = 64
        draft_vocab_size = 32
        verifier_cfg = {
            "architectures": ["LlamaForCausalLM"],
            "max_position_embeddings": 128,
            "rope_theta": 500000.0,
            "vocab_size": vocab_size,
        }
        eagle_cfg = {
            "target_vocab_size": vocab_size,
            "hidden_size": hidden_size,
            "intermediate_size": 64,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "hidden_act": "silu",
            "max_position_embeddings": 64,
            "initializer_range": 0.02,
            "rms_norm_eps": 1e-6,
            "attention_bias": False,
            "rope_theta": 10000.0,
            "mlp_bias": False,
            "draft_vocab_size": draft_vocab_size,
        }
        converter = Eagle3Converter()
        with patch(
            "speculators.convert.eagle.eagle3_converter.PretrainedConfig.get_config_dict",
            return_value=(verifier_cfg, {}),
        ):
            config = converter._build_eagle3_speculator_config(
                eagle_cfg,
                "meta-llama/Llama-3.1-8B-Instruct",
                norm_before_residual=norm_before_residual,
            )

        ref = Eagle3Speculator(  # type: ignore[abstract]
            config=config,
            verifier=None,
            verifier_attachment_mode="detached",
            reduce_vocab_size=reduce_vocab_size,
            has_drafter_embedding=has_drafter_embedding,
        )
        source_weights: dict[str, torch.Tensor] = {}
        for key, value in ref.state_dict().items():
            if use_midlayer_prefix and key.startswith("layers.0."):
                mapped_key = f"midlayer.{key.removeprefix('layers.0.')}"
            else:
                mapped_key = key
            source_weights[mapped_key] = value.clone()
        return converter, config, source_weights

    @pytest.mark.sanity
    def test_save_raises_on_renamed_weight_key_and_writes_no_files(
        self, tmp_path: Path
    ):
        converter, config, weights = self._build_tiny_config_and_weights()
        weights["midlayer.mlp.up_projection.weight"] = weights.pop(
            "midlayer.mlp.up_proj.weight"
        )
        out_dir = tmp_path / "converted"

        with pytest.raises(ValueError, match="Missing required keys") as exc_info:
            converter._save_converted_checkpoint(
                config,
                weights,
                out_dir,
                reduce_vocab_size=True,
                has_drafter_embedding=True,
            )

        message = str(exc_info.value)
        assert "layers.0.mlp.up_proj.weight" in message
        assert "layers.0.mlp.up_projection.weight" in message
        assert not out_dir.exists()

    @pytest.mark.sanity
    @pytest.mark.parametrize("missing_key", ["fc.weight", "d2t"])
    def test_save_raises_on_missing_required_key(
        self, tmp_path: Path, missing_key: str
    ):
        converter, config, weights = self._build_tiny_config_and_weights(
            reduce_vocab_size=True,
            has_drafter_embedding=True,
        )
        del weights[missing_key]
        out_dir = tmp_path / "converted"

        with pytest.raises(ValueError, match="Missing required keys") as exc_info:
            converter._save_converted_checkpoint(
                config,
                weights,
                out_dir,
                reduce_vocab_size=True,
                has_drafter_embedding=True,
            )

        assert missing_key in str(exc_info.value)
        assert not out_dir.exists()

    @pytest.mark.sanity
    @pytest.mark.parametrize("midlayer_first", [True, False])
    def test_save_raises_on_midlayer_remapping_collision(
        self, tmp_path: Path, midlayer_first: bool
    ):
        converter, config, base_weights = self._build_tiny_config_and_weights()
        up_proj = base_weights["midlayer.mlp.up_proj.weight"]
        weights: dict[str, torch.Tensor] = {}
        for key, value in base_weights.items():
            if key == "midlayer.mlp.up_proj.weight":
                if midlayer_first:
                    weights["midlayer.mlp.up_proj.weight"] = up_proj
                    weights["layers.0.mlp.up_proj.weight"] = up_proj.clone()
                else:
                    weights["layers.0.mlp.up_proj.weight"] = up_proj.clone()
                    weights["midlayer.mlp.up_proj.weight"] = up_proj
            else:
                weights[key] = value
        out_dir = tmp_path / "converted"

        with pytest.raises(ValueError, match="collision") as exc_info:
            converter._save_converted_checkpoint(
                config,
                weights,
                out_dir,
                reduce_vocab_size=True,
                has_drafter_embedding=True,
            )

        assert "layers.0.mlp.up_proj.weight" in str(exc_info.value)
        assert not out_dir.exists()

    @pytest.mark.sanity
    def test_save_allows_unexpected_extra_keys_when_all_required_present(
        self, tmp_path: Path
    ):
        converter, config, weights = self._build_tiny_config_and_weights()
        expected_up_proj = weights["midlayer.mlp.up_proj.weight"].clone()
        weights["layers.1.mlp.up_proj.weight"] = torch.randn_like(expected_up_proj)
        weights["extra.unused.weight"] = torch.randn(4, 4)
        out_dir = tmp_path / "converted"

        saved_dir = converter._save_converted_checkpoint(
            config,
            weights,
            out_dir,
            reduce_vocab_size=True,
            has_drafter_embedding=True,
        )

        saved_weights = load_file(str(saved_dir / "model.safetensors"))
        assert torch.equal(
            saved_weights["layers.0.mlp.up_proj.weight"].float(),
            expected_up_proj.float(),
        )

    @pytest.mark.sanity
    @pytest.mark.parametrize("reduce_vocab_size", [False, True])
    @pytest.mark.parametrize("has_drafter_embedding", [False, True])
    @pytest.mark.parametrize("norm_before_residual", [False, True])
    def test_save_succeeds_across_optional_flag_combinations(
        self,
        tmp_path: Path,
        reduce_vocab_size: bool,
        has_drafter_embedding: bool,
        norm_before_residual: bool,
    ):
        converter, config, weights = self._build_tiny_config_and_weights(
            reduce_vocab_size=reduce_vocab_size,
            has_drafter_embedding=has_drafter_embedding,
            norm_before_residual=norm_before_residual,
        )
        expected_up_proj = weights["midlayer.mlp.up_proj.weight"].clone()
        out_dir = tmp_path / "converted"

        saved_dir = converter._save_converted_checkpoint(
            config,
            weights,
            out_dir,
            reduce_vocab_size=reduce_vocab_size,
            has_drafter_embedding=has_drafter_embedding,
        )

        saved_weights = load_file(str(saved_dir / "model.safetensors"))
        assert torch.equal(
            saved_weights["layers.0.mlp.up_proj.weight"].float(),
            expected_up_proj.float(),
        )
