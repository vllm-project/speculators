"""
Unit tests for Eagle3 converter focusing on the specific fixes implemented.

Tests cover:
- Embeddings replacement with verifier embeddings
- Weight remapping from midlayer.* to layers.0.*
- Configuration compatibility (max_position_embeddings, rope_theta)
- Validation of the conversion process
"""

import json
from unittest.mock import patch

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import save_file
from transformers import LlamaConfig

from speculators.config import SpeculatorsConfig, VerifierConfig
from speculators.convert.eagle.eagle3_converter import Eagle3Converter
from speculators.convert.utils import load_checkpoint_config
from speculators.models.eagle3 import Eagle3SpeculatorConfig
from speculators.proposals.greedy import GreedyTokenProposalConfig


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


def _tiny_eagle3_speculator_config() -> Eagle3SpeculatorConfig:
    return Eagle3SpeculatorConfig(
        transformer_layer_config=LlamaConfig(
            vocab_size=32,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=8,
            max_position_embeddings=64,
            tie_word_embeddings=False,
        ),
        draft_vocab_size=16,
        speculators_config=SpeculatorsConfig(
            algorithm="eagle3",
            proposal_methods=[GreedyTokenProposalConfig(speculative_tokens=3)],
            default_proposal_method="greedy",
            verifier=VerifierConfig(
                name_or_path="dummy/verifier",
                architectures=["LlamaForCausalLM"],
            ),
        ),
    )


def _tiny_eagle3_weights(*, layer_prefix: str = "midlayer.") -> dict[str, torch.Tensor]:
    g = torch.Generator().manual_seed(42)
    return {
        "embed_tokens.weight": torch.randn(32, 16, generator=g),
        "fc.weight": torch.randn(16, 48, generator=g),
        f"{layer_prefix}input_layernorm.weight": torch.randn(16, generator=g),
        f"{layer_prefix}hidden_norm.weight": torch.randn(16, generator=g),
        f"{layer_prefix}post_attention_layernorm.weight": torch.randn(16, generator=g),
        f"{layer_prefix}self_attn.q_proj.weight": torch.randn(16, 32, generator=g),
        f"{layer_prefix}self_attn.k_proj.weight": torch.randn(8, 32, generator=g),
        f"{layer_prefix}self_attn.v_proj.weight": torch.randn(8, 32, generator=g),
        f"{layer_prefix}self_attn.o_proj.weight": torch.randn(16, 16, generator=g),
        f"{layer_prefix}mlp.gate_proj.weight": torch.randn(32, 16, generator=g),
        f"{layer_prefix}mlp.up_proj.weight": torch.randn(32, 16, generator=g),
        f"{layer_prefix}mlp.down_proj.weight": torch.randn(16, 32, generator=g),
        "norm.weight": torch.randn(16, generator=g),
        "lm_head.weight": torch.randn(16, 16, generator=g),
        "d2t": torch.arange(16, dtype=torch.long),
        "t2d": torch.cat(
            [torch.ones(16, dtype=torch.bool), torch.zeros(16, dtype=torch.bool)]
        ),
    }


class TestSaveConvertedCheckpoint:
    """Tests for missing/unexpected key handling in _save_converted_checkpoint."""

    @pytest.mark.sanity
    def test_valid_checkpoint_saves_exact_weights(self, tmp_path):
        """All expected keys load into Eagle3Speculator and persist unchanged."""
        weights = _tiny_eagle3_weights()
        out_dir = tmp_path / "converted"

        Eagle3Converter()._save_converted_checkpoint(
            _tiny_eagle3_speculator_config(),
            weights,
            out_dir,
            reduce_vocab_size=True,
            has_drafter_embedding=True,
        )

        with safe_open(str(out_dir / "model.safetensors"), framework="pt") as f:
            saved = {k: f.get_tensor(k) for k in f.keys()}  # noqa: SIM118

        for key, tensor in weights.items():
            expected_key = (
                key.replace("midlayer.", "layers.0.", 1)
                if key.startswith("midlayer.")
                else key
            )
            assert torch.equal(saved[expected_key], tensor)

    @pytest.mark.sanity
    def test_valid_checkpoint_without_optional_vocab_or_embed(self, tmp_path):
        """Checkpoints without t2d/d2t or embed_tokens.weight convert cleanly."""
        weights = _tiny_eagle3_weights()
        for optional_key in ("embed_tokens.weight", "d2t", "t2d"):
            weights.pop(optional_key)
        out_dir = tmp_path / "converted"

        Eagle3Converter()._save_converted_checkpoint(
            _tiny_eagle3_speculator_config(),
            weights,
            out_dir,
            reduce_vocab_size=False,
            has_drafter_embedding=False,
        )

        assert (out_dir / "model.safetensors").exists()

    @pytest.mark.sanity
    @pytest.mark.parametrize(
        "dropped_key",
        ["midlayer.self_attn.q_proj.weight", "fc.weight", "d2t"],
    )
    def test_missing_key_raises_and_writes_nothing(self, tmp_path, dropped_key):
        """Missing checkpoint keys must raise ValueError without writing files."""
        weights = _tiny_eagle3_weights()
        weights.pop(dropped_key)
        out_dir = tmp_path / "converted"

        with pytest.raises(ValueError, match="Missing required keys"):
            Eagle3Converter()._save_converted_checkpoint(
                _tiny_eagle3_speculator_config(),
                weights,
                out_dir,
                reduce_vocab_size=True,
                has_drafter_embedding=True,
            )

        assert not out_dir.exists()

    @pytest.mark.sanity
    def test_unexpected_keys_warn_and_still_save(self, tmp_path):
        """2-layer checkpoints carry layers.1.* keys and must still convert."""
        weights = _tiny_eagle3_weights(layer_prefix="layers.0.")
        weights["layers.1.self_attn.q_proj.weight"] = torch.randn(16, 32)
        out_dir = tmp_path / "converted"

        with patch("speculators.convert.eagle.eagle3_converter.logger") as mock_logger:
            Eagle3Converter()._save_converted_checkpoint(
                _tiny_eagle3_speculator_config(),
                weights,
                out_dir,
                reduce_vocab_size=True,
                has_drafter_embedding=True,
            )

        assert (out_dir / "model.safetensors").exists()
        assert any(
            "layers.1.self_attn.q_proj.weight" in call.args[0]
            for call in mock_logger.warning.call_args_list
        )

    @pytest.mark.sanity
    def test_remap_collision_raises(self, tmp_path):
        """Both midlayer.* and layers.0.* for the same tensor must fail loudly."""
        weights = _tiny_eagle3_weights(layer_prefix="midlayer.")
        weights["layers.0.self_attn.q_proj.weight"] = torch.randn(16, 32)
        out_dir = tmp_path / "converted"

        with pytest.raises(ValueError, match="Key collision"):
            Eagle3Converter()._save_converted_checkpoint(
                _tiny_eagle3_speculator_config(),
                weights,
                out_dir,
                reduce_vocab_size=True,
                has_drafter_embedding=True,
            )

        assert not out_dir.exists()

    @pytest.mark.sanity
    def test_convert_mismatched_key_fails_end_to_end(self, tmp_path):
        """End-to-end convert() fails when a checkpoint key does not match."""
        src_dir = tmp_path / "source"
        src_dir.mkdir()
        out_dir = tmp_path / "converted"

        (src_dir / "config.json").write_text(
            json.dumps(
                {
                    "hidden_size": 16,
                    "intermediate_size": 32,
                    "num_hidden_layers": 1,
                    "num_attention_heads": 2,
                    "num_key_value_heads": 1,
                    "head_dim": 8,
                    "draft_vocab_size": 16,
                    "max_position_embeddings": 64,
                }
            )
        )
        weights = _tiny_eagle3_weights()
        q = weights.pop("midlayer.self_attn.q_proj.weight")
        weights["model.layers.0.self_attn.q_proj.weight"] = q
        save_file(weights, str(src_dir / "model.safetensors"))

        with (
            patch(
                "speculators.convert.eagle.eagle3_converter.PretrainedConfig.get_config_dict",
                return_value=(
                    {
                        "architectures": ["LlamaForCausalLM"],
                        "vocab_size": 32,
                        "max_position_embeddings": 64,
                    },
                    None,
                ),
            ),
            pytest.raises(ValueError, match="Missing required keys"),
        ):
            Eagle3Converter().convert(
                src_dir,
                out_dir,
                base_model="dummy/verifier",
                validate=False,
            )

        assert not out_dir.exists()
