"""
Unit tests for the Eagle converter module in the Speculators library.
"""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch
from transformers import LlamaConfig, PretrainedConfig, PreTrainedModel

from speculators import SpeculatorsConfig, VerifierConfig
from speculators.convert.converters import EagleSpeculatorConverter, SpeculatorConverter
from speculators.models import EagleSpeculator, EagleSpeculatorConfig

# ===== Test Fixtures =====


@pytest.fixture
def mock_eagle_model():
    """Mock Eagle model for testing."""
    model = MagicMock(spec=PreTrainedModel)
    model.config = MagicMock(spec=PretrainedConfig)
    return model


@pytest.fixture
def mock_eagle_config():
    """Mock Eagle configuration dictionary."""
    return {
        "vocab_size": 32000,
        "hidden_size": 4096,
        "intermediate_size": 11008,
        "num_attention_heads": 32,
        "num_key_value_heads": 32,
        "hidden_act": "silu",
        "max_position_embeddings": 4096,
        "initializer_range": 0.02,
        "rms_norm_eps": 1e-6,
        "use_cache": True,
        "pad_token_id": None,
        "bos_token_id": 1,
        "eos_token_id": 2,
        "rope_theta": 10000.0,
        "rope_scaling": None,
        "attention_bias": False,
        "attention_dropout": 0.0,
        "mlp_bias": False,
    }


@pytest.fixture
def mock_eagle_state_dict():
    """Mock Eagle state dictionary with typical Eagle weights."""
    return {
        "fc.weight": torch.randn(32000, 4096),
        "fc.bias": torch.randn(32000),
        "layers.0.self_attn.q_proj.weight": torch.randn(4096, 4096),
        "layers.0.self_attn.k_proj.weight": torch.randn(4096, 4096),
        "layers.0.self_attn.v_proj.weight": torch.randn(4096, 4096),
        "layers.0.self_attn.o_proj.weight": torch.randn(4096, 4096),
        "layers.0.mlp.gate_proj.weight": torch.randn(11008, 4096),
        "layers.0.mlp.up_proj.weight": torch.randn(11008, 4096),
        "layers.0.mlp.down_proj.weight": torch.randn(4096, 11008),
        "layers.0.input_layernorm.weight": torch.randn(4096),
        "layers.0.post_attention_layernorm.weight": torch.randn(4096),
        "embed_tokens.weight": torch.randn(32000, 4096),
        "embed_layernorm.weight": torch.randn(4096),
        "hidden_layernorm.weight": torch.randn(4096),
        "lm_head_layernorm.weight": torch.randn(4096),
    }


@pytest.fixture
def mock_eagle_state_dict_minimal():
    """Mock minimal Eagle state dictionary without optional components."""
    return {
        "fc.weight": torch.randn(32000, 4096),
        "layers.0.self_attn.q_proj.weight": torch.randn(4096, 4096),
        "layers.0.self_attn.k_proj.weight": torch.randn(4096, 4096),
        "layers.0.self_attn.v_proj.weight": torch.randn(4096, 4096),
        "layers.0.self_attn.o_proj.weight": torch.randn(4096, 4096),
        "layers.0.mlp.gate_proj.weight": torch.randn(11008, 4096),
        "layers.0.mlp.up_proj.weight": torch.randn(11008, 4096),
        "layers.0.mlp.down_proj.weight": torch.randn(4096, 11008),
        "layers.0.input_layernorm.weight": torch.randn(4096),
        "layers.0.post_attention_layernorm.weight": torch.randn(4096),
        "embed_tokens.weight": torch.randn(32000, 4096),
    }


@pytest.fixture
def mock_verifier():
    """Create a mock verifier with proper config attribute for testing."""
    verifier = MagicMock()
    verifier._spec_class = PreTrainedModel
    verifier.config = MagicMock()
    verifier.config._spec_class = PretrainedConfig
    verifier.config.architectures = ["TestModel"]
    verifier.config.name_or_path = "test/model"
    verifier.config.to_dict.return_value = {
        "architectures": ["TestModel"],
        "name_or_path": "test/model",
        "_name_or_path": "test/model",
    }
    verifier.name_or_path = "test/model"
    verifier.smart_apply = MagicMock()
    verifier.apply = MagicMock()
    verifier.state_dict = MagicMock(return_value={})

    return verifier


@pytest.fixture
def mock_eagle_speculator():
    """Mock Eagle speculator model for testing."""
    model = MagicMock(spec=EagleSpeculator)
    model.save_pretrained = MagicMock()

    # Mock config for validation
    mock_config = MagicMock()
    mock_transformer_config = MagicMock()
    mock_transformer_config.vocab_size = 32000
    mock_transformer_config.hidden_size = 4096
    mock_transformer_config.max_position_embeddings = 4096
    mock_config.transformer_layer_config = mock_transformer_config
    model.config = mock_config

    # Mock to method for device movement
    model.to = MagicMock(return_value=model)

    return model


@pytest.fixture
def temp_directory():
    """Temporary directory for testing file operations."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield temp_dir


# ===== Test Classes =====


class TestEagleSpeculatorConverter:
    """Test class for EagleSpeculatorConverter functionality."""

    @pytest.mark.smoke
    def test_registration(self):
        """Test that EagleSpeculatorConverter is properly registered."""
        assert SpeculatorConverter.registry is not None
        assert "eagle" in SpeculatorConverter.registry
        assert "eagle2" in SpeculatorConverter.registry
        assert "hass" in SpeculatorConverter.registry
        assert SpeculatorConverter.registry["eagle"] is EagleSpeculatorConverter

    @pytest.mark.smoke
    @pytest.mark.parametrize(
        (
            "verifier",
            "fusion_bias",
            "layernorms",
            "expected_fusion_bias",
            "expected_layernorms",
        ),
        [
            ("mock_verifier", None, None, None, None),  # Basic initialization
            (None, True, False, True, False),  # With features
        ],
    )
    def test_initialization(
        self,
        mock_eagle_model,
        mock_eagle_config,
        mock_verifier,
        verifier,
        fusion_bias,
        layernorms,
        expected_fusion_bias,
        expected_layernorms,
    ):
        """Test initialization of EagleSpeculatorConverter."""
        actual_verifier = mock_verifier if verifier == "mock_verifier" else None

        converter = EagleSpeculatorConverter(
            mock_eagle_model,
            mock_eagle_config,
            actual_verifier,
            fusion_bias=fusion_bias,
            layernorms=layernorms,
        )

        assert converter.model is mock_eagle_model
        assert converter.config is mock_eagle_config
        assert converter.verifier is actual_verifier
        assert converter.fusion_bias is expected_fusion_bias
        assert converter.layernorms is expected_layernorms

    @pytest.mark.smoke
    @pytest.mark.parametrize(
        ("model", "config"),
        [
            (None, "valid_config"),
            ("valid_model", None),
            ("", "valid_config"),
            ("valid_model", ""),
        ],
    )
    def test_initialization_invalid(
        self, mock_eagle_model, mock_eagle_config, model, config
    ):
        """Test initialization fails with invalid inputs."""
        actual_model = mock_eagle_model if model == "valid_model" else model
        actual_config = mock_eagle_config if config == "valid_config" else config

        with pytest.raises(ValueError) as exc_info:
            EagleSpeculatorConverter(actual_model, actual_config, None)

        assert "Model and config paths must be provided" in str(exc_info.value)

    @pytest.mark.smoke
    @patch("speculators.convert.converters.eagle.load_model_checkpoint_state_dict")
    def test_is_supported_valid_eagle(
        self, mock_load_state_dict, mock_eagle_state_dict
    ):
        """Test is_supported returns True for valid Eagle checkpoints."""
        mock_load_state_dict.return_value = mock_eagle_state_dict

        result = EagleSpeculatorConverter.is_supported(
            "path/to/model", "path/to/config"
        )

        assert result is True
        mock_load_state_dict.assert_called_once_with("path/to/model")

    @pytest.mark.smoke
    @pytest.mark.parametrize(
        ("test_case", "state_dict"),
        [
            (
                "no_fc",
                {
                    "layers.0.self_attn.q_proj.weight": torch.randn(4096, 4096),
                    "layers.0.self_attn.k_proj.weight": torch.randn(4096, 4096),
                },
            ),
            (
                "no_layers_0",
                {
                    "fc.weight": torch.randn(32000, 4096),
                    "layers.1.self_attn.q_proj.weight": torch.randn(4096, 4096),
                },
            ),
            (
                "multiple_layers",
                {
                    "fc.weight": torch.randn(32000, 4096),
                    "layers.0.self_attn.q_proj.weight": torch.randn(4096, 4096),
                    "layers.1.self_attn.q_proj.weight": torch.randn(4096, 4096),
                },
            ),
        ],
    )
    @patch("speculators.convert.converters.eagle.load_model_checkpoint_state_dict")
    def test_is_supported_invalid(self, mock_load_state_dict, test_case, state_dict):
        """Test is_supported returns False for invalid Eagle checkpoints."""
        mock_load_state_dict.return_value = state_dict

        result = EagleSpeculatorConverter.is_supported(
            "path/to/model", "path/to/config"
        )

        assert result is False

    @pytest.mark.sanity
    @pytest.mark.parametrize(
        (
            "config",
            "expected_vocab_size",
            "expected_hidden_size",
            "expected_intermediate_size",
        ),
        [
            (
                "mock_eagle_config",
                32000,
                4096,
                11008,
            ),
            (
                {},
                32000,
                4096,
                11008,
            ),
        ],
    )
    def test_pretrained_config_from_eagle(
        self,
        mock_eagle_config,
        config,
        expected_vocab_size,
        expected_hidden_size,
        expected_intermediate_size,
    ):
        """Test conversion of Eagle config to LlamaConfig."""
        converter = EagleSpeculatorConverter("model", "config", None)
        actual_config = mock_eagle_config if config == "mock_eagle_config" else config

        llama_config = converter._pretrained_config_from_eagle(actual_config)

        assert isinstance(llama_config, LlamaConfig)
        assert llama_config.vocab_size == expected_vocab_size
        assert llama_config.hidden_size == expected_hidden_size
        assert llama_config.intermediate_size == expected_intermediate_size
        assert llama_config.num_hidden_layers == 1  # Eagle always uses 1 layer
        assert llama_config.hidden_act == "silu"
        assert llama_config.tie_word_embeddings is False

    @pytest.mark.sanity
    @patch("speculators.convert.converters.eagle.VerifierConfig.from_pretrained")
    def test_eagle_speculator_config(
        self, mock_verifier_from_pretrained, mock_eagle_config
    ):
        """Test creation of EagleSpeculatorConfig."""
        mock_verifier_config = MagicMock(spec=VerifierConfig)
        mock_verifier_from_pretrained.return_value = mock_verifier_config

        converter = EagleSpeculatorConverter("model", "config", "verifier")

        config = converter._eagle_speculator_config(mock_eagle_config, True, True)

        assert isinstance(config, EagleSpeculatorConfig)
        assert isinstance(config.transformer_layer_config, LlamaConfig)
        assert isinstance(config.speculators_config, SpeculatorsConfig)
        assert config.layernorms is True
        assert config.fusion_bias is True
        assert config.speculators_config.algorithm == "eagle"
        assert config.speculators_config.default_proposal_method == "greedy"

    @pytest.mark.sanity
    @pytest.mark.parametrize(
        ("weight_name", "fusion_bias", "layernorms", "expected"),
        [
            ("embed_tokens.weight", True, True, "ignore"),  # Always ignore
            ("fc.bias", False, True, "extra"),  # Extra when fusion_bias=False
            ("fc.bias", True, True, "keep"),  # Keep when fusion_bias=True
            (
                "embed_layernorm.weight",
                True,
                False,
                "extra",
            ),  # Extra when layernorms=False
            (
                "embed_layernorm.weight",
                True,
                True,
                "keep",
            ),  # Keep when layernorms=True
            ("unknown.weight", True, True, "extra"),  # Extra for unmapped weights
            ("fc.weight", True, True, "keep"),  # Keep mapped weights
            (
                "layers.0.self_attn.q_proj.weight",
                True,
                True,
                "keep",
            ),  # Keep mapped weights
        ],
    )
    def test_classify_param_key(self, weight_name, fusion_bias, layernorms, expected):
        """Test parameter key classification logic."""
        converter = EagleSpeculatorConverter("model", "config", None)

        result = converter._classify_param_key(weight_name, fusion_bias, layernorms)

        assert result == expected

    @pytest.mark.sanity
    @pytest.mark.parametrize(
        ("weight_name", "expected"),
        [
            ("fc.weight", "fusion_fc.weight"),
            ("fc.bias", "fusion_fc.bias"),
            ("layers.0.self_attn.q_proj.weight", "transformer.self_attn.q_proj.weight"),
            ("layers.0.mlp.gate_proj.weight", "transformer.mlp.gate_proj.weight"),
            ("embed_layernorm.weight", "embedding_layernorm.weight"),
            ("hidden_layernorm.weight", "transformer.input_layernorm.weight"),
            ("lm_head_layernorm.weight", "pre_lm_head_layernorm.weight"),
        ],
    )
    def test_remap_param_name(self, weight_name, expected):
        """Test parameter name remapping."""
        converter = EagleSpeculatorConverter("model", "config", None)

        result = converter._remap_param_name(weight_name)

        assert result == expected

    @pytest.mark.sanity
    def test_remap_param_name_invalid(self):
        """Test parameter name remapping with invalid name."""
        converter = EagleSpeculatorConverter("model", "config", None)

        with pytest.raises(ValueError) as exc_info:
            converter._remap_param_name("unknown.weight")

        assert "Unexpected parameter name format" in str(exc_info.value)

    @pytest.mark.sanity
    @pytest.mark.parametrize(
        (
            "state_dict_fixture",
            "fusion_bias",
            "layernorms",
            "expected_fusion_bias",
            "expected_layernorms",
            "expected_extra_count",
        ),
        [
            (
                "mock_eagle_state_dict",
                True,
                True,
                True,
                True,
                0,
            ),
            (
                "mock_eagle_state_dict_minimal",
                False,
                False,
                False,
                False,
                0,
            ),
            (
                "invalid_state_dict",
                True,
                True,
                True,
                False,
                1,
            ),
        ],
    )
    def test_eagle_speculator_state_dict(
        self,
        mock_eagle_state_dict,
        mock_eagle_state_dict_minimal,
        state_dict_fixture,
        fusion_bias,
        layernorms,
        expected_fusion_bias,
        expected_layernorms,
        expected_extra_count,
    ):
        """Test state dict conversion with different configurations."""
        converter = EagleSpeculatorConverter("model", "config", None)

        # Select the appropriate state dict
        if state_dict_fixture == "mock_eagle_state_dict":
            state_dict = mock_eagle_state_dict
        elif state_dict_fixture == "mock_eagle_state_dict_minimal":
            state_dict = mock_eagle_state_dict_minimal
        else:  # invalid_state_dict
            state_dict = {
                "fc.weight": torch.randn(32000, 4096),
                "invalid.weight": torch.randn(100, 100),
            }

        converted_state_dict, extra = converter._eagle_speculator_state_dict(
            state_dict, fusion_bias=fusion_bias, layernorms=layernorms
        )

        # Check fusion_fc.weight is always included
        assert "fusion_fc.weight" in converted_state_dict

        # Check fusion_fc.bias based on fusion_bias setting AND whether it exists in
        # original state dict
        if expected_fusion_bias and "fc.bias" in state_dict:
            assert "fusion_fc.bias" in converted_state_dict
        else:
            assert "fusion_fc.bias" not in converted_state_dict

        # Check transformer weights are included (except for invalid case)
        if state_dict_fixture != "invalid_state_dict":
            assert "transformer.self_attn.q_proj.weight" in converted_state_dict

        # Check layernorms based on layernorms setting
        if expected_layernorms and state_dict_fixture == "mock_eagle_state_dict":
            assert "embedding_layernorm.weight" in converted_state_dict
        else:
            assert "embedding_layernorm.weight" not in converted_state_dict

        # Check embed_tokens is ignored (not included in converted_state_dict)
        assert "embed_tokens.weight" not in converted_state_dict

        # Check extra keys count
        assert len(extra) == expected_extra_count

        # For invalid case, check specific behavior
        if state_dict_fixture == "invalid_state_dict":
            assert "invalid.weight" in extra
            assert "invalid.weight" not in converted_state_dict

    @pytest.mark.sanity
    @pytest.mark.parametrize(
        (
            "explicit_fusion_bias",
            "explicit_layernorms",
            "expected_fusion_bias",
            "expected_layernorms",
        ),
        [
            (None, None, True, True),  # Auto-detection
            (False, False, False, False),  # Explicit settings
        ],
    )
    @patch("speculators.convert.converters.eagle.load_model_checkpoint_state_dict")
    @patch("speculators.convert.converters.eagle.load_model_checkpoint_config_dict")
    @patch("speculators.convert.converters.eagle.VerifierConfig.from_pretrained")
    def test_convert_config_state_dict(
        self,
        mock_verifier_from_pretrained,
        mock_load_config,
        mock_load_state_dict,
        mock_eagle_config,
        mock_eagle_state_dict,
        explicit_fusion_bias,
        explicit_layernorms,
        expected_fusion_bias,
        expected_layernorms,
    ):
        """Test the complete conversion process."""
        mock_load_config.return_value = mock_eagle_config
        mock_load_state_dict.return_value = mock_eagle_state_dict
        mock_verifier_config = MagicMock(spec=VerifierConfig)
        mock_verifier_from_pretrained.return_value = mock_verifier_config

        converter = EagleSpeculatorConverter(
            "model",
            "config",
            "verifier",
            fusion_bias=explicit_fusion_bias,
            layernorms=explicit_layernorms,
        )

        config, state_dict = converter.convert_config_state_dict()

        assert isinstance(config, EagleSpeculatorConfig)
        assert isinstance(state_dict, dict)
        assert len(state_dict) > 0

        # Check feature settings
        assert config.fusion_bias is expected_fusion_bias
        assert config.layernorms is expected_layernorms

    @pytest.mark.sanity
    @pytest.mark.parametrize(
        ("device", "should_fail", "skip_if_no_cuda"),
        [
            ("cpu", False, False),
            ("cuda", False, True),
            ("cpu", True, False),
        ],
    )
    def test_validate(
        self, mock_eagle_speculator, device, should_fail, skip_if_no_cuda
    ):
        """Test validation with different devices and failure scenarios."""
        if skip_if_no_cuda and not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        converter = EagleSpeculatorConverter("model", "config", None)

        if should_fail:
            # Make the model call raise an exception
            mock_eagle_speculator.side_effect = RuntimeError("Model forward failed")

            with pytest.raises(RuntimeError) as exc_info:
                converter.validate(mock_eagle_speculator, device)

            assert "Model forward failed" in str(exc_info.value)
        else:
            # Should not raise any exception
            converter.validate(mock_eagle_speculator, device)

            # Check that model was moved to device and back
            assert mock_eagle_speculator.to.call_count == 2
            mock_eagle_speculator.to.assert_any_call(device)
            mock_eagle_speculator.to.assert_any_call("cpu")

    @pytest.mark.sanity
    @pytest.mark.parametrize(
        ("output_path", "validate_device", "expect_save", "expect_validate"),
        [
            ("temp_directory", "cpu", True, True),  # Full conversion
            (None, None, False, False),  # No save or validate
        ],
    )
    @patch("speculators.convert.converters.eagle.load_model_checkpoint_state_dict")
    @patch("speculators.convert.converters.eagle.load_model_checkpoint_config_dict")
    @patch("speculators.convert.converters.eagle.VerifierConfig.from_pretrained")
    @patch("speculators.models.eagle.EagleSpeculator.from_pretrained")
    def test_conversion_call(
        self,
        mock_from_pretrained,
        mock_verifier_from_pretrained,
        mock_load_config,
        mock_load_state_dict,
        mock_eagle_config,
        mock_eagle_state_dict,
        mock_eagle_speculator,
        temp_directory,
        output_path,
        validate_device,
        expect_save,
        expect_validate,
    ):
        """Test complete conversion call workflow."""
        mock_load_config.return_value = mock_eagle_config
        mock_load_state_dict.return_value = mock_eagle_state_dict
        mock_verifier_config = MagicMock(spec=VerifierConfig)
        mock_verifier_from_pretrained.return_value = mock_verifier_config
        mock_from_pretrained.return_value = mock_eagle_speculator

        converter = EagleSpeculatorConverter("model", "config", "verifier")

        # Use temp_directory if output_path is "temp_directory"
        actual_output_path = (
            temp_directory if output_path == "temp_directory" else output_path
        )

        result = converter(
            output_path=actual_output_path, validate_device=validate_device
        )

        assert result is mock_eagle_speculator

        if expect_save:
            mock_eagle_speculator.save_pretrained.assert_called_once_with(
                actual_output_path
            )
        else:
            mock_eagle_speculator.save_pretrained.assert_not_called()

        if expect_validate:
            # Validate should have been called (moves to device and back)
            assert mock_eagle_speculator.to.call_count == 2
        else:
            mock_eagle_speculator.to.assert_not_called()

    @pytest.mark.regression
    @patch("speculators.convert.converters.eagle.load_model_checkpoint_state_dict")
    def test_is_supported_load_error(self, mock_load_state_dict):
        """Test is_supported handles load errors gracefully."""
        mock_load_state_dict.side_effect = FileNotFoundError("Model not found")

        with pytest.raises(FileNotFoundError):
            EagleSpeculatorConverter.is_supported("invalid/path", "config")

    @pytest.mark.regression
    @patch("speculators.convert.converters.eagle.load_model_checkpoint_state_dict")
    @patch("speculators.convert.converters.eagle.load_model_checkpoint_config_dict")
    def test_convert_config_state_dict_load_error(
        self, mock_load_config, mock_load_state_dict
    ):
        """Test convert_config_state_dict handles load errors."""
        mock_load_state_dict.side_effect = FileNotFoundError("Model not found")

        converter = EagleSpeculatorConverter("model", "config", None)

        with pytest.raises(FileNotFoundError):
            converter.convert_config_state_dict()

    @pytest.mark.smoke
    @pytest.mark.parametrize(
        ("path_type", "expected_path_type"),
        [
            ("string", str),
            ("path_object", Path),
        ],
    )
    def test_save_method(
        self, mock_eagle_speculator, temp_directory, path_type, expected_path_type
    ):
        """Test save method with different path types."""
        converter = EagleSpeculatorConverter("model", "config", None)

        path = temp_directory if path_type == "string" else Path(temp_directory)

        converter.save(mock_eagle_speculator, path)

        mock_eagle_speculator.save_pretrained.assert_called_once_with(path)
        assert isinstance(path, expected_path_type)

    @pytest.mark.smoke
    @pytest.mark.parametrize(
        "algorithm",
        ["eagle", "auto"],
    )
    def test_resolve_converter(self, algorithm):
        """Test resolve_converter returns EagleSpeculatorConverter."""
        mock_state_dict = {
            "fc.weight": torch.randn(32000, 4096),
            "layers.0.self_attn.q_proj.weight": torch.randn(4096, 4096),
        }

        with patch(
            "speculators.convert.converters.eagle.load_model_checkpoint_state_dict"
        ) as mock_load:
            mock_load.return_value = mock_state_dict

            converter_class = SpeculatorConverter.resolve_converter(
                algorithm, "path/to/model", "path/to/config"
            )

            assert converter_class is EagleSpeculatorConverter
