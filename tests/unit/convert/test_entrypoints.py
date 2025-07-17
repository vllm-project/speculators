"""
Unit tests for the entrypoints module in the Speculators library.

This module tests the convert_model function which serves as the main entry point
for converting external research model checkpoints into the Speculators format.
"""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch
from torch import nn
from transformers import PretrainedConfig, PreTrainedModel

from speculators.convert.converters import SpeculatorConverter
from speculators.convert.entrypoints import convert_model
from speculators.model import SpeculatorModel

# ===== Test Fixtures =====


@pytest.fixture
def mock_pretrained_model():
    """Mock PreTrainedModel instance."""
    model = MagicMock(spec=PreTrainedModel)
    model.config = MagicMock(spec=PretrainedConfig)
    model.state_dict.return_value = {"test_param": torch.tensor([1.0, 2.0])}
    return model


@pytest.fixture
def mock_nn_module():
    """Mock nn.Module instance."""
    module = MagicMock(spec=nn.Module)
    module.state_dict.return_value = {"test_param": torch.tensor([1.0, 2.0])}
    return module


@pytest.fixture
def mock_pretrained_config():
    """Mock PretrainedConfig instance."""
    config = MagicMock(spec=PretrainedConfig)
    config.to_dict.return_value = {
        "model_type": "test_model",
        "hidden_size": 768,
        "vocab_size": 50000,
    }
    return config


@pytest.fixture
def mock_speculator_model():
    """Mock SpeculatorModel instance."""
    model = MagicMock(spec=SpeculatorModel)
    model.save_pretrained = MagicMock()
    return model


@pytest.fixture
def mock_converter():
    """Mock SpeculatorConverter instance."""
    converter = MagicMock(spec=SpeculatorConverter)
    converter.return_value = MagicMock(spec=SpeculatorModel)
    return converter


@pytest.fixture
def temp_directory():
    """Create a temporary directory for testing."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)


@pytest.fixture
def sample_config_dict():
    """Sample configuration dictionary."""
    return {
        "model_type": "llama",
        "hidden_size": 4096,
        "vocab_size": 32000,
        "num_hidden_layers": 32,
        "num_attention_heads": 32,
        "intermediate_size": 11008,
        "max_position_embeddings": 4096,
        "rms_norm_eps": 1e-6,
        "bos_token_id": 1,
        "eos_token_id": 2,
    }


# ===== Main Test Class =====


class TestConvertModel:
    """Test class for convert_model function."""

    @pytest.mark.parametrize(
        (
            "model_input",
            "config_input",
            "output_path",
            "verifier",
            "validate_device",
            "algorithm",
            "algorithm_kwargs",
            "cache_dir",
            "force_download",
            "local_files_only",
            "token",
            "revision",
            "extra_kwargs",
            "expected_config_source",
        ),
        [
            # Basic functionality tests
            (
                "/path/to/model",
                "/path/to/config",
                None,
                None,
                None,
                "auto",
                None,
                None,
                False,
                False,
                None,
                None,
                {},
                "/path/to/config",
            ),
            # Config auto-inference
            (
                "/path/to/model",
                None,
                None,
                None,
                None,
                "auto",
                None,
                None,
                False,
                False,
                None,
                None,
                {},
                "/path/to/model",
            ),
            # With output path and validation
            (
                "/path/to/model",
                "/path/to/config",
                "/output/path",
                None,
                "cuda",
                "eagle",
                None,
                None,
                False,
                False,
                None,
                None,
                {},
                "/path/to/config",
            ),
            # With verifier
            (
                "/path/to/model",
                "/path/to/config",
                None,
                "/path/to/verifier",
                None,
                "auto",
                None,
                None,
                False,
                False,
                None,
                None,
                {},
                "/path/to/config",
            ),
            # With algorithm kwargs
            (
                "/path/to/model",
                "/path/to/config",
                None,
                None,
                None,
                "eagle",
                {"fusion_bias": True, "layernorms": False},
                None,
                False,
                False,
                None,
                None,
                {},
                "/path/to/config",
            ),
            # With download parameters
            (
                "/path/to/model",
                "/path/to/config",
                None,
                None,
                None,
                "auto",
                None,
                "/cache/dir",
                True,
                False,
                "token123",
                "v1.0",
                {"custom_param": "value"},
                "/path/to/config",
            ),
            # EAGLE2 algorithm
            (
                "/path/to/model",
                "/path/to/config",
                None,
                None,
                None,
                "eagle2",
                None,
                None,
                False,
                False,
                None,
                None,
                {},
                "/path/to/config",
            ),
            # HASS algorithm
            (
                "/path/to/model",
                "/path/to/config",
                None,
                None,
                None,
                "hass",
                None,
                None,
                False,
                False,
                None,
                None,
                {},
                "/path/to/config",
            ),
            # Complex scenario with all parameters
            (
                "/path/to/model",
                "/path/to/config",
                "/output/path",
                "/path/to/verifier",
                "cpu",
                "eagle",
                {"fusion_bias": True, "layernorms": True},
                "/cache/dir",
                False,
                True,
                True,
                "main",
                {"custom_param": "value"},
                "/path/to/config",
            ),
        ],
    )
    def test_general(
        self,
        model_input,
        config_input,
        output_path,
        verifier,
        validate_device,
        algorithm,
        algorithm_kwargs,
        cache_dir,
        force_download,
        local_files_only,
        token,
        revision,
        extra_kwargs,
        expected_config_source,
        mock_speculator_model,
        temp_directory,
    ):
        """
        Test general convert_model functionality with various parameter combinations.
        """
        with (
            patch(
                "speculators.convert.entrypoints.check_download_model_checkpoint"
            ) as mock_check_model,
            patch(
                "speculators.convert.entrypoints.check_download_model_config"
            ) as mock_check_config,
            patch(
                "speculators.convert.entrypoints.SpeculatorConverter.resolve_converter"
            ) as mock_resolve,
        ):
            # Set up mocks
            mock_check_model.return_value = model_input
            mock_check_config.return_value = expected_config_source

            mock_converter_class = MagicMock()
            mock_converter_instance = MagicMock()
            mock_converter_instance.return_value = mock_speculator_model
            mock_converter_class.return_value = mock_converter_instance
            mock_resolve.return_value = mock_converter_class

            # Handle temp directory fixture replacement
            if output_path == "/output/path":
                output_path = temp_directory / "output"

            # Build kwargs for convert_model call
            kwargs = {
                "model": model_input,
                "algorithm": algorithm,
                "cache_dir": cache_dir,
                "force_download": force_download,
                "local_files_only": local_files_only,
                "token": token,
                "revision": revision,
                **extra_kwargs,
            }

            # Add optional parameters
            if config_input is not None:
                kwargs["config"] = config_input
            if output_path is not None:
                kwargs["output_path"] = output_path
            if verifier is not None:
                kwargs["verifier"] = verifier
            if validate_device is not None:
                kwargs["validate_device"] = validate_device
            if algorithm_kwargs is not None:
                kwargs["algorithm_kwargs"] = algorithm_kwargs

            # Call the function
            result = convert_model(**kwargs)

            # Verify result
            assert result is mock_speculator_model

            # Verify check_download_model_checkpoint was called correctly
            mock_check_model.assert_called_once_with(
                model_input,
                cache_dir=cache_dir,
                force_download=force_download,
                local_files_only=local_files_only,
                token=token,
                revision=revision,
                **extra_kwargs,
            )

            # Verify check_download_model_config was called correctly
            mock_check_config.assert_called_once_with(
                expected_config_source,
                cache_dir=cache_dir,
                force_download=force_download,
                local_files_only=local_files_only,
                token=token,
                revision=revision,
                **extra_kwargs,
            )

            # Build expected resolve_converter call args
            expected_resolve_args = {
                "model": model_input,
                "config": expected_config_source,
                "verifier": verifier,
            }
            if algorithm_kwargs:
                expected_resolve_args.update(algorithm_kwargs)

            mock_resolve.assert_called_once_with(algorithm, **expected_resolve_args)

            # Verify converter instantiation
            expected_converter_args = {
                "model": model_input,
                "config": expected_config_source,
                "verifier": verifier,
            }
            if algorithm_kwargs:
                expected_converter_args.update(algorithm_kwargs)

            mock_converter_class.assert_called_once_with(**expected_converter_args)

            # Verify converter call
            mock_converter_instance.assert_called_once_with(
                output_path=output_path,
                validate_device=validate_device,
            )

    @pytest.mark.parametrize(
        (
            "test_case",
            "model_input",
            "config_input",
            "exception_stage",
            "exception_type",
            "exception_message",
            "setup_mocks",
        ),
        [
            # nn.Module without config
            (
                "nn_module_no_config",
                "nn_module",
                None,
                "config_validation",
                ValueError,
                (
                    "A model config must be provided when converting "
                    "a PyTorch nn.Module instance"
                ),
                lambda mock_nn_module: {"model_return": mock_nn_module},
            ),
            # Model checkpoint resolution error
            (
                "model_checkpoint_error",
                "/invalid/model",
                "/path/to/config",
                "check_download_model_checkpoint",
                FileNotFoundError,
                "Model not found",
                lambda mock_nn_module: {"model_exception": True},
            ),
            # Config resolution error
            (
                "config_resolution_error",
                "/path/to/model",
                "/invalid/config",
                "check_download_model_config",
                FileNotFoundError,
                "Config not found",
                lambda mock_nn_module: {"config_exception": True},
            ),
            # Converter resolution error
            (
                "converter_resolution_error",
                "/path/to/model",
                "/path/to/config",
                "resolve_converter",
                ValueError,
                "No supported converter found",
                lambda mock_nn_module: {"converter_exception": True},
            ),
            # Converter instantiation error
            (
                "converter_instantiation_error",
                "/path/to/model",
                "/path/to/config",
                "converter_class",
                ValueError,
                "Invalid converter parameters",
                lambda mock_nn_module: {"converter_class_exception": True},
            ),
            # Conversion error
            (
                "conversion_error",
                "/path/to/model",
                "/path/to/config",
                "converter_instance",
                RuntimeError,
                "Conversion failed",
                lambda mock_nn_module: {"converter_instance_exception": True},
            ),
        ],
    )
    def test_invalid(
        self,
        test_case,
        model_input,
        config_input,
        exception_stage,
        exception_type,
        exception_message,
        setup_mocks,
        mock_nn_module,
    ):
        """Test convert_model with invalid parameters and expected error handling."""
        with (
            patch(
                "speculators.convert.entrypoints.check_download_model_checkpoint"
            ) as mock_check_model,
            patch(
                "speculators.convert.entrypoints.check_download_model_config"
            ) as mock_check_config,
            patch(
                "speculators.convert.entrypoints.SpeculatorConverter.resolve_converter"
            ) as mock_resolve,
        ):
            # Setup mocks based on test case
            mock_setup = setup_mocks(mock_nn_module)

            # Handle special case for nn.Module
            if model_input == "nn_module":
                model_input = mock_nn_module

            # Configure mocks based on exception stage
            if exception_stage == "check_download_model_checkpoint" or mock_setup.get(
                "model_exception"
            ):
                mock_check_model.side_effect = exception_type(exception_message)
            elif mock_setup.get("model_return"):
                mock_check_model.return_value = mock_setup["model_return"]
            else:
                mock_check_model.return_value = model_input

            if exception_stage == "check_download_model_config" or mock_setup.get(
                "config_exception"
            ):
                mock_check_config.side_effect = exception_type(exception_message)
            else:
                mock_check_config.return_value = config_input or model_input

            if exception_stage == "resolve_converter" or mock_setup.get(
                "converter_exception"
            ):
                mock_resolve.side_effect = exception_type(exception_message)
            else:
                mock_converter_class = MagicMock()
                mock_converter_instance = MagicMock()

                if exception_stage == "converter_class" or mock_setup.get(
                    "converter_class_exception"
                ):
                    mock_converter_class.side_effect = exception_type(exception_message)
                elif exception_stage == "converter_instance" or mock_setup.get(
                    "converter_instance_exception"
                ):
                    mock_converter_instance.side_effect = exception_type(
                        exception_message
                    )

                mock_converter_class.return_value = mock_converter_instance
                mock_resolve.return_value = mock_converter_class

            # Build kwargs for convert_model call
            kwargs = {"model": model_input}
            if config_input is not None:
                kwargs["config"] = config_input

            # Expect the exception
            with pytest.raises(exception_type) as exc_info:
                convert_model(**kwargs)

            assert exception_message in str(exc_info.value)

    @pytest.mark.parametrize(
        (
            "algorithm",
            "model_path",
            "config_path",
            "verifier_path",
            "algorithm_kwargs",
            "output_path",
            "validate_device",
            "should_use_eagle_converter",
        ),
        [
            # Auto algorithm that resolves to eagle
            (
                "auto",
                "/path/to/eagle/model",
                "/path/to/eagle/config",
                None,
                None,
                None,
                None,
                True,
            ),
            # Explicit eagle algorithm
            (
                "eagle",
                "/path/to/eagle/model",
                "/path/to/eagle/config",
                None,
                None,
                None,
                None,
                True,
            ),
            # Eagle with verifier
            (
                "eagle",
                "/path/to/eagle/model",
                "/path/to/eagle/config",
                "/path/to/verifier",
                None,
                None,
                None,
                True,
            ),
            # Eagle with algorithm kwargs
            (
                "eagle",
                "/path/to/eagle/model",
                "/path/to/eagle/config",
                None,
                {"fusion_bias": True, "layernorms": False},
                None,
                None,
                True,
            ),
            # Eagle with output and validation
            (
                "eagle",
                "/path/to/eagle/model",
                "/path/to/eagle/config",
                None,
                None,
                "/output/path",
                "cuda",
                True,
            ),
            # Eagle2 algorithm
            (
                "eagle2",
                "/path/to/eagle2/model",
                "/path/to/eagle2/config",
                None,
                None,
                None,
                None,
                True,
            ),
            # HASS algorithm
            (
                "hass",
                "/path/to/hass/model",
                "/path/to/hass/config",
                None,
                {"fusion_bias": True},
                None,
                None,
                True,
            ),
            # Complex eagle scenario
            (
                "eagle",
                "/path/to/eagle/model",
                "/path/to/eagle/config",
                "/path/to/verifier",
                {"fusion_bias": True, "layernorms": True},
                "/output/path",
                "cpu",
                True,
            ),
        ],
    )
    def test_eagle(
        self,
        algorithm,
        model_path,
        config_path,
        verifier_path,
        algorithm_kwargs,
        output_path,
        validate_device,
        should_use_eagle_converter,
        mock_speculator_model,
        temp_directory,
    ):
        """
        Test convert_model with eagle algorithm and ensure proper Eagle converter usage.
        """
        with (
            patch(
                "speculators.convert.entrypoints.check_download_model_checkpoint"
            ) as mock_check_model,
            patch(
                "speculators.convert.entrypoints.check_download_model_config"
            ) as mock_check_config,
            patch(
                "speculators.convert.converters.eagle.EagleSpeculatorConverter"
            ) as mock_eagle_converter_class,
        ):
            # Set up mocks
            mock_check_model.return_value = model_path
            mock_check_config.return_value = config_path

            # Create a mock Eagle converter instance
            mock_eagle_converter_instance = MagicMock()
            mock_eagle_converter_instance.return_value = mock_speculator_model
            mock_eagle_converter_class.return_value = mock_eagle_converter_instance

            # Register the Eagle converter for the test
            with patch.object(
                SpeculatorConverter,
                "resolve_converter",
                return_value=mock_eagle_converter_class,
            ) as mock_resolve:
                # Handle temp directory fixture replacement
                if output_path == "/output/path":
                    output_path = temp_directory / "output"

                # Build kwargs for convert_model call
                kwargs = {
                    "model": model_path,
                    "config": config_path,
                    "algorithm": algorithm,
                }

                # Add optional parameters
                if verifier_path is not None:
                    kwargs["verifier"] = verifier_path
                if algorithm_kwargs is not None:
                    kwargs["algorithm_kwargs"] = algorithm_kwargs
                if output_path is not None:
                    kwargs["output_path"] = output_path
                if validate_device is not None:
                    kwargs["validate_device"] = validate_device

                # Call the function
                result = convert_model(**kwargs)

                # Verify result
                assert result is mock_speculator_model

                # Verify the Eagle converter was resolved
                if should_use_eagle_converter:
                    expected_resolve_args = {
                        "model": model_path,
                        "config": config_path,
                        "verifier": verifier_path,
                    }
                    if algorithm_kwargs:
                        expected_resolve_args.update(algorithm_kwargs)

                    mock_resolve.assert_called_once_with(
                        algorithm, **expected_resolve_args
                    )

                    # Verify Eagle converter was instantiated correctly
                    expected_converter_args = {
                        "model": model_path,
                        "config": config_path,
                        "verifier": verifier_path,
                    }
                    if algorithm_kwargs:
                        expected_converter_args.update(algorithm_kwargs)

                    mock_eagle_converter_class.assert_called_once_with(
                        **expected_converter_args
                    )

                    # Verify Eagle converter was called correctly
                    mock_eagle_converter_instance.assert_called_once_with(
                        output_path=output_path,
                        validate_device=validate_device,
                    )

                # Verify checkpoint resolution calls
                mock_check_model.assert_called_once()
                mock_check_config.assert_called_once()
