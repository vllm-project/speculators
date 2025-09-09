"""
Unit tests for speculators.convert.entrypoints module.

Tests the convert_model function which provides the primary conversion interface
for transforming non-Speculators model checkpoints to Speculators format.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import Mock, patch

import pytest
import torch
from torch import nn

from speculators.convert.entrypoints import convert_model
from speculators.model import SpeculatorModel


class TestConvertModel:
    """Test suite for convert_model function."""

    @pytest.fixture
    def mock_converter_class(self):
        """Fixture providing a mock SpeculatorConverter class."""
        converter_class = Mock()
        converter_instance = Mock()
        mock_speculator_model = Mock(spec=SpeculatorModel)
        converter_instance.return_value = mock_speculator_model
        converter_class.return_value = converter_instance
        return converter_class

    @pytest.mark.smoke
    @pytest.mark.parametrize(
        ("model_input", "config_input", "extra_params", "expected_calls"),
        [
            # String model path with basic parameters
            (
                "/path/to/model",
                None,
                {"algorithm": "eagle", "output_path": "/output"},
                {"config_is_model": True},
            ),
            # HuggingFace model ID with explicit config
            (
                "huggingface/model-id",
                "/config/path",
                {"algorithm": "eagle2", "verifier": "/verifier"},
                {"config_is_model": False},
            ),
            # Model with all parameters
            (
                "/model/path",
                "/config/path",
                {
                    "algorithm": "hass",
                    "algorithm_kwargs": {"param1": "value1"},
                    "output_path": "/output",
                    "verifier": "/verifier",
                    "validate_device": "cuda:0",
                    "cache_dir": "/cache",
                    "force_download": True,
                    "token": "test_token",
                },
                {"config_is_model": False},
            ),
            # Auto algorithm detection
            (
                "/model/path",
                None,
                {"algorithm": "auto"},
                {"config_is_model": True},
            ),
            # PathLib paths and torch device
            (
                Path("/model/path"),
                None,
                {"algorithm": "eagle", "validate_device": torch.device("cuda:0")},
                {"config_is_model": True},
            ),
            # None algorithm_kwargs
            (
                "/model/path",
                None,
                {"algorithm": "eagle", "algorithm_kwargs": None},
                {"config_is_model": True},
            ),
        ],
        ids=[
            "basic_string_model",
            "hf_model_with_config",
            "all_parameters",
            "auto_algorithm",
            "pathlib_torch_device",
            "none_algorithm_kwargs",
        ],
    )
    @patch("speculators.convert.entrypoints.check_download_model_checkpoint")
    @patch("speculators.convert.entrypoints.check_download_model_config")
    @patch("speculators.convert.entrypoints.SpeculatorConverter.resolve_converter")
    @patch("speculators.convert.entrypoints.logger")
    def test_invocation_variations(
        self,
        mock_logger,
        mock_resolve_converter,
        mock_check_config,
        mock_check_checkpoint,
        mock_converter_class,
        model_input,
        config_input,
        extra_params,
        expected_calls,
    ):
        """Test convert_model with various parameter combinations."""
        # Setup
        resolved_model = (
            str(model_input) if isinstance(model_input, Path) else model_input
        )
        resolved_config = config_input if config_input else resolved_model

        mock_check_checkpoint.return_value = resolved_model
        mock_check_config.return_value = resolved_config
        mock_resolve_converter.return_value = mock_converter_class
        mock_speculator_model = Mock(spec=SpeculatorModel)
        mock_converter_class.return_value.return_value = mock_speculator_model

        # Execute
        result = convert_model(model=model_input, config=config_input, **extra_params)

        # Verify core functionality
        assert result == mock_speculator_model
        mock_check_checkpoint.assert_called_once()

        if expected_calls["config_is_model"]:
            mock_check_config.assert_called_once_with(
                resolved_model,
                cache_dir=extra_params.get("cache_dir"),
                force_download=extra_params.get("force_download", False),
                local_files_only=extra_params.get("local_files_only", False),
                token=extra_params.get("token"),
                revision=extra_params.get("revision"),
            )
        else:
            mock_check_config.assert_called_once()

        mock_resolve_converter.assert_called_once()
        mock_converter_class.assert_called_once()
        mock_converter_class.return_value.assert_called_once()
        assert mock_logger.info.call_count >= 3

    @pytest.mark.smoke
    @patch("speculators.convert.entrypoints.check_download_model_checkpoint")
    @patch("speculators.convert.entrypoints.check_download_model_config")
    @patch("speculators.convert.entrypoints.SpeculatorConverter.resolve_converter")
    def test_invocation_pretrained_model_instance(
        self,
        mock_resolve_converter,
        mock_check_config,
        mock_check_checkpoint,
        mock_converter_class,
    ):
        """Test convert_model with PreTrainedModel instance and config inference."""
        # Setup
        mock_model_instance = Mock()

        with patch("speculators.convert.entrypoints.isinstance") as mock_isinstance:
            mock_isinstance.side_effect = lambda obj, cls: cls is not nn.Module

            mock_check_checkpoint.return_value = mock_model_instance
            mock_check_config.return_value = mock_model_instance
            mock_resolve_converter.return_value = mock_converter_class
            mock_speculator_model = Mock(spec=SpeculatorModel)
            mock_converter_class.return_value.return_value = mock_speculator_model

            # Execute
            result = convert_model(model=mock_model_instance, algorithm="eagle")

            # Verify
            assert result == mock_speculator_model
            mock_check_checkpoint.assert_called_once_with(
                mock_model_instance,
                cache_dir=None,
                force_download=False,
                local_files_only=False,
                token=None,
                revision=None,
            )
            mock_check_config.assert_called_once_with(
                mock_model_instance,
                cache_dir=None,
                force_download=False,
                local_files_only=False,
                token=None,
                revision=None,
            )

    @pytest.mark.sanity
    @pytest.mark.parametrize(
        (
            "error_scenario",
            "model_input",
            "config_input",
            "algorithm",
            "expected_error",
        ),
        [
            # nn.Module without config
            (
                "nn_module_no_config",
                Mock(spec=nn.Module),
                None,
                "eagle",
                "A model config must be provided",
            ),
            # Invalid algorithm
            (
                "invalid_algorithm",
                "/model/path",
                None,
                "invalid_algorithm",
                "Algorithm .* is not registered",
            ),
            # Empty algorithm
            (
                "empty_algorithm",
                "/model/path",
                None,
                "",
                "Algorithm .* is not registered",
            ),
        ],
        ids=["nn_module_no_config", "invalid_algorithm", "empty_algorithm"],
    )
    @patch("speculators.convert.entrypoints.check_download_model_checkpoint")
    @patch("speculators.convert.entrypoints.check_download_model_config")
    @patch("speculators.convert.entrypoints.SpeculatorConverter.resolve_converter")
    def test_invalid_invocations(
        self,
        mock_resolve_converter,
        mock_check_config,
        mock_check_checkpoint,
        mock_converter_class,
        error_scenario,
        model_input,
        config_input,
        algorithm,
        expected_error,
    ):
        """Test convert_model error conditions."""
        # Setup based on scenario
        if error_scenario == "nn_module_no_config":
            mock_check_checkpoint.return_value = model_input
        else:
            mock_check_checkpoint.return_value = model_input
            mock_check_config.return_value = model_input
            mock_resolve_converter.side_effect = ValueError(
                f"Algorithm '{algorithm}' is not registered"
            )

        # Execute & Verify
        with pytest.raises(ValueError, match=expected_error):
            convert_model(model=model_input, config=config_input, algorithm=algorithm)

    @pytest.mark.sanity
    @pytest.mark.parametrize(
        ("model_input", "config_input"),
        [
            (None, "/config/path"),
            ("", "/config/path"),
            ("/model/path", None),
            ("/model/path", ""),
        ],
        ids=["none_model", "empty_model", "none_config", "empty_config"],
    )
    @patch("speculators.convert.entrypoints.check_download_model_checkpoint")
    @patch("speculators.convert.entrypoints.check_download_model_config")
    @patch("speculators.convert.entrypoints.SpeculatorConverter.resolve_converter")
    def test_invalid_empty_paths(
        self,
        mock_resolve_converter,
        mock_check_config,
        mock_check_checkpoint,
        mock_converter_class,
        model_input,
        config_input,
    ):
        """Test convert_model with empty/None model or config paths."""
        # Setup
        mock_check_checkpoint.return_value = model_input
        mock_check_config.return_value = config_input
        mock_resolve_converter.return_value = mock_converter_class
        mock_converter_class.side_effect = ValueError(
            "Model and config paths must be provided"
        )

        # Execute & Verify
        with pytest.raises(ValueError, match="Model and config paths must be provided"):
            convert_model(model=model_input, config=config_input, algorithm="eagle")
