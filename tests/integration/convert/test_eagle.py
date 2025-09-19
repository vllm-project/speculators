"""
End-to-end tests for Eagle checkpoint conversion.

Verifies the complete conversion workflow for Eagle and HASS checkpoints:
1. Converting checkpoints to speculators format
2. Loading converted models using from_pretrained
3. Executing forward passes
4. Saving models using save_pretrained
5. Validating saved directories and configs
"""

import json
from pathlib import Path
from typing import Optional

import pytest
import torch
from loguru import logger

from speculators.convert.eagle import EagleConverter
from speculators.models.eagle import EagleSpeculator, EagleSpeculatorConfig


class TestEagleConversion:
    """End-to-end tests for Eagle checkpoint conversion."""

    def setup_method(self):
        """Clear any cached models or state before each test."""
        # Clear transformers model cache to ensure clean state
        import gc

        import torch

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    @pytest.fixture
    def temp_cache_dir(self, tmp_path, monkeypatch):
        """Create a temporary cache directory for model downloads."""
        cache_dir = tmp_path / "hf_cache"
        cache_dir.mkdir(exist_ok=True)

        # Also set environment variables to ensure HF uses our cache
        monkeypatch.setenv("HF_HOME", str(cache_dir))
        monkeypatch.setenv("TRANSFORMERS_CACHE", str(cache_dir))
        monkeypatch.setenv("HUGGINGFACE_HUB_CACHE", str(cache_dir))

        return cache_dir

    @pytest.fixture
    def converter(self):
        """Create an Eagle converter instance."""
        return EagleConverter()

    @pytest.fixture
    def base_model(self):
        """Base model name for conversions."""
        return "meta-llama/Llama-3.1-8B-Instruct"

    @pytest.fixture
    def temp_dir(self, tmp_path):
        """Create a temporary directory for test outputs."""
        return tmp_path / "e2e_test"

    def verify_config(
        self, config_path: Path, expected_type: str, expected_features: dict
    ):
        """
        Verify the saved config file contains expected values.

        :param config_path: Path to config.json
        :param expected_type: Expected speculators_model_type
        :param expected_features: Expected feature flags (layernorms, fusion_bias)
        """
        assert config_path.exists(), f"Config file not found: {config_path}"

        with config_path.open() as f:
            config_dict = json.load(f)

        # Verify model type
        assert config_dict.get("speculators_model_type") == expected_type

        # Verify features
        for feature, expected_value in expected_features.items():
            assert config_dict.get(feature) == expected_value, (
                f"Expected {feature}={expected_value}, got {config_dict.get(feature)}"
            )

        # Verify essential fields
        assert "transformer_layer_config" in config_dict
        assert "speculators_config" in config_dict
        assert config_dict["speculators_config"]["algorithm"] == "eagle"
        assert (
            config_dict["speculators_config"]["verifier"]["name_or_path"]
            == "meta-llama/Llama-3.1-8B-Instruct"
        )

    def verify_checkpoint_structure(self, checkpoint_dir: Path):
        """
        Verify checkpoint directory structure after conversion.

        After conversion, checkpoints are always stored in safetensors format.

        :param checkpoint_dir: Path to checkpoint directory
        """
        assert checkpoint_dir.exists(), (
            f"Checkpoint directory not found: {checkpoint_dir}"
        )
        assert (checkpoint_dir / "config.json").exists(), "Missing config.json"

        # Check for weights in safetensors format only
        single_safetensors = checkpoint_dir / "model.safetensors"
        sharded_safetensors_index = checkpoint_dir / "model.safetensors.index.json"

        has_weights = single_safetensors.exists() or sharded_safetensors_index.exists()

        assert has_weights, "Missing model weights in safetensors format"

        # For sharded models, check that at least one shard exists
        if sharded_safetensors_index.exists():
            shard_files = list(checkpoint_dir.glob("model-*.safetensors"))
            assert len(shard_files) > 0, "Index file exists but no shard files found"

    def execute_forward_pass(self, model: EagleSpeculator) -> Optional[torch.Tensor]:
        """
        Execute a forward pass with the model.

        :param model: EagleSpeculator model instance
        :return: Output logits or None if model is on meta device
        """

        # Check if model is on meta device
        device = next(model.parameters()).device  # type: ignore[attr-defined]
        if device.type == "meta":
            logger.info("Model is on meta device, skipping forward pass test")
            return None

        batch_size = 2
        seq_length = 10
        hidden_size = model.config.transformer_layer_config.hidden_size
        vocab_size = model.config.transformer_layer_config.vocab_size

        # Create dummy inputs on the same device as the model
        input_ids = torch.randint(
            0, min(1000, vocab_size), (batch_size, seq_length)
        ).to(device)
        hidden_states = torch.randn(batch_size, seq_length, hidden_size).to(device)

        # Execute forward pass
        with torch.no_grad():
            output = model(input_ids=input_ids, hidden_states=hidden_states)  # type: ignore[operator]

        # Verify output shape
        assert hasattr(output, "logits"), "Output missing logits attribute"
        assert output.logits.shape == (batch_size, seq_length, vocab_size), (
            f"Unexpected output shape: {output.logits.shape}"
        )

        # Check for NaN/Inf
        assert not torch.isnan(output.logits).any(), "Output contains NaN values"
        assert not torch.isinf(output.logits).any(), "Output contains Inf values"

        return output.logits

    @pytest.mark.smoke
    @pytest.mark.skip("Missing Llama HF Token")
    @pytest.mark.parametrize(
        "checkpoint_info",
        [
            {
                "name": "Eagle Standard",
                "input_path": "yuhuili/EAGLE-LLaMA3.1-Instruct-8B",
                "expected_features": {"layernorms": False, "fusion_bias": False},
            },
            {
                "name": "HASS with Layernorms",
                "input_path": "nm-testing/Eagle_Speculator_Llama_3_1_8B_TTT",
                "expected_features": {"layernorms": True, "fusion_bias": False},
            },
        ],
    )
    def test_eagle_checkpoint_conversion(
        self, checkpoint_info, converter, base_model, temp_dir, temp_cache_dir
    ):
        """
        Test end-to-end conversion workflow for Eagle checkpoints.

        This test:
        1. Converts the checkpoint to speculators format
        2. Loads the converted model
        3. Executes a forward pass
        4. Saves the model again
        5. Validates the saved checkpoint
        """
        name = checkpoint_info["name"]
        input_path = checkpoint_info["input_path"]
        expected_features = checkpoint_info["expected_features"]

        # Create test directories
        converted_dir = temp_dir / f"{name.lower().replace(' ', '_')}_converted"
        resaved_dir = temp_dir / f"{name.lower().replace(' ', '_')}_resaved"

        logger.info(f"Testing: {name}")
        logger.info(f"Input: {input_path}")
        logger.info(f"Expected features: {expected_features}")

        # Step 1: Convert checkpoint
        logger.info("Converting checkpoint...")
        converter.convert(
            input_path=input_path,
            output_path=converted_dir,
            base_model=base_model,
            validate=True,  # This already tests loading and forward pass
            cache_dir=temp_cache_dir,
        )

        # Verify converted checkpoint structure
        assert converted_dir.exists(), f"Converted directory not found: {converted_dir}"
        assert (converted_dir / "config.json").exists(), "Missing config.json"
        assert (converted_dir / "model.safetensors").exists(), (
            "Missing model.safetensors"
        )

        # Verify config
        self.verify_config(
            converted_dir / "config.json",
            expected_type="eagle",
            expected_features=expected_features,
        )
        logger.success("Conversion successful")

        # Step 2: Load converted model
        logger.info("Loading converted model...")
        model = EagleSpeculator.from_pretrained(converted_dir)
        assert isinstance(model, EagleSpeculator), "Wrong model type loaded"
        assert isinstance(model.config, EagleSpeculatorConfig), "Wrong config type"

        # Verify config attributes
        assert model.config.layernorms == expected_features["layernorms"]
        assert model.config.fusion_bias == expected_features["fusion_bias"]
        logger.success("Model loaded successfully")

        # Step 3: Execute forward pass
        logger.info("Executing forward pass...")
        logits = self.execute_forward_pass(model)
        if logits is not None:
            logger.success(f"Forward pass successful, output shape: {logits.shape}")
        else:
            logger.info("Forward pass skipped (model on meta device)")

        # Step 4: Save model using save_pretrained
        logger.info("Saving model using save_pretrained...")
        model.save_pretrained(resaved_dir)  # type: ignore[attr-defined]
        logger.success(f"Model saved to: {resaved_dir}")

        # Step 5: Validate saved checkpoint
        logger.info("Validating saved checkpoint...")
        self.verify_checkpoint_structure(resaved_dir)
        self.verify_config(
            resaved_dir / "config.json",
            expected_type="eagle",
            expected_features=expected_features,
        )

        # Load the resaved model to ensure it works
        logger.info("Loading resaved model...")
        model2 = EagleSpeculator.from_pretrained(resaved_dir)
        assert isinstance(model2, EagleSpeculator)
        assert isinstance(model2.config, EagleSpeculatorConfig)

        # Verify configs match
        assert model2.config.layernorms == model.config.layernorms
        assert model2.config.fusion_bias == model.config.fusion_bias
        assert (
            model2.config.transformer_layer_config.vocab_size
            == model.config.transformer_layer_config.vocab_size
        )

        # Execute forward pass on resaved model
        self.execute_forward_pass(model2)
        logger.success("Resaved model forward pass successful")

        logger.success(f"{name} - All tests passed!")

    @pytest.mark.smoke
    @pytest.mark.skip("Missing Llama HF Token")
    def test_conversion_with_explicit_features(
        self, converter, base_model, temp_dir, temp_cache_dir
    ):
        """
        Test conversion with explicitly set features overriding auto-detection.
        """
        # Use the standard Eagle checkpoint but force fusion_bias=True
        input_path = "yuhuili/EAGLE-LLaMA3.1-Instruct-8B"
        output_dir = temp_dir / "eagle_forced_fusion_bias"

        logger.info("Testing explicit feature override")

        # Convert with forced fusion_bias
        converter.convert(
            input_path=input_path,
            output_path=output_dir,
            base_model=base_model,
            fusion_bias=True,  # Force this even though checkpoint doesn't have fc.bias
            layernorms=False,
            validate=True,
            cache_dir=temp_cache_dir,
        )

        # Load and verify
        model = EagleSpeculator.from_pretrained(output_dir)
        assert model.config.fusion_bias is True, "fusion_bias should be True"
        assert model.config.layernorms is False, "layernorms should be False"

        # Check that fc layer has bias
        assert model.fusion_fc.bias is not None, (  # type: ignore[union-attr,attr-defined]
            "fusion_fc layer should have bias parameter"
        )

        logger.success("Explicit feature override successful")

    @pytest.mark.smoke
    @pytest.mark.parametrize("validate", [True, False])
    def test_validation_flag(
        self, converter, base_model, temp_dir, temp_cache_dir, validate
    ):
        """
        Test that the validate flag works correctly.
        """
        input_path = "yuhuili/EAGLE-LLaMA3.1-Instruct-8B"
        output_dir = temp_dir / f"eagle_validate_{validate}"

        logger.info(f"Testing validation flag: validate={validate}")

        # Convert with specified validation setting
        converter.convert(
            input_path=input_path,
            output_path=output_dir,
            base_model=base_model,
            validate=validate,
            cache_dir=temp_cache_dir,
        )

        # Conversion should succeed regardless of validation
        assert output_dir.exists()
        assert (output_dir / "config.json").exists()
        assert (output_dir / "model.safetensors").exists()

        # Try loading the model - should work even if validation was skipped
        model = EagleSpeculator.from_pretrained(output_dir)
        self.execute_forward_pass(model)  # type: ignore[arg-type]

        logger.success(f"Conversion with validate={validate} successful")


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "-s"])
