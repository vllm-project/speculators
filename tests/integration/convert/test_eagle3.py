import json
from pathlib import Path

import pytest
import torch
from loguru import logger

from speculators.convert.eagle.eagle3_converter import Eagle3Converter
from speculators.models.eagle3 import Eagle3Speculator


class TestEagle3Conversion:
    """End-to-end tests for Eagle3 checkpoint conversion."""

    def setup_method(self):
        import gc

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    @pytest.fixture
    def temp_cache_dir(self, tmp_path, monkeypatch):
        cache_dir = tmp_path / "hf_cache"
        cache_dir.mkdir(exist_ok=True)
        monkeypatch.setenv("HF_HOME", str(cache_dir))
        monkeypatch.setenv("TRANSFORMERS_CACHE", str(cache_dir))
        monkeypatch.setenv("HUGGINGFACE_HUB_CACHE", str(cache_dir))
        return cache_dir

    @pytest.fixture
    def converter(self):
        return Eagle3Converter()

    @pytest.fixture
    def temp_dir(self, tmp_path):
        return tmp_path / "eagle3_e2e_test"

    def verify_config(
        self, config_path: Path, expected_base_model: str, expected_type: str = "eagle3"
    ):
        """
        Validates that the converted model's config.json has all required fields.
        This ensures the conversion produced a valid speculators-format model.
        """
        assert config_path.exists(), f"Config file not found: {config_path}"
        with config_path.open() as f:
            config_dict = json.load(f)

        assert config_dict.get("speculators_model_type") == expected_type
        assert "transformer_layer_config" in config_dict
        assert "speculators_config" in config_dict
        assert config_dict["speculators_config"]["algorithm"] == "eagle3"
        assert (
            config_dict["speculators_config"]["verifier"]["name_or_path"]
            == expected_base_model
        )

        # Verify norm_before_residual is present in config
        assert "norm_before_residual" in config_dict, (
            "norm_before_residual should be in config"
        )

    def verify_checkpoint_structure(self, checkpoint_dir: Path):
        """
        Validates that the converted model's checkpoint directory
        has all required files. This ensures the conversion produced
        a valid speculators-format model.
        """
        assert checkpoint_dir.exists(), f"Checkpoint dir not found: {checkpoint_dir}"
        assert (checkpoint_dir / "config.json").exists(), "Missing config.json"

        # Check for model weights file
        weight_files = list(checkpoint_dir.glob("*.safetensors")) + list(
            checkpoint_dir.glob("pytorch_model.bin")
        )
        assert len(weight_files) > 0, "No model weight files found"

    def execute_forward_pass(self, model: Eagle3Speculator) -> bool:
        """
        Actually runs the model to verify it works correctly after conversion.
        This catches issues like shape mismatches or broken layers.
        """
        try:
            batch_size = 1
            seq_len = 5

            # Create dummy input_ids from draft vocabulary
            input_ids = torch.tensor([[1, 2, 3, 4, 5]], dtype=torch.long)

            # Eagle3 requires hidden_states from 3 verifier layers
            target_hidden_size = model.target_hidden_size
            hidden_states = torch.randn(batch_size, seq_len, 3 * target_hidden_size)
            logger.info(
                f"Forward pass inputs - input_ids: {input_ids.shape}, "
                f"hidden_states: {hidden_states.shape}"
            )

            # Execute forward pass
            with torch.no_grad():
                output = model(input_ids=input_ids, hidden_states=hidden_states)  # type: ignore[operator]

            # Basic checks
            assert hasattr(output, "logits"), "Output missing logits"
            assert output.logits.shape[0] == batch_size, (
                f"Wrong batch size: {output.logits.shape[0]}"
            )
            assert output.logits.shape[1] == seq_len, (
                f"Wrong sequence length: {output.logits.shape[1]}"
            )

            # Verify output uses target vocabulary size (mapped from draft)
            expected_vocab_size = model.config.target_vocab_size
            assert output.logits.shape[2] == expected_vocab_size, (
                f"Wrong vocab size: expected {expected_vocab_size}, "
                f"got {output.logits.shape[2]}"
            )

            logger.info(f"Forward pass successful, logits shape: {output.logits.shape}")
            return True

        except (RuntimeError, ValueError, AssertionError) as e:
            logger.error(f"Forward pass failed: {e}")
            return False

    # TODO: @dsikka - add llama3 example # noqa: FIX002
    @pytest.mark.smoke
    @pytest.mark.parametrize(
        "checkpoint_info",
        [
            {
                "name": "Research Eagle3 Qwen3 8B with Norm Before Residual",
                "input_path": "nm-testing/Speculator-Qwen3-8B-Eagle3",
                "base_model": "Qwen/Qwen3-8B",
                "expected_algorithm": "eagle3",
                "norm_before_residual": True,
            },
        ],
    )
    def test_eagle3_checkpoint_conversion(
        self, checkpoint_info, converter, temp_dir, temp_cache_dir
    ):
        """
        Test end-to-end conversion workflow for Eagle3 checkpoints.

        This test:
        1. Converts the checkpoint to speculators format and validates the conversion
        2. Loads the converted model
        3. Executes a forward pass
        4. Saves the model again
        5. Validates the saved checkpoint
        """
        name = checkpoint_info["name"]
        input_path = checkpoint_info["input_path"]
        base_model = checkpoint_info["base_model"]
        norm_before_residual = checkpoint_info["norm_before_residual"]

        converted_dir = temp_dir / f"{name.lower().replace(' ', '_')}_converted"
        resaved_dir = temp_dir / f"{name.lower().replace(' ', '_')}_resaved"

        logger.info(f"Testing: {name}")
        logger.info(f"Input: {input_path}")
        logger.info(f"Base model: {base_model}")
        logger.info(f"Norm before residual: {norm_before_residual}")

        # Step 1: Convert checkpoint
        logger.info("Converting Eagle3 checkpoint...")
        converter.convert(
            input_path=input_path,
            output_path=converted_dir,
            base_model=base_model,
            validate=True,
            cache_dir=temp_cache_dir,
            norm_before_residual=norm_before_residual,
        )

        # Verify converted checkpoint
        self.verify_checkpoint_structure(converted_dir)
        self.verify_config(
            converted_dir / "config.json", base_model, expected_type="eagle3"
        )
        logger.success("Conversion successful")

        # Step 2: Load model
        logger.info("Loading converted model...")
        model = Eagle3Speculator.from_pretrained(
            converted_dir, verifier=base_model, verifier_attachment_mode="detached"
        )
        assert isinstance(model, Eagle3Speculator), "Wrong model type loaded"
        assert model.config.speculators_model_type == "eagle3"

        # Verify norm_before_residual setting
        assert model.config.norm_before_residual == norm_before_residual, (
            f"Expected norm_before_residual={norm_before_residual}, "
            f"got {model.config.norm_before_residual}"
        )
        logger.success("Model loaded successfully")

        # Step 3: Forward pass
        logger.info("Executing forward pass...")
        forward_pass_success = self.execute_forward_pass(model)
        assert forward_pass_success, (
            "Forward pass failed - this indicates the converted model has issues"
        )
        logger.success("Forward pass completed successfully")

        # Step 4: Save model
        logger.info("Saving model...")
        model.save_pretrained(resaved_dir)  # type: ignore[attr-defined]
        logger.success(f"Model saved to: {resaved_dir}")

        # Step 5: Verify resaved model
        self.verify_checkpoint_structure(resaved_dir)
        self.verify_config(
            resaved_dir / "config.json", base_model, expected_type="eagle3"
        )
        logger.success("Full Integration test completed successfully")
