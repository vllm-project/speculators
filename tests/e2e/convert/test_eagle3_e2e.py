import json
from pathlib import Path
from typing import Optional

import pytest
import torch
from loguru import logger
from safetensors import safe_open

from speculators.convert.eagle.eagle3_converter import Eagle3Converter
from speculators.models.eagle3 import Eagle3Speculator


class TestEagle3ConversionE2E:
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
    def base_model(self):
        return "meta-llama/Llama-3.1-8B"

    @pytest.fixture
    def temp_dir(self, tmp_path):
        return tmp_path / "eagle3_e2e_test"

    def verify_config(self, config_path: Path, expected_type: str = "eagle3"):
        assert config_path.exists(), f"Config file not found: {config_path}"
        with config_path.open() as f:
            config_dict = json.load(f)

        assert config_dict.get("speculators_model_type") == expected_type
        assert "transformer_layer_config" in config_dict
        assert "speculators_config" in config_dict
        assert config_dict["speculators_config"]["algorithm"] == "eagle3"
        assert (
            config_dict["speculators_config"]["verifier"]["name_or_path"]
            == "meta-llama/Llama-3.1-8B"
        )

    def verify_checkpoint_structure(self, checkpoint_dir: Path):
        assert checkpoint_dir.exists(), f"Checkpoint dir not found: {checkpoint_dir}"
        assert (checkpoint_dir / "config.json").exists(), "Missing config.json"

        single_safetensors = checkpoint_dir / "model.safetensors"
        sharded_index = checkpoint_dir / "model.safetensors.index.json"
        has_weights = single_safetensors.exists() or sharded_index.exists()
        assert has_weights, "Missing model weights in safetensors format"

        if sharded_index.exists():
            shards = list(checkpoint_dir.glob("model-*.safetensors"))
            assert shards, "Index file exists but no shards found"

    def execute_forward_pass(self, model: Eagle3Speculator) -> Optional[torch.Tensor]:
        device = next(model.parameters()).device
        if device.type == "meta":
            logger.info("Model on meta device, skipping forward pass")
            return None

        batch_size = 2
        seq_len = 5
        hidden_size = model.config.transformer_layer_config.hidden_size
        vocab_size = model.config.target_vocab_size

        input_ids = torch.randint(0, min(1000, vocab_size), (batch_size, seq_len)).to(
            device
        )
        hidden_states = torch.randn(batch_size, seq_len, 3 * hidden_size).to(device)

        with torch.no_grad():
            output = model(input_ids=input_ids, hidden_states=hidden_states)

        assert hasattr(output, "logits"), "Output missing logits attribute"
        assert output.logits.shape == (batch_size, seq_len, vocab_size)

        assert not torch.isnan(output.logits).any(), "Output contains NaN"
        assert not torch.isinf(output.logits).any(), "Output contains Inf"

        return output.logits

    @pytest.mark.parametrize(
        "checkpoint_info",
        [
            {
                "name": "Eagle3 Speculator",
                "input_path": "nm-testing/SpeculatorLlama3-1-8B-Eagle3",
                "expected_algorithm": "eagle3",
            },
        ],
    )
    def test_eagle3_checkpoint_conversion_e2e(
        self, checkpoint_info, converter, base_model, temp_dir, temp_cache_dir
    ):
        name = checkpoint_info["name"]
        input_path = checkpoint_info["input_path"]

        converted_dir = temp_dir / f"{name.lower().replace(' ', '_')}_converted"
        resaved_dir = temp_dir / f"{name.lower().replace(' ', '_')}_resaved"

        logger.info(f"Testing: {name}")

        # Step 1: Convert checkpoint
        logger.info("Converting Eagle3 checkpoint...")
        converter.convert(
            input_path=input_path,
            output_path=converted_dir,
            base_model=base_model,
            validate=True,
            cache_dir=temp_cache_dir,
        )

        # Verify converted checkpoint
        self.verify_checkpoint_structure(converted_dir)
        self.verify_config(converted_dir / "config.json", expected_type="eagle3")
        logger.success("Conversion successful")

        # Step 2: Load model
        logger.info("Loading converted model...")
        model = Eagle3Speculator.from_pretrained(converted_dir)
        assert isinstance(model, Eagle3Speculator), "Wrong model type loaded"
        assert model.config.speculators_model_type == "eagle3"
        logger.success("Model loaded successfully")

        # Step 3: Forward pass
        logger.info("Executing forward pass...")
        logits = self.execute_forward_pass(model)
        if logits is not None:
            logger.success(f"Forward pass successful, output shape: {logits.shape}")

        # Step 4: Save model
        logger.info("Saving model...")
        model.save_pretrained(resaved_dir)
        logger.success(f"Model saved to: {resaved_dir}")

        # Step 5: Verify resaved model
        self.verify_checkpoint_structure(resaved_dir)
        self.verify_config(resaved_dir / "config.json", expected_type="eagle3")

        logger.info("Loading resaved model...")
        model2 = Eagle3Speculator.from_pretrained(resaved_dir)
        assert isinstance(model2, Eagle3Speculator), "Wrong model type loaded"
        assert model2.config.speculators_model_type == "eagle3"
        self.execute_forward_pass(model2)  # type: ignore[arg-type]

        logger.success(f"{name} - All tests passed!")


if __name__ == "__main__":
    import pytest

    pytest.main([__file__, "-v", "-s"])
