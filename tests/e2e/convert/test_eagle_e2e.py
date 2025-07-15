import json
from pathlib import Path
from typing import Optional, Type

import pytest
import torch
from loguru import logger

from speculators.convert.eagle.eagle_converter import EagleConverter
from speculators.convert.eagle.eagle3_converter import Eagle3Converter
from speculators.model import SpeculatorModel


class TestEagleConversionE2E:
    """End-to-end tests for Eagle and Eagle3 checkpoint conversion."""

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
    def base_model(self):
        return "meta-llama/Llama-3.1-8B-Instruct"

    @pytest.fixture
    def temp_dir(self, tmp_path):
        return tmp_path / "e2e_test"

    def verify_config(self, config_path: Path, expected_type: str, expected_features: dict):
        assert config_path.exists(), f"Config file not found: {config_path}"
        with config_path.open() as f:
            config_dict = json.load(f)

        assert config_dict.get("speculators_model_type") == expected_type
        for feature, expected_value in expected_features.items():
            assert config_dict.get(feature) == expected_value, (
                f"Expected {feature}={expected_value}, got {config_dict.get(feature)}"
            )

        assert "transformer_layer_config" in config_dict
        assert "speculators_config" in config_dict
        assert config_dict["speculators_config"]["algorithm"] == expected_type
        assert config_dict["speculators_config"]["verifier"]["name_or_path"] == \
            "meta-llama/Llama-3.1-8B-Instruct"

    def verify_checkpoint_structure(self, checkpoint_dir: Path):
        assert checkpoint_dir.exists(), f"Checkpoint directory not found: {checkpoint_dir}"
        assert (checkpoint_dir / "config.json").exists(), "Missing config.json"
        has_weights = (
            (checkpoint_dir / "model.safetensors").exists()
            or (checkpoint_dir / "model.safetensors.index.json").exists()
        )
        assert has_weights, "Missing model weights in safetensors format"
        if (checkpoint_dir / "model.safetensors.index.json").exists():
            shards = list(checkpoint_dir.glob("model-*.safetensors"))
            assert len(shards) > 0, "Index exists but no shards found"

    def execute_forward_pass(self, model: SpeculatorModel) -> Optional[torch.Tensor]:
        device = next(model.parameters()).device
        if device.type == "meta":
            logger.info("Model is on meta device, skipping forward pass test")
            return None

        B, L = 2, 10
        H = model.config.transformer_layer_config.hidden_size
        V = model.config.transformer_layer_config.vocab_size
        input_ids = torch.randint(0, min(1000, V), (B, L)).to(device)
        hidden_states = torch.randn(B, L, H).to(device)

        with torch.no_grad():
            output = model(input_ids=input_ids, hidden_states=hidden_states)
        assert hasattr(output, "logits")
        assert output.logits.shape == (B, L, V)
        assert not torch.isnan(output.logits).any()
        assert not torch.isinf(output.logits).any()
        return output.logits

    @pytest.mark.parametrize("converter_cls, model_type, input_path, convert_kwargs", [
        (
            EagleConverter,
            "eagle",
            "yuhuili/EAGLE-LLaMA3.1-Instruct-8B",
            {"layernorms": False, "fusion_bias": False},
        ),
        (
            EagleConverter,
            "eagle",
            "nm-testing/Eagle_Speculator_Llama_3_1_8B_TTT",
            {"layernorms": True, "fusion_bias": False},
        ),
        (
            Eagle3Converter,
            "eagle3",
            "nm-testing/SpeculatorLlama3-1-8B-Eagle3",
            {"norm_before_residual": True},
        ),
    ])
    def test_checkpoint_conversion_e2e(
        self,
        converter_cls: Type,
        model_type: str,
        input_path: str,
        convert_kwargs: dict,
        base_model,
        temp_dir,
        temp_cache_dir,
    ):
        converted_dir = temp_dir / f"{model_type}_converted"
        resaved_dir = temp_dir / f"{model_type}_resaved"

        converter = converter_cls(
            model=input_path,
            output_path=converted_dir,
            verifier=base_model,
            cache_dir=temp_cache_dir,
        )

        # Pass feature flags directly to convert()
        converter.convert(
            validate=True,
            **convert_kwargs,
        )

        self.verify_checkpoint_structure(converted_dir)
        self.verify_config(converted_dir / "config.json", model_type, convert_kwargs)

        model = SpeculatorModel.from_pretrained(converted_dir)
        assert isinstance(model, SpeculatorModel)
        self.execute_forward_pass(model)

        model.save_pretrained(resaved_dir)
        self.verify_checkpoint_structure(resaved_dir)
        self.verify_config(resaved_dir / "config.json", model_type, convert_kwargs)

        model2 = SpeculatorModel.from_pretrained(resaved_dir)
        self.execute_forward_pass(model2)

        logger.success(f"{model_type} conversion flow passed.")
