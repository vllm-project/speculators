"""Integration tests for FastMTPConverter using a real Hub checkpoint.

Downloads ``inference-optimization/test_qwen3_next_mtp``, a small public
checkpoint that stores only the MTP layer (plus embed_tokens and lm_head) in
the Qwen3-Next ``mtp.*`` key format.  The full base model is never downloaded;
only a minimal local config is needed to satisfy the converter's VerifierConfig.
"""

import gc
import json
from pathlib import Path

import pytest
import torch
from loguru import logger
from safetensors import safe_open

from speculators.convert.fast_mtp import FastMTPConverter
from speculators.models.fast_mtp import FastMTPConfig, FastMTPSpeculator

# ---------------------------------------------------------------------------
# Test checkpoint — small public repo with only MTP head weights (~300 MB)
# ---------------------------------------------------------------------------
HUB_CHECKPOINT = "inference-optimization/test_qwen3_next_mtp"

# Minimal base-model config: the converter only reads ``architectures`` from
# this; all architecture dimensions come from the source checkpoint's config.
_BASE_MODEL_CONFIG = {
    "model_type": "qwen2",
    "architectures": ["Qwen2ForCausalLM"],
    "hidden_size": 64,
    "num_hidden_layers": 2,
    "num_attention_heads": 2,
    "num_key_value_heads": 2,
    "intermediate_size": 128,
    "vocab_size": 256,
    "max_position_embeddings": 64,
    "rms_norm_eps": 1e-6,
    "rope_theta": 10000.0,
}


# ---------------------------------------------------------------------------
# Module-scoped fixtures: download + convert once, reuse across tests
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def hf_cache_dir(tmp_path_factory):
    """Temporary HF cache directory shared across the module."""
    return tmp_path_factory.mktemp("hf_cache")


@pytest.fixture(scope="module")
def base_model_dir(tmp_path_factory):
    """Minimal local base-model config directory (no weight download needed)."""
    d = tmp_path_factory.mktemp("base_model")
    (d / "config.json").write_text(json.dumps(_BASE_MODEL_CONFIG))
    return str(d)


@pytest.fixture(scope="module")
def converted_dir(tmp_path_factory, base_model_dir, hf_cache_dir):
    """Convert the Hub checkpoint once and share the output across all tests."""
    gc.collect()
    output = tmp_path_factory.mktemp("converted")
    converter = FastMTPConverter()
    logger.info(f"Converting {HUB_CHECKPOINT} → {output}")
    converter.convert(
        input_path=HUB_CHECKPOINT,
        output_path=output,
        base_model=base_model_dir,
        num_speculative_steps=3,
        validate=True,
        cache_dir=str(hf_cache_dir),
    )
    logger.success("Conversion complete")
    return output


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.smoke
class TestConvertOutputFiles:
    """Verify that the converter writes the expected files."""

    def test_config_json_created(self, converted_dir: Path):
        assert (converted_dir / "config.json").exists()

    def test_weight_file_created(self, converted_dir: Path):
        weight_files = list(converted_dir.glob("*.safetensors"))
        assert len(weight_files) > 0, "No safetensors weight file found"

    def test_config_speculators_model_type(self, converted_dir: Path):
        cfg = json.loads((converted_dir / "config.json").read_text())
        assert cfg["speculators_model_type"] == "mtp"

    def test_config_has_transformer_layer_config(self, converted_dir: Path):
        cfg = json.loads((converted_dir / "config.json").read_text())
        assert "transformer_layer_config" in cfg

    def test_config_algorithm(self, converted_dir: Path):
        cfg = json.loads((converted_dir / "config.json").read_text())
        assert cfg["speculators_config"]["algorithm"] == "mtp"


@pytest.mark.smoke
class TestSelfContainedLoad:
    """The converted checkpoint must load without specifying a verifier path."""

    @pytest.fixture(scope="class")
    def model(self, converted_dir: Path) -> FastMTPSpeculator:
        return FastMTPSpeculator.from_pretrained(str(converted_dir))  # type: ignore[return-value]

    def test_embed_tokens_present(self, model: FastMTPSpeculator):
        assert model.embed_tokens is not None

    def test_lm_head_present(self, model: FastMTPSpeculator):
        assert model.lm_head is not None

    def test_mtp_layer_present(self, model: FastMTPSpeculator):
        assert len(model.mtp_layers) == 1
        assert model.mtp_layers[0] is not None  # type: ignore[index]

    def test_config_type(self, model: FastMTPSpeculator):
        assert isinstance(model.config, FastMTPConfig)

    def test_num_speculative_steps(self, model: FastMTPSpeculator):
        assert model.config.num_speculative_steps == 3


@pytest.mark.smoke
class TestCheckpointContents:
    """Verify that embed_tokens and lm_head weights are saved in the checkpoint."""

    def test_embed_tokens_weight_in_safetensors(self, converted_dir: Path):
        weight_file = next(converted_dir.glob("*.safetensors"))
        with safe_open(str(weight_file), framework="pt") as f:
            keys = list(f.keys())  # noqa: SIM118
        assert "embed_tokens.weight" in keys

    def test_lm_head_weight_in_safetensors(self, converted_dir: Path):
        weight_file = next(converted_dir.glob("*.safetensors"))
        with safe_open(str(weight_file), framework="pt") as f:
            keys = list(f.keys())  # noqa: SIM118
        assert "lm_head.weight" in keys

    def test_mtp_layer_weight_in_safetensors(self, converted_dir: Path):
        weight_file = next(converted_dir.glob("*.safetensors"))
        with safe_open(str(weight_file), framework="pt") as f:
            mtp_keys = [k for k in f.keys() if k.startswith("mtp_layers.0.")]  # noqa: SIM118
        assert len(mtp_keys) > 0, "No mtp_layers.0.* keys found in safetensors"


@pytest.mark.smoke
class TestForwardPass:
    """Forward pass with random inputs must produce finite logits."""

    @pytest.fixture(scope="class")
    def model(self, converted_dir: Path) -> FastMTPSpeculator:
        m = FastMTPSpeculator.from_pretrained(str(converted_dir))  # type: ignore[assignment]
        m.eval()
        return m  # type: ignore[return-value]

    def test_logits_produced(self, model: FastMTPSpeculator):
        """At least one step must produce logits for seq_len=8, num_steps=3."""
        batch_size, seq_len = 1, 8
        vocab_size = model.config.vocab_size
        hidden_size = model.config.hidden_size

        input_ids = torch.randint(0, min(vocab_size, 100), (batch_size, seq_len))
        hidden_states = torch.randn(batch_size, seq_len, hidden_size)

        with torch.no_grad():
            output = model(input_ids=input_ids, hidden_states=hidden_states)

        assert len(output["logits_list"]) > 0

    def test_logits_finite(self, model: FastMTPSpeculator):
        batch_size, seq_len = 1, 8
        vocab_size = model.config.vocab_size
        hidden_size = model.config.hidden_size

        input_ids = torch.randint(0, min(vocab_size, 100), (batch_size, seq_len))
        hidden_states = torch.randn(batch_size, seq_len, hidden_size)

        with torch.no_grad():
            output = model(input_ids=input_ids, hidden_states=hidden_states)

        for step, logits in enumerate(output["logits_list"]):
            assert torch.isfinite(logits).all(), f"Non-finite logits at step {step}"

    def test_logits_shape(self, model: FastMTPSpeculator):
        """Step k logits must have shape [batch, seq_len - k - 2, vocab_size]."""
        batch_size, seq_len = 1, 8
        vocab_size = model.config.vocab_size
        hidden_size = model.config.hidden_size

        input_ids = torch.randint(0, min(vocab_size, 100), (batch_size, seq_len))
        hidden_states = torch.randn(batch_size, seq_len, hidden_size)

        with torch.no_grad():
            output = model(input_ids=input_ids, hidden_states=hidden_states)

        for step, logits in enumerate(output["logits_list"]):
            expected_len = seq_len - step - 2
            assert logits.shape == (batch_size, expected_len, vocab_size), (
                f"Step {step}: expected {(batch_size, expected_len, vocab_size)}, "
                f"got {logits.shape}"
            )


@pytest.mark.smoke
class TestRoundTrip:
    """Convert → load → save_pretrained → from_pretrained → config unchanged."""

    def test_round_trip_config(self, converted_dir: Path, tmp_path: Path):
        model = FastMTPSpeculator.from_pretrained(str(converted_dir))
        resaved = tmp_path / "resaved"
        model.save_pretrained(str(resaved))

        cfg_orig = json.loads((converted_dir / "config.json").read_text())
        cfg_resaved = json.loads((resaved / "config.json").read_text())

        assert (
            cfg_orig["speculators_model_type"] == cfg_resaved["speculators_model_type"]
        )
        assert cfg_orig["speculators_config"] == cfg_resaved["speculators_config"]

    def test_round_trip_loadable(self, converted_dir: Path, tmp_path: Path):
        model = FastMTPSpeculator.from_pretrained(str(converted_dir))
        resaved = tmp_path / "resaved_load"
        model.save_pretrained(str(resaved))

        model2 = FastMTPSpeculator.from_pretrained(str(resaved))
        assert isinstance(model2, FastMTPSpeculator)
        assert model2.embed_tokens is not None
        assert model2.lm_head is not None
        assert model2.config.num_speculative_steps == model.config.num_speculative_steps
