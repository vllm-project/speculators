"""Unit tests for FastMTPConverter."""

import json

import pytest
import torch
from safetensors.torch import save_file

from speculators.convert.fast_mtp.converter import FastMTPConverter
from speculators.models.fast_mtp import FastMTPSpeculator

H = 64
V = 256
INT = 128


def _make_weights() -> dict[str, torch.Tensor]:
    return {
        "model.embed_tokens.weight": torch.randn(V, H),
        "model.lm_head.weight": torch.randn(V, H),
        "model.mtp_layers.0.hidden_layernorm.weight": torch.ones(H),
        "model.mtp_layers.0.token_layernorm.weight": torch.ones(H),
        "model.mtp_layers.0.input_proj.weight": torch.randn(H, 2 * H),
        "model.mtp_layers.0.final_layernorm.weight": torch.ones(H),
        "model.mtp_layers.0.input_layernorm.weight": torch.ones(H),
        "model.mtp_layers.0.post_attention_layernorm.weight": torch.ones(H),
        "model.mtp_layers.0.self_attn.q_proj.weight": torch.randn(H, H),
        "model.mtp_layers.0.self_attn.q_proj.bias": torch.zeros(H),
        "model.mtp_layers.0.self_attn.k_proj.weight": torch.randn(H, H),
        "model.mtp_layers.0.self_attn.k_proj.bias": torch.zeros(H),
        "model.mtp_layers.0.self_attn.v_proj.weight": torch.randn(H, H),
        "model.mtp_layers.0.self_attn.v_proj.bias": torch.zeros(H),
        "model.mtp_layers.0.self_attn.o_proj.weight": torch.randn(H, H),
        "model.mtp_layers.0.mlp.gate_proj.weight": torch.randn(INT, H),
        "model.mtp_layers.0.mlp.up_proj.weight": torch.randn(INT, H),
        "model.mtp_layers.0.mlp.down_proj.weight": torch.randn(H, INT),
    }


@pytest.fixture
def tiny_qwen2_config_dict():
    return {
        "model_type": "qwen2",
        "hidden_size": H,
        "num_hidden_layers": 2,
        "num_attention_heads": 2,
        "num_key_value_heads": 2,
        "intermediate_size": INT,
        "vocab_size": V,
        "max_position_embeddings": 64,
        "rms_norm_eps": 1e-6,
        "rope_theta": 10000.0,
        "attention_bias": True,
    }


@pytest.fixture
def base_model_dir(tmp_path, tiny_qwen2_config_dict):
    d = tmp_path / "base_model"
    d.mkdir()
    config = {**tiny_qwen2_config_dict, "architectures": ["Qwen2ForCausalLM"]}
    (d / "config.json").write_text(json.dumps(config))
    save_file(
        {
            "model.embed_tokens.weight": torch.randn(V, H),
            "model.lm_head.weight": torch.randn(V, H),
        },
        str(d / "model.safetensors"),
    )
    return d


@pytest.fixture
def checkpoint(tmp_path, tiny_qwen2_config_dict):
    d = tmp_path / "source"
    d.mkdir()
    (d / "config.json").write_text(json.dumps(tiny_qwen2_config_dict))
    save_file(_make_weights(), str(d / "model.safetensors"))
    return d


@pytest.fixture
def converter():
    return FastMTPConverter()


@pytest.fixture
def converted_output(converter, checkpoint, base_model_dir, tmp_path):
    output = tmp_path / "converted"
    converter.convert(
        input_path=checkpoint,
        output_path=output,
        base_model=str(base_model_dir),
        validate=False,
    )
    return output


@pytest.mark.smoke
class TestVerifyFormat:
    def test_valid_checkpoint_passes(self, converter):
        converter._verify_qwen3_next_format(list(_make_weights()))

    def test_missing_mtp_prefix_raises(self, converter):
        with pytest.raises(ValueError, match="No MTP layer keys"):
            converter._verify_qwen3_next_format(["unrelated.weight", "also.unrelated"])


@pytest.mark.smoke
@pytest.mark.parametrize(
    ("key", "expected"),
    [
        ("model.embed_tokens.weight", "embed_tokens.weight"),
        ("model.lm_head.weight", "lm_head.weight"),
        (
            "model.mtp_layers.0.self_attn.q_proj.weight",
            "mtp_layers.0.self_attn.q_proj.weight",
        ),
        ("embed_tokens.weight", "embed_tokens.weight"),
    ],
)
def test_remap_key(converter, key, expected):
    assert converter._remap_key(key) == expected


@pytest.mark.smoke
class TestExtractWeights:
    def test_all_keys_remapped(self, converter, checkpoint):
        weights = converter._extract_weights(checkpoint, list(_make_weights()))

        assert all(not k.startswith("model.") for k in weights)
        assert "embed_tokens.weight" in weights
        assert "lm_head.weight" in weights
        assert all(
            k.startswith("mtp_layers.0.")
            or k in {"embed_tokens.weight", "lm_head.weight"}
            for k in weights
        )

    def test_shapes_preserved(self, converter, checkpoint):
        weights = converter._extract_weights(checkpoint, list(_make_weights()))

        assert weights["embed_tokens.weight"].shape == (V, H)
        assert weights["mtp_layers.0.input_proj.weight"].shape == (H, 2 * H)

    def test_sharded_checkpoint(self, converter, tmp_path, tiny_qwen2_config_dict):
        shard_dir = tmp_path / "sharded"
        shard_dir.mkdir()
        (shard_dir / "config.json").write_text(json.dumps(tiny_qwen2_config_dict))

        all_weights = _make_weights()
        keys = list(all_weights)
        mid = len(keys) // 2

        save_file(
            {k: all_weights[k] for k in keys[:mid]},
            str(shard_dir / "model-00001-of-00002.safetensors"),
        )
        save_file(
            {k: all_weights[k] for k in keys[mid:]},
            str(shard_dir / "model-00002-of-00002.safetensors"),
        )

        weight_map = dict.fromkeys(keys[:mid], "model-00001-of-00002.safetensors")
        weight_map.update(dict.fromkeys(keys[mid:], "model-00002-of-00002.safetensors"))
        (shard_dir / "model.safetensors.index.json").write_text(
            json.dumps({"metadata": {}, "weight_map": weight_map})
        )

        extracted = converter._extract_weights(shard_dir, keys)
        assert "embed_tokens.weight" in extracted or "lm_head.weight" in extracted
        assert all(not k.startswith("model.") for k in extracted)


@pytest.mark.smoke
class TestBuildConfig:
    def test_model_type_preserved(
        self, converter, base_model_dir, tiny_qwen2_config_dict
    ):
        config = converter._build_config(tiny_qwen2_config_dict, str(base_model_dir), 3)
        assert config.transformer_layer_config.model_type == "qwen2"

    def test_num_speculative_steps(
        self, converter, base_model_dir, tiny_qwen2_config_dict
    ):
        config = converter._build_config(tiny_qwen2_config_dict, str(base_model_dir), 5)
        assert config.num_speculative_steps == 5

    def test_verifier_path_set(self, converter, base_model_dir, tiny_qwen2_config_dict):
        config = converter._build_config(tiny_qwen2_config_dict, str(base_model_dir), 3)
        assert config.speculators_config.verifier.name_or_path == str(base_model_dir)

    def test_hidden_and_vocab_size(
        self, converter, base_model_dir, tiny_qwen2_config_dict
    ):
        config = converter._build_config(tiny_qwen2_config_dict, str(base_model_dir), 3)
        assert config.hidden_size == H
        assert config.vocab_size == V


@pytest.mark.smoke
class TestFullConvert:
    def test_output_files_created(self, converted_output):
        assert (converted_output / "config.json").exists()
        assert (converted_output / "model.safetensors").exists()

    def test_saved_config_has_correct_type(self, converted_output):
        saved = json.loads((converted_output / "config.json").read_text())
        assert saved["speculators_model_type"] == "mtp"

    def test_saved_checkpoint_loadable(
        self, converter, checkpoint, base_model_dir, tmp_path
    ):
        output = tmp_path / "converted_validated"
        converter.convert(
            input_path=checkpoint,
            output_path=output,
            base_model=str(base_model_dir),
            validate=True,
        )
        model = FastMTPSpeculator.from_pretrained(str(output))
        assert model.embed_tokens is not None
        assert model.lm_head is not None
        assert model.mtp_layers[0] is not None  # type: ignore[index]

    def test_num_speculative_steps_in_output(self, converted_output):
        model = FastMTPSpeculator.from_pretrained(str(converted_output))
        assert model.config.num_speculative_steps == 3
