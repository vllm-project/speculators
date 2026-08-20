"""
Unit tests for the loading module in the Speculators library.
"""

import json

import pytest
import torch
from safetensors.torch import save_file
from transformers import AutoModelForCausalLM

from speculators.utils.loading import (
    _dequantize_unswizzled_nvfp4,
    _resolve_file,
    _resolve_key,
    is_config_only_dir,
    load_model_layers,
)

# Test model from HuggingFace
TEST_MODEL_REPO = "nm-testing/tiny-testing-random-weights"
SMALL_MODEL_REPO = "nm-testing/tinysmokellama-3.2"


def _nvfp4_tensors(rows: int = 4, groups: int = 2) -> dict[str, torch.Tensor]:
    hidden_size = groups * 16
    nibbles = (
        torch.arange(rows * hidden_size, dtype=torch.uint8).reshape(rows, hidden_size)
        % 16
    )
    packed = nibbles[:, 0::2] | (nibbles[:, 1::2] << 4)
    scales = torch.arange(1, rows * groups + 1, dtype=torch.float32).reshape(
        rows, groups
    )
    return {
        "lm_head.weight": packed,
        "lm_head.weight_scale": scales.to(torch.float8_e4m3fn),
        "lm_head.weight_scale_2": torch.tensor(0.5, dtype=torch.float32),
    }


def _write_nvfp4_checkpoint(
    path,
    *,
    tensors: dict[str, torch.Tensor] | None = None,
    quant_algo: str = "W4A16_NVFP4",
    group_size: int = 16,
    write_quant_config: bool = True,
) -> None:
    tensors = tensors or _nvfp4_tensors()
    shard_name = "model-00001-of-00001.safetensors"
    save_file(tensors, path / shard_name)
    (path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": dict.fromkeys(tensors, shard_name)})
    )
    if write_quant_config:
        (path / "hf_quant_config.json").write_text(
            json.dumps(
                {
                    "producer": {"name": "modelopt"},
                    "quantization": {
                        "quant_algo": "MIXED_PRECISION",
                        "quantized_layers": {
                            "lm_head": {
                                "quant_algo": quant_algo,
                                "group_size": group_size,
                            }
                        },
                    },
                }
            )
        )


def _reference_nvfp4(tensors: dict[str, torch.Tensor]) -> torch.Tensor:
    packed = tensors["lm_head.weight"]
    low = packed & 0x0F
    high = (packed >> 4) & 0x0F
    nibbles = torch.stack((low, high), dim=-1).flatten(start_dim=1)
    e2m1 = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=torch.float32)
    values = e2m1[(nibbles & 0x07).long()]
    values *= torch.where((nibbles & 0x08) != 0, -1.0, 1.0)
    scales = tensors["lm_head.weight_scale"].float().repeat_interleave(16, dim=1)
    return (values * scales * tensors["lm_head.weight_scale_2"]).to(torch.bfloat16)


# is_config_only_dir Tests


@pytest.mark.smoke
def test_is_config_only_dir(tmp_path):
    # Missing directory and a directory without config.json are not config-only.
    assert is_config_only_dir(tmp_path / "does-not-exist") is False
    assert is_config_only_dir(tmp_path) is False

    # config.json present, no weights -> config-only.
    (tmp_path / "config.json").write_text("{}")
    assert is_config_only_dir(tmp_path) is True

    # A weight file makes it a full checkpoint.
    (tmp_path / "model.safetensors").write_text("")
    assert is_config_only_dir(tmp_path) is False


@pytest.mark.smoke
def test_is_config_only_dir_detects_bin_weights(tmp_path):
    (tmp_path / "config.json").write_text("{}")
    (tmp_path / "pytorch_model.bin").write_text("")

    assert is_config_only_dir(tmp_path) is False


@pytest.mark.smoke
@pytest.mark.parametrize(
    "index_file",
    ["model.safetensors.index.json", "pytorch_model.bin.index.json"],
)
def test_is_config_only_dir_detects_sharded_index(tmp_path, index_file):
    # A sharded-checkpoint manifest ends in .json (so it dodges the *.safetensors /
    # *.bin globs); it must still count as weights, not config-only.
    (tmp_path / "config.json").write_text("{}")
    (tmp_path / index_file).write_text("{}")

    assert is_config_only_dir(tmp_path) is False


# _resolve_key Tests


FAKE_WEIGHT_MAP = {
    "model.embed_tokens.weight": "shard-0.safetensors",
    "model.layers.0.self_attn.q_proj.weight": "shard-0.safetensors",
    "tok_embeddings.weight": "shard-1.safetensors",
    "output.weight": "shard-1.safetensors",
    "norm.weight": "shard-1.safetensors",
}


@pytest.mark.smoke
def test_resolve_key_exact_match():
    assert _resolve_key("model.embed_tokens.weight", FAKE_WEIGHT_MAP) == (
        "model.embed_tokens.weight"
    )


@pytest.mark.smoke
def test_resolve_key_suffix_match():
    assert _resolve_key("self_attn.q_proj.weight", FAKE_WEIGHT_MAP) == (
        "model.layers.0.self_attn.q_proj.weight"
    )


@pytest.mark.smoke
def test_resolve_key_alias_exact():
    wm = {"tok_embeddings.weight": "shard.safetensors"}
    assert _resolve_key("embed_tokens.weight", wm) == "tok_embeddings.weight"


@pytest.mark.smoke
def test_resolve_key_alias_suffix():
    wm = {"model.tok_embeddings.weight": "shard.safetensors"}
    assert _resolve_key("embed_tokens.weight", wm) == "model.tok_embeddings.weight"


@pytest.mark.smoke
def test_resolve_key_all_aliases():
    wm_lm = {"output.weight": "s.safetensors"}
    assert _resolve_key("lm_head.weight", wm_lm) == "output.weight"

    wm_norm = {"norm.weight": "s.safetensors"}
    assert _resolve_key("model.norm.weight", wm_norm) == "norm.weight"


@pytest.mark.smoke
def test_resolve_key_miss():
    assert _resolve_key("nonexistent.weight", FAKE_WEIGHT_MAP) is None


@pytest.mark.smoke
def test_resolve_key_prefers_exact_over_alias():
    wm = {
        "embed_tokens.weight": "shard-0.safetensors",
        "tok_embeddings.weight": "shard-1.safetensors",
    }
    assert _resolve_key("embed_tokens.weight", wm) == "embed_tokens.weight"


# _resolve_file Tests


@pytest.mark.sanity
def test_resolve_file_hub_download():
    """Test resolving a file from HuggingFace Hub using real model."""
    result = _resolve_file(TEST_MODEL_REPO, "config.json")

    assert result.exists()
    assert result.name == "config.json"


# load_model_layers Tests


@pytest.mark.sanity
@pytest.mark.parametrize(
    "test_model_repo",
    [
        TEST_MODEL_REPO,  # Multi-shard model
        SMALL_MODEL_REPO,  # Single-shard model
    ],
)
def test_load_model(test_model_repo: str):
    """Test loading layers from a model repository."""
    result = load_model_layers(
        ["model.embed_tokens.weight", "lm_head.weight"],
        test_model_repo,
    )

    assert len(result) == 2
    assert "model.embed_tokens.weight" in result
    assert "lm_head.weight" in result
    assert isinstance(result["model.embed_tokens.weight"], torch.Tensor)
    assert isinstance(result["lm_head.weight"], torch.Tensor)
    # Both should have same vocab dimension
    assert (
        result["model.embed_tokens.weight"].shape[0]
        == result["lm_head.weight"].shape[0]
    )
    # Verify CPU device
    assert result["model.embed_tokens.weight"].device.type == "cpu"


@pytest.mark.sanity
def test_load_model_layers_matches_full_model():
    """Test that tensors loaded via utility match those from full model loading."""
    # Load full model
    full_model = AutoModelForCausalLM.from_pretrained(
        TEST_MODEL_REPO,
        torch_dtype="auto",
    )

    # Get state dict from full model
    state_dict = full_model.state_dict()

    # Load specific layers using our utility
    layer_names = [
        "model.embed_tokens.weight",
        "lm_head.weight",
        "model.norm.weight",
        "model.layers.0.input_layernorm.weight",
        "model.layers.0.mlp.gate_proj.weight",
        "model.layers.1.mlp.down_proj.weight",
    ]

    loaded_tensors = load_model_layers(layer_names, TEST_MODEL_REPO)

    # Compare each tensor
    for layer_name in layer_names:
        assert layer_name in loaded_tensors, f"Layer {layer_name} not loaded"
        assert layer_name in state_dict, f"Layer {layer_name} not in state_dict"

        util_tensor = loaded_tensors[layer_name]
        model_tensor = state_dict[layer_name]

        # Check dtype matches
        assert util_tensor.dtype == model_tensor.dtype, (
            f"Dtype mismatch for {layer_name}: "
            f"{util_tensor.dtype} vs {model_tensor.dtype}"
        )

        # Check shape matches
        assert util_tensor.shape == model_tensor.shape, (
            f"Shape mismatch for {layer_name}: "
            f"{util_tensor.shape} vs {model_tensor.shape}"
        )

        # Check values are identical
        assert torch.equal(util_tensor, model_tensor), (
            f"Tensor values don't match for {layer_name}"
        )


@pytest.mark.smoke
def test_load_model_layers_materializes_modelopt_w4a16_nvfp4(tmp_path):
    tensors = _nvfp4_tensors()
    _write_nvfp4_checkpoint(tmp_path, tensors=tensors)

    loaded = load_model_layers(["lm_head.weight"], str(tmp_path))["lm_head.weight"]

    assert loaded.dtype == torch.bfloat16
    assert torch.equal(loaded, _reference_nvfp4(tensors))


@pytest.mark.smoke
@pytest.mark.parametrize(
    "selector",
    [
        torch.tensor([True, False, False, True]),
        torch.tensor([3, 1], dtype=torch.int64),
    ],
)
def test_nvfp4_select_then_dequantize_matches_dequantize_then_select(
    tmp_path, selector
):
    tensors = _nvfp4_tensors()
    _write_nvfp4_checkpoint(tmp_path, tensors=tensors)

    selected = load_model_layers(
        ["lm_head.weight"],
        str(tmp_path),
        row_indices={"lm_head.weight": selector},
    )["lm_head.weight"]
    full = load_model_layers(["lm_head.weight"], str(tmp_path))["lm_head.weight"]

    assert torch.equal(selected, full[selector])


@pytest.mark.smoke
def test_load_model_layers_selects_dense_rows(tmp_path):
    weight = torch.arange(24, dtype=torch.bfloat16).reshape(4, 6)
    save_file({"lm_head.weight": weight}, tmp_path / "model.safetensors")
    selector = torch.tensor([2, 0], dtype=torch.int64)

    selected = load_model_layers(
        ["lm_head.weight"],
        str(tmp_path),
        row_indices={"lm_head.weight": selector},
    )["lm_head.weight"]

    assert torch.equal(selected, weight[selector])


@pytest.mark.smoke
def test_packed_weight_requires_modelopt_quant_config(tmp_path):
    _write_nvfp4_checkpoint(tmp_path, write_quant_config=False)

    with pytest.raises(ValueError, match="requires hf_quant_config.json"):
        load_model_layers(["lm_head.weight"], str(tmp_path))


@pytest.mark.smoke
def test_packed_weight_rejects_non_modelopt_producer(tmp_path):
    _write_nvfp4_checkpoint(tmp_path)
    config_path = tmp_path / "hf_quant_config.json"
    config = json.loads(config_path.read_text())
    config["producer"]["name"] = "unknown"
    config_path.write_text(json.dumps(config))

    with pytest.raises(ValueError, match="requires a ModelOpt-produced"):
        load_model_layers(["lm_head.weight"], str(tmp_path))


@pytest.mark.smoke
@pytest.mark.parametrize(
    ("quant_algo", "group_size"),
    [("FP8", 16), ("W4A16_NVFP4", 32)],
)
def test_packed_weight_rejects_unsupported_quantization(
    tmp_path, quant_algo, group_size
):
    _write_nvfp4_checkpoint(tmp_path, quant_algo=quant_algo, group_size=group_size)

    with pytest.raises(ValueError, match="only the verified unswizzled"):
        load_model_layers(["lm_head.weight"], str(tmp_path))


@pytest.mark.smoke
def test_nvfp4_rejects_missing_scales(tmp_path):
    tensors = _nvfp4_tensors()
    del tensors["lm_head.weight_scale_2"]
    _write_nvfp4_checkpoint(tmp_path, tensors=tensors)

    with pytest.raises(ValueError, match="missing ModelOpt NVFP4 metadata"):
        load_model_layers(["lm_head.weight"], str(tmp_path))


@pytest.mark.smoke
def test_nvfp4_rejects_non_matrix_packed_weight(tmp_path):
    tensors = _nvfp4_tensors()
    tensors["lm_head.weight"] = torch.zeros(16, dtype=torch.uint8)
    _write_nvfp4_checkpoint(tmp_path, tensors=tensors)

    with pytest.raises(ValueError, match="expects 2-D weight/scales"):
        load_model_layers(["lm_head.weight"], str(tmp_path))


@pytest.mark.smoke
@pytest.mark.parametrize(
    ("tensor_name", "replacement", "match"),
    [
        (
            "lm_head.weight_scale",
            torch.ones((4, 2), dtype=torch.float32),
            "must use float8_e4m3fn",
        ),
        (
            "lm_head.weight_scale_2",
            torch.tensor(0.5, dtype=torch.bfloat16),
            "must be a scalar float32",
        ),
        (
            "lm_head.weight_scale_2",
            torch.ones(1, dtype=torch.float32),
            "must be a scalar float32",
        ),
        (
            "lm_head.weight_scale",
            torch.ones((3, 2), dtype=torch.float8_e4m3fn),
            "row mismatch",
        ),
        (
            "lm_head.weight_scale",
            torch.ones((4, 1), dtype=torch.float8_e4m3fn),
            "incompatible packed/scales shapes",
        ),
        (
            "lm_head.weight_scale",
            torch.ones(8, dtype=torch.float8_e4m3fn),
            "expects 2-D weight/scales",
        ),
    ],
)
def test_nvfp4_rejects_invalid_scale_layout(tmp_path, tensor_name, replacement, match):
    tensors = _nvfp4_tensors()
    tensors[tensor_name] = replacement
    _write_nvfp4_checkpoint(tmp_path, tensors=tensors)

    with pytest.raises(ValueError, match=match):
        load_model_layers(["lm_head.weight"], str(tmp_path))


@pytest.mark.smoke
@pytest.mark.parametrize(
    "selector",
    [
        torch.tensor([[True, False, True, False]]),
        torch.tensor([True, False]),
        torch.tensor([4], dtype=torch.int64),
        torch.tensor([0.0]),
    ],
)
def test_load_model_layers_rejects_invalid_row_selector(tmp_path, selector):
    _write_nvfp4_checkpoint(tmp_path)

    with pytest.raises((IndexError, TypeError, ValueError)):
        load_model_layers(
            ["lm_head.weight"],
            str(tmp_path),
            row_indices={"lm_head.weight": selector},
        )


@pytest.mark.smoke
def test_dequantize_unswizzled_nvfp4_matches_vllm_when_available():
    try:
        from vllm.model_executor.layers.quantization.utils.nvfp4_emulation_utils import (  # noqa: E501, PLC0415
            dequantize_to_dtype,
        )
    except (ImportError, RuntimeError) as error:
        pytest.skip(f"vLLM NVFP4 helper is unavailable: {error}")
    if not torch.cuda.is_available():
        pytest.skip("vLLM's NVFP4 parity helper requires CUDA tensors")

    tensors = _nvfp4_tensors(rows=3, groups=4)
    actual = _dequantize_unswizzled_nvfp4(
        tensors["lm_head.weight"],
        tensors["lm_head.weight_scale"],
        tensors["lm_head.weight_scale_2"],
        chunk_rows=2,
    )
    expected = dequantize_to_dtype(
        tensors["lm_head.weight"].cuda(),
        tensors["lm_head.weight_scale"].cuda(),
        tensors["lm_head.weight_scale_2"].cuda(),
        torch.bfloat16,
        block_size=16,
        swizzle=False,
    ).cpu()

    assert torch.equal(actual, expected)
