"""Unit tests for FastMTP checkpoint key remapping."""

import json
from pathlib import Path

import pytest
import torch

from speculators.models.fast_mtp.checkpoint import (
    _MTP_TOP_LEVEL_SUFFIXES,
    _SPECULATORS_PREFIX,
    SKIP_KEYS,
    remap_key,
    remap_keys,
    update_weight_index,
)

# ---------------------------------------------------------------------------
# remap_key — top-level mixin weights (mtp.<suffix>)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "suffix",
    list(_MTP_TOP_LEVEL_SUFFIXES),
)
def test_remap_key_top_level_mixin(suffix: str) -> None:
    """Top-level mixin suffixes map to mtp.{suffix} (not mtp.layers.0.{suffix})."""
    key = _SPECULATORS_PREFIX + suffix
    result = remap_key(key)
    assert result == f"mtp.{suffix}"
    assert "layers.0" not in result


def test_remap_key_pre_fc_norm_hidden() -> None:
    result = remap_key("mtp_layers.0.pre_fc_norm_hidden.weight")
    assert result == "mtp.pre_fc_norm_hidden.weight"


def test_remap_key_pre_fc_norm_embedding() -> None:
    result = remap_key("mtp_layers.0.pre_fc_norm_embedding.weight")
    assert result == "mtp.pre_fc_norm_embedding.weight"


def test_remap_key_fc() -> None:
    result = remap_key("mtp_layers.0.fc.weight")
    assert result == "mtp.fc.weight"


def test_remap_key_norm() -> None:
    result = remap_key("mtp_layers.0.norm.weight")
    assert result == "mtp.norm.weight"


# ---------------------------------------------------------------------------
# remap_key — decoder layer weights (mtp.layers.0.{suffix})
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "suffix",
    [
        "self_attn.q_proj.weight",
        "self_attn.k_proj.weight",
        "self_attn.v_proj.weight",
        "self_attn.o_proj.weight",
        "mlp.gate_proj.weight",
        "mlp.up_proj.weight",
        "mlp.down_proj.weight",
        "input_layernorm.weight",
        "post_attention_layernorm.weight",
    ],
)
def test_remap_key_decoder_layer_passthrough(suffix: str) -> None:
    """Non-mixin MTP weights map to mtp.layers.0.{suffix}."""
    key = _SPECULATORS_PREFIX + suffix
    result = remap_key(key)
    assert result == f"mtp.layers.0.{suffix}"


def test_remap_key_bad_prefix_raises() -> None:
    with pytest.raises(ValueError, match="expected prefix"):
        remap_key("model.mtp_layers.0.fc.weight")


# ---------------------------------------------------------------------------
# remap_keys — full state dict
# ---------------------------------------------------------------------------


def test_remap_keys_skips_embed_tokens() -> None:
    state_dict = {
        "mtp_layers.0.fc.weight": torch.zeros(4, 8),
        "embed_tokens.weight": torch.zeros(100, 8),
    }
    result = remap_keys(state_dict)
    assert "embed_tokens.weight" not in result
    assert "mtp.fc.weight" in result


def test_remap_keys_skips_lm_head() -> None:
    state_dict = {
        "mtp_layers.0.norm.weight": torch.zeros(8),
        "lm_head.weight": torch.zeros(100, 8),
    }
    result = remap_keys(state_dict)
    assert "lm_head.weight" not in result
    assert "mtp.norm.weight" in result


def test_remap_keys_complete_state_dict() -> None:
    """A complete MTP state dict remaps all 4 mixin + layer keys correctly."""
    state_dict = {
        "mtp_layers.0.pre_fc_norm_hidden.weight": torch.zeros(8),
        "mtp_layers.0.pre_fc_norm_embedding.weight": torch.zeros(8),
        "mtp_layers.0.fc.weight": torch.zeros(4, 8),
        "mtp_layers.0.norm.weight": torch.zeros(8),
        "mtp_layers.0.self_attn.q_proj.weight": torch.zeros(8, 8),
        "mtp_layers.0.mlp.gate_proj.weight": torch.zeros(16, 8),
        "embed_tokens.weight": torch.zeros(100, 8),  # must be skipped
        "lm_head.weight": torch.zeros(100, 8),  # must be skipped
    }
    result = remap_keys(state_dict)

    assert len(result) == 6  # 4 mixin + 2 layer, 2 embed/lm_head skipped
    assert "mtp.pre_fc_norm_hidden.weight" in result
    assert "mtp.pre_fc_norm_embedding.weight" in result
    assert "mtp.fc.weight" in result
    assert "mtp.norm.weight" in result
    assert "mtp.layers.0.self_attn.q_proj.weight" in result
    assert "mtp.layers.0.mlp.gate_proj.weight" in result


def test_remap_keys_skips_all_skip_keys() -> None:
    """All keys in SKIP_KEYS are dropped."""
    state_dict = {k: torch.zeros(1) for k in SKIP_KEYS}
    state_dict["mtp_layers.0.fc.weight"] = torch.zeros(4, 8)
    result = remap_keys(state_dict)
    for skip_key in SKIP_KEYS:
        assert skip_key not in result
    assert "mtp.fc.weight" in result


# ---------------------------------------------------------------------------
# update_weight_index
# ---------------------------------------------------------------------------


def _make_index(mtp_shard: str, other_shard: str) -> dict:
    return {
        "metadata": {"total_size": 1000},
        "weight_map": {
            "mtp.pre_fc_norm_hidden.weight": mtp_shard,
            "mtp.layers.0.self_attn.q_proj.weight": mtp_shard,
            "model.layers.0.self_attn.q_proj.weight": other_shard,
        },
    }


def test_update_weight_index_removes_stale_mtp_entries(tmp_path: Path) -> None:
    verifier_dir = tmp_path / "verifier"
    verifier_dir.mkdir()
    index = _make_index("original_mtp.safetensors", "model-00001.safetensors")
    (verifier_dir / "model.safetensors.index.json").write_text(json.dumps(index))

    output_dir = tmp_path / "output"
    output_dir.mkdir()
    update_weight_index(verifier_dir, output_dir, [], "new_mtp.safetensors")

    result = json.loads((output_dir / "model.safetensors.index.json").read_text())
    weight_map = result["weight_map"]

    assert "mtp.pre_fc_norm_hidden.weight" not in weight_map
    assert "mtp.layers.0.self_attn.q_proj.weight" not in weight_map
    assert "model.layers.0.self_attn.q_proj.weight" in weight_map


def test_update_weight_index_adds_new_shard_entries(tmp_path: Path) -> None:
    verifier_dir = tmp_path / "verifier"
    verifier_dir.mkdir()
    index = _make_index("original.safetensors", "other.safetensors")
    (verifier_dir / "model.safetensors.index.json").write_text(json.dumps(index))

    output_dir = tmp_path / "output"
    output_dir.mkdir()
    new_keys = ["mtp.fc.weight", "mtp.norm.weight"]
    update_weight_index(verifier_dir, output_dir, new_keys, "finetuned.safetensors")

    result = json.loads((output_dir / "model.safetensors.index.json").read_text())
    weight_map = result["weight_map"]

    for k in new_keys:
        assert weight_map[k] == "finetuned.safetensors"


def test_update_weight_index_missing_index_raises(tmp_path: Path) -> None:
    verifier_dir = tmp_path / "verifier"
    verifier_dir.mkdir()
    # No model.safetensors.index.json

    output_dir = tmp_path / "output"
    output_dir.mkdir()

    with pytest.raises(FileNotFoundError, match="model.safetensors.index.json"):
        update_weight_index(verifier_dir, output_dir, [], "mtp.safetensors")
