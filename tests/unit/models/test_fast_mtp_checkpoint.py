"""Unit tests for FastMTP checkpoint utilities."""

import json
from pathlib import Path

import pytest
import torch

from speculators.models.fast_mtp.checkpoint import (
    SKIP_KEYS,
    filter_mtp_keys,
    update_weight_index,
)

# ---------------------------------------------------------------------------
# filter_mtp_keys
# ---------------------------------------------------------------------------


def test_filter_mtp_keys_excludes_embed_tokens() -> None:
    state_dict = {
        "mtp.fc.weight": torch.zeros(4, 8),
        "embed_tokens.weight": torch.zeros(100, 8),
    }
    result = filter_mtp_keys(state_dict)
    assert "embed_tokens.weight" not in result
    assert "mtp.fc.weight" in result


def test_filter_mtp_keys_excludes_lm_head() -> None:
    state_dict = {
        "mtp.norm.weight": torch.zeros(8),
        "lm_head.weight": torch.zeros(100, 8),
    }
    result = filter_mtp_keys(state_dict)
    assert "lm_head.weight" not in result
    assert "mtp.norm.weight" in result


def test_filter_mtp_keys_keys_are_unchanged() -> None:
    """All mtp.* keys pass through with identical names — no renaming."""
    mtp_keys = [
        "mtp.pre_fc_norm_hidden.weight",
        "mtp.pre_fc_norm_embedding.weight",
        "mtp.fc.weight",
        "mtp.norm.weight",
        "mtp.layers.0.self_attn.q_proj.weight",
        "mtp.layers.0.mlp.gate_proj.weight",
    ]
    state_dict = {k: torch.zeros(1) for k in mtp_keys}
    result = filter_mtp_keys(state_dict)
    assert set(result.keys()) == set(mtp_keys)


def test_filter_mtp_keys_complete_state_dict() -> None:
    """A complete MTP training state dict: 6 mtp.* keys kept, SKIP_KEYS excluded."""
    state_dict = {
        "mtp.pre_fc_norm_hidden.weight": torch.zeros(8),
        "mtp.pre_fc_norm_embedding.weight": torch.zeros(8),
        "mtp.fc.weight": torch.zeros(4, 8),
        "mtp.norm.weight": torch.zeros(8),
        "mtp.layers.0.self_attn.q_proj.weight": torch.zeros(8, 8),
        "mtp.layers.0.mlp.gate_proj.weight": torch.zeros(16, 8),
        "embed_tokens.weight": torch.zeros(100, 8),
        "lm_head.weight": torch.zeros(100, 8),
    }
    result = filter_mtp_keys(state_dict)
    assert len(result) == 6
    assert "mtp.pre_fc_norm_hidden.weight" in result
    assert "mtp.pre_fc_norm_embedding.weight" in result
    assert "mtp.fc.weight" in result
    assert "mtp.norm.weight" in result
    assert "mtp.layers.0.self_attn.q_proj.weight" in result
    assert "mtp.layers.0.mlp.gate_proj.weight" in result


@pytest.mark.parametrize("skip_key", list(SKIP_KEYS))
def test_filter_mtp_keys_skips_all_skip_keys(skip_key: str) -> None:
    state_dict = {skip_key: torch.zeros(1), "mtp.fc.weight": torch.zeros(4, 8)}
    result = filter_mtp_keys(state_dict)
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

    output_dir = tmp_path / "output"
    output_dir.mkdir()

    with pytest.raises(FileNotFoundError, match="model.safetensors.index.json"):
        update_weight_index(verifier_dir, output_dir, [], "mtp.safetensors")
