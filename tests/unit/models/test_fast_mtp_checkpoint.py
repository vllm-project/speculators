"""Tests for FastMTP checkpoint utilities."""

import json
from pathlib import Path

import pytest
import torch

from speculators.models.fast_mtp.checkpoint import (
    SKIP_KEYS,
    filter_mtp_keys,
    update_weight_index,
)


@pytest.mark.parametrize("skip_key", list(SKIP_KEYS))
def test_filter_mtp_keys_excludes_skip_keys(skip_key: str) -> None:
    result = filter_mtp_keys(
        {skip_key: torch.zeros(1), "mtp.fc.weight": torch.zeros(4, 8)}
    )
    assert skip_key not in result
    assert "mtp.fc.weight" in result


def test_filter_mtp_keys_no_renaming() -> None:
    keys = [
        "mtp.pre_fc_norm_hidden.weight",
        "mtp.pre_fc_norm_embedding.weight",
        "mtp.fc.weight",
        "mtp.norm.weight",
        "mtp.layers.0.self_attn.q_proj.weight",
    ]
    result = filter_mtp_keys({k: torch.zeros(1) for k in keys})
    assert set(result.keys()) == set(keys)


def _make_index(mtp_shard: str, other_shard: str) -> dict:
    return {
        "metadata": {"total_size": 1000},
        "weight_map": {
            "mtp.pre_fc_norm_hidden.weight": mtp_shard,
            "mtp.layers.0.self_attn.q_proj.weight": mtp_shard,
            "model.layers.0.self_attn.q_proj.weight": other_shard,
        },
    }


def test_update_weight_index_replaces_mtp_entries(tmp_path: Path) -> None:
    verifier_dir = tmp_path / "verifier"
    verifier_dir.mkdir()
    (verifier_dir / "model.safetensors.index.json").write_text(
        json.dumps(_make_index("old.safetensors", "model-00001.safetensors"))
    )
    output_dir = tmp_path / "output"
    output_dir.mkdir()

    new_keys = ["mtp.fc.weight", "mtp.norm.weight"]
    update_weight_index(verifier_dir, output_dir, new_keys, "finetuned.safetensors")

    wm = json.loads((output_dir / "model.safetensors.index.json").read_text())[
        "weight_map"
    ]
    assert "mtp.pre_fc_norm_hidden.weight" not in wm
    assert "mtp.layers.0.self_attn.q_proj.weight" not in wm
    assert "model.layers.0.self_attn.q_proj.weight" in wm
    for k in new_keys:
        assert wm[k] == "finetuned.safetensors"


def test_update_weight_index_missing_index_raises(tmp_path: Path) -> None:
    (tmp_path / "verifier").mkdir()
    (tmp_path / "output").mkdir()
    with pytest.raises(FileNotFoundError, match="model.safetensors.index.json"):
        update_weight_index(
            tmp_path / "verifier", tmp_path / "output", [], "mtp.safetensors"
        )
