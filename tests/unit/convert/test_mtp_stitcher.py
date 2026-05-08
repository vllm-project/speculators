"""Unit tests for MTP stitcher.

Tests cover:
- Inverse key remapping (speculators mtp_layers.0.* -> native mtp.* format)
- Round-trip: converter remap then stitcher remap = identity
- Frozen key filtering (embed_tokens, lm_head excluded)
- MoE expert weight unfusing (packed 3D tensors -> individual per-expert)
- MoE round-trip: fuse then unfuse preserves values
- End-to-end stitching with synthetic checkpoints
"""

import json

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import save_file

from speculators.convert.mtp.converter import (
    EXACT_KEY_MAP,
    PREFIX_KEY_MAP,
    MTPConverter,
)
from speculators.convert.mtp.stitcher import (
    INVERSE_EXACT_KEY_MAP,
    INVERSE_PREFIX_KEY_MAP,
    MTPStitcher,
)


class TestInverseKeyRemapping:
    """Test key remapping from speculators format to native format."""

    def test_input_proj_remapped(self):
        assert (
            MTPStitcher._remap_key("mtp_layers.0.input_proj.weight") == "mtp.fc.weight"
        )

    def test_final_layernorm_remapped(self):
        assert (
            MTPStitcher._remap_key("mtp_layers.0.final_layernorm.weight")
            == "mtp.norm.weight"
        )

    def test_hidden_layernorm_remapped(self):
        assert (
            MTPStitcher._remap_key("mtp_layers.0.hidden_layernorm.weight")
            == "mtp.pre_fc_norm_hidden.weight"
        )

    def test_token_layernorm_remapped(self):
        assert (
            MTPStitcher._remap_key("mtp_layers.0.token_layernorm.weight")
            == "mtp.pre_fc_norm_embedding.weight"
        )

    def test_layer_key_remapped(self):
        assert (
            MTPStitcher._remap_key("mtp_layers.0.self_attn.q_proj.weight")
            == "mtp.layers.0.self_attn.q_proj.weight"
        )

    def test_mlp_key_remapped(self):
        assert (
            MTPStitcher._remap_key("mtp_layers.0.mlp.gate_proj.weight")
            == "mtp.layers.0.mlp.gate_proj.weight"
        )

    def test_unknown_key_passthrough(self):
        key = "some.other.weight"
        assert MTPStitcher._remap_key(key) == key


class TestRoundTrip:
    """Verify converter remap then stitcher remap produces the original key."""

    @pytest.mark.sanity
    def test_exact_keys_round_trip(self):
        for native_key in EXACT_KEY_MAP:
            speculators_key = MTPConverter._remap_key(native_key)
            restored = MTPStitcher._remap_key(speculators_key)
            assert restored == native_key, (
                f"Round-trip failed: {native_key} -> {speculators_key} -> {restored}"
            )

    @pytest.mark.sanity
    def test_prefix_keys_round_trip(self):
        for native_prefix, _ in PREFIX_KEY_MAP:
            native_key = f"{native_prefix}weight"
            speculators_key = MTPConverter._remap_key(native_key)
            restored = MTPStitcher._remap_key(speculators_key)
            assert restored == native_key, (
                f"Round-trip failed: {native_key} -> {speculators_key} -> {restored}"
            )

    @pytest.mark.sanity
    def test_mtp_layer_deep_key_round_trip(self):
        native_key = "mtp.layers.0.self_attn.q_proj.weight"
        speculators_key = MTPConverter._remap_key(native_key)
        restored = MTPStitcher._remap_key(speculators_key)
        assert restored == native_key

    @pytest.mark.sanity
    def test_inverse_maps_cover_forward_maps(self):
        """Every forward exact mapping has a corresponding inverse."""
        for native, speculators in EXACT_KEY_MAP.items():
            assert speculators in INVERSE_EXACT_KEY_MAP
            assert INVERSE_EXACT_KEY_MAP[speculators] == native

        assert len(INVERSE_PREFIX_KEY_MAP) == len(PREFIX_KEY_MAP)


class TestFrozenKeyFiltering:
    """Test that frozen weights are excluded during stitching."""

    @pytest.mark.sanity
    def test_embed_tokens_excluded(self):
        weights = {
            "embed_tokens.weight": torch.randn(100, 64),
            "mtp_layers.0.input_proj.weight": torch.randn(64, 128),
        }
        stitcher = MTPStitcher()
        filtered = stitcher._filter_frozen_keys(weights)
        assert "embed_tokens.weight" not in filtered
        assert "mtp_layers.0.input_proj.weight" in filtered

    @pytest.mark.sanity
    def test_lm_head_excluded(self):
        weights = {
            "lm_head.weight": torch.randn(100, 64),
            "mtp_layers.0.input_proj.weight": torch.randn(64, 128),
        }
        stitcher = MTPStitcher()
        filtered = stitcher._filter_frozen_keys(weights)
        assert "lm_head.weight" not in filtered

    def test_mtp_keys_preserved(self):
        weights = {
            "mtp_layers.0.input_proj.weight": torch.randn(64, 128),
            "mtp_layers.0.self_attn.q_proj.weight": torch.randn(64, 64),
        }
        stitcher = MTPStitcher()
        filtered = stitcher._filter_frozen_keys(weights)
        assert len(filtered) == 2


class TestMoEExpertUnfusing:
    """Test MoE expert weight unfusing (inverse of converter's fusing)."""

    @pytest.fixture
    def dense_weights(self):
        return {
            "mtp_layers.0.self_attn.q_proj.weight": torch.randn(64, 64),
            "mtp_layers.0.input_proj.weight": torch.randn(64, 128),
        }

    @pytest.fixture
    def fused_weights(self):
        num_experts = 4
        gate_up = torch.randn(num_experts, 64, 64)
        down = torch.randn(num_experts, 64, 32)
        return {
            "mtp_layers.0.mlp.experts.gate_up_proj": gate_up,
            "mtp_layers.0.mlp.experts.down_proj": down,
            "mtp_layers.0.self_attn.q_proj.weight": torch.randn(64, 64),
        }

    @pytest.mark.sanity
    def test_no_fused_experts_passthrough(self, dense_weights):
        result = MTPStitcher._unfuse_moe_experts(dense_weights)
        assert result == dense_weights

    @pytest.mark.sanity
    def test_gate_proj_split_to_per_expert(self, fused_weights):
        result = MTPStitcher._unfuse_moe_experts(fused_weights)
        for i in range(4):
            key = f"mtp_layers.0.mlp.experts.{i}.gate_proj.weight"
            assert key in result
            assert result[key].shape == (32, 64)

    @pytest.mark.sanity
    def test_up_proj_split_to_per_expert(self, fused_weights):
        result = MTPStitcher._unfuse_moe_experts(fused_weights)
        for i in range(4):
            key = f"mtp_layers.0.mlp.experts.{i}.up_proj.weight"
            assert key in result
            assert result[key].shape == (32, 64)

    @pytest.mark.sanity
    def test_down_proj_split_to_per_expert(self, fused_weights):
        result = MTPStitcher._unfuse_moe_experts(fused_weights)
        for i in range(4):
            key = f"mtp_layers.0.mlp.experts.{i}.down_proj.weight"
            assert key in result
            assert result[key].shape == (64, 32)

    @pytest.mark.sanity
    def test_fused_keys_removed(self, fused_weights):
        result = MTPStitcher._unfuse_moe_experts(fused_weights)
        assert "mtp_layers.0.mlp.experts.gate_up_proj" not in result
        assert "mtp_layers.0.mlp.experts.down_proj" not in result

    @pytest.mark.sanity
    def test_non_expert_keys_preserved(self, fused_weights):
        result = MTPStitcher._unfuse_moe_experts(fused_weights)
        assert "mtp_layers.0.self_attn.q_proj.weight" in result

    @pytest.mark.sanity
    def test_gate_up_content_correct(self):
        gate_up = torch.randn(1, 64, 64)
        weights = {
            "mtp_layers.0.mlp.experts.gate_up_proj": gate_up,
            "mtp_layers.0.mlp.experts.down_proj": torch.randn(1, 64, 32),
        }
        result = MTPStitcher._unfuse_moe_experts(weights)
        gate = result["mtp_layers.0.mlp.experts.0.gate_proj.weight"]
        up = result["mtp_layers.0.mlp.experts.0.up_proj.weight"]
        assert torch.equal(gate, gate_up[0, :32])
        assert torch.equal(up, gate_up[0, 32:])

    def test_unfused_tensors_contiguous(self, fused_weights):
        result = MTPStitcher._unfuse_moe_experts(fused_weights)
        for key, tensor in result.items():
            assert tensor.is_contiguous(), f"{key} is not contiguous"


class TestMoERoundTrip:
    """Verify fuse then unfuse preserves tensor values."""

    @pytest.mark.sanity
    def test_fuse_then_unfuse_preserves_values(self):
        num_experts = 4
        original = {}
        for i in range(num_experts):
            original[f"mtp_layers.0.mlp.experts.{i}.gate_proj.weight"] = torch.randn(
                32, 64
            )
            original[f"mtp_layers.0.mlp.experts.{i}.up_proj.weight"] = torch.randn(
                32, 64
            )
            original[f"mtp_layers.0.mlp.experts.{i}.down_proj.weight"] = torch.randn(
                64, 32
            )

        fused = MTPConverter._fuse_moe_experts(original)
        unfused = MTPStitcher._unfuse_moe_experts(fused)

        for key, value in original.items():
            assert key in unfused, f"Missing key after round-trip: {key}"
            assert torch.equal(value, unfused[key]), f"Value mismatch for {key}"


class TestStitchEndToEnd:
    """End-to-end stitching with synthetic checkpoints on disk."""

    @pytest.fixture
    def verifier_checkpoint(self, tmp_path):
        """Create a synthetic verifier checkpoint with MTP weights."""
        verifier_dir = tmp_path / "verifier"
        verifier_dir.mkdir()

        weights = {
            "model.layers.0.self_attn.q_proj.weight": torch.randn(64, 64),
            "model.layers.0.mlp.gate_proj.weight": torch.randn(128, 64),
            "model.embed_tokens.weight": torch.randn(100, 64),
            "lm_head.weight": torch.randn(100, 64),
            "mtp.fc.weight": torch.randn(64, 128),
            "mtp.norm.weight": torch.randn(64),
            "mtp.pre_fc_norm_hidden.weight": torch.randn(64),
            "mtp.pre_fc_norm_embedding.weight": torch.randn(64),
            "mtp.layers.0.self_attn.q_proj.weight": torch.randn(64, 64),
            "mtp.layers.0.mlp.gate_proj.weight": torch.randn(128, 64),
        }

        save_file(weights, str(verifier_dir / "model.safetensors"))

        config = {
            "model_type": "qwen3_next",
            "hidden_size": 64,
            "vocab_size": 100,
        }
        (verifier_dir / "config.json").write_text(json.dumps(config))
        (verifier_dir / "tokenizer.json").write_text('{"type": "test"}')

        return verifier_dir, weights

    @pytest.fixture
    def finetuned_checkpoint(self, tmp_path):
        """Create a synthetic finetuned MTP checkpoint in speculators format."""
        finetuned_dir = tmp_path / "finetuned"
        finetuned_dir.mkdir()

        weights = {
            "embed_tokens.weight": torch.randn(100, 64),
            "lm_head.weight": torch.randn(100, 64),
            "mtp_layers.0.input_proj.weight": torch.randn(64, 128),
            "mtp_layers.0.final_layernorm.weight": torch.randn(64),
            "mtp_layers.0.hidden_layernorm.weight": torch.randn(64),
            "mtp_layers.0.token_layernorm.weight": torch.randn(64),
            "mtp_layers.0.self_attn.q_proj.weight": torch.randn(64, 64),
            "mtp_layers.0.mlp.gate_proj.weight": torch.randn(128, 64),
        }

        save_file(weights, str(finetuned_dir / "model.safetensors"))
        return finetuned_dir, weights

    @pytest.mark.sanity
    def test_stitch_single_file(
        self, tmp_path, verifier_checkpoint, finetuned_checkpoint
    ):
        verifier_dir, _ = verifier_checkpoint
        finetuned_dir, finetuned_weights = finetuned_checkpoint
        output_dir = tmp_path / "output"

        stitcher = MTPStitcher()
        stitcher.stitch(finetuned_dir, verifier_dir, output_dir)

        with safe_open(str(output_dir / "model.safetensors"), framework="pt") as f:
            output_keys = set(f.keys())
            mtp_fc = f.get_tensor("mtp.fc.weight")

        assert "mtp.fc.weight" in output_keys
        assert "mtp.norm.weight" in output_keys
        assert "mtp.layers.0.self_attn.q_proj.weight" in output_keys
        assert torch.equal(mtp_fc, finetuned_weights["mtp_layers.0.input_proj.weight"])

    @pytest.mark.sanity
    def test_frozen_weights_not_overwritten(
        self, tmp_path, verifier_checkpoint, finetuned_checkpoint
    ):
        verifier_dir, verifier_weights = verifier_checkpoint
        finetuned_dir, _ = finetuned_checkpoint
        output_dir = tmp_path / "output"

        stitcher = MTPStitcher()
        stitcher.stitch(finetuned_dir, verifier_dir, output_dir)

        with safe_open(str(output_dir / "model.safetensors"), framework="pt") as f:
            embed = f.get_tensor("model.embed_tokens.weight")

        assert torch.equal(embed, verifier_weights["model.embed_tokens.weight"])

    @pytest.mark.sanity
    def test_metadata_files_copied(
        self, tmp_path, verifier_checkpoint, finetuned_checkpoint
    ):
        verifier_dir, _ = verifier_checkpoint
        finetuned_dir, _ = finetuned_checkpoint
        output_dir = tmp_path / "output"

        stitcher = MTPStitcher()
        stitcher.stitch(finetuned_dir, verifier_dir, output_dir)

        assert (output_dir / "config.json").exists()
        assert (output_dir / "tokenizer.json").exists()

    @pytest.mark.sanity
    def test_non_mtp_weights_unchanged(
        self, tmp_path, verifier_checkpoint, finetuned_checkpoint
    ):
        verifier_dir, verifier_weights = verifier_checkpoint
        finetuned_dir, _ = finetuned_checkpoint
        output_dir = tmp_path / "output"

        stitcher = MTPStitcher()
        stitcher.stitch(finetuned_dir, verifier_dir, output_dir)

        with safe_open(str(output_dir / "model.safetensors"), framework="pt") as f:
            layers_weight = f.get_tensor("model.layers.0.self_attn.q_proj.weight")

        assert torch.equal(
            layers_weight,
            verifier_weights["model.layers.0.self_attn.q_proj.weight"],
        )

    def test_stitch_sharded(self, tmp_path):
        verifier_dir = tmp_path / "verifier_sharded"
        verifier_dir.mkdir()

        shard1_weights = {
            "model.layers.0.self_attn.q_proj.weight": torch.randn(64, 64),
            "model.embed_tokens.weight": torch.randn(100, 64),
        }
        shard2_weights = {
            "mtp.fc.weight": torch.randn(64, 128),
            "mtp.norm.weight": torch.randn(64),
            "mtp.layers.0.self_attn.q_proj.weight": torch.randn(64, 64),
            "lm_head.weight": torch.randn(100, 64),
        }

        save_file(
            shard1_weights, str(verifier_dir / "model-00001-of-00002.safetensors")
        )
        save_file(
            shard2_weights, str(verifier_dir / "model-00002-of-00002.safetensors")
        )

        weight_map = {}
        for key in shard1_weights:
            weight_map[key] = "model-00001-of-00002.safetensors"
        for key in shard2_weights:
            weight_map[key] = "model-00002-of-00002.safetensors"

        index = {"metadata": {}, "weight_map": weight_map}
        (verifier_dir / "model.safetensors.index.json").write_text(json.dumps(index))
        (verifier_dir / "config.json").write_text('{"model_type": "qwen3_next"}')

        finetuned_dir = tmp_path / "finetuned_sharded"
        finetuned_dir.mkdir()
        finetuned_weights = {
            "embed_tokens.weight": torch.randn(100, 64),
            "lm_head.weight": torch.randn(100, 64),
            "mtp_layers.0.input_proj.weight": torch.randn(64, 128),
            "mtp_layers.0.final_layernorm.weight": torch.randn(64),
            "mtp_layers.0.self_attn.q_proj.weight": torch.randn(64, 64),
        }
        save_file(finetuned_weights, str(finetuned_dir / "model.safetensors"))

        output_dir = tmp_path / "output_sharded"
        stitcher = MTPStitcher()
        stitcher.stitch(finetuned_dir, verifier_dir, output_dir)

        with safe_open(
            str(output_dir / "model-00001-of-00002.safetensors"), framework="pt"
        ) as f:
            shard1_embed = f.get_tensor("model.embed_tokens.weight")
        assert torch.equal(shard1_embed, shard1_weights["model.embed_tokens.weight"])

        with safe_open(
            str(output_dir / "model-00002-of-00002.safetensors"), framework="pt"
        ) as f:
            mtp_fc = f.get_tensor("mtp.fc.weight")
        assert torch.equal(mtp_fc, finetuned_weights["mtp_layers.0.input_proj.weight"])
