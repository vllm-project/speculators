"""Tests for loading ModelOpt NVFP4 verifier-owned projections."""

import json

import pytest
import torch
from safetensors.torch import save_file
from transformers.models.qwen3.configuration_qwen3 import Qwen3Config

from speculators import SpeculatorsConfig, VerifierConfig
from speculators.models.dflash import DFlashDraftModel, DFlashSpeculatorConfig
from speculators.proposals.greedy import GreedyTokenProposalConfig
from speculators.utils.loading import _dequantize_unswizzled_nvfp4


def _write_verifier(path) -> dict[str, torch.Tensor]:
    vocab_size = 4
    hidden_size = 32
    nibbles = (
        torch.arange(vocab_size * hidden_size, dtype=torch.uint8).reshape(
            vocab_size, hidden_size
        )
        % 16
    )
    tensors = {
        "model.embed_tokens.weight": torch.randn(
            vocab_size, hidden_size, dtype=torch.bfloat16
        ),
        "lm_head.weight": nibbles[:, 0::2] | (nibbles[:, 1::2] << 4),
        "lm_head.weight_scale": torch.tensor(
            [[1.0, 2.0], [2.0, 1.0], [0.5, 1.5], [1.5, 0.5]]
        ).to(torch.float8_e4m3fn),
        "lm_head.weight_scale_2": torch.tensor(0.5, dtype=torch.float32),
        "model.norm.weight": torch.ones(hidden_size, dtype=torch.bfloat16),
    }
    shard_name = "model.safetensors"
    save_file(tensors, path / shard_name)
    (path / "hf_quant_config.json").write_text(
        json.dumps(
            {
                "producer": {"name": "modelopt"},
                "quantization": {
                    "quant_algo": "MIXED_PRECISION",
                    "quantized_layers": {
                        "lm_head": {
                            "quant_algo": "W4A16_NVFP4",
                            "group_size": 16,
                        }
                    },
                },
            }
        )
    )
    return tensors


@pytest.mark.smoke
def test_reduced_vocab_nvfp4_projection_is_frozen_and_backpropagates(tmp_path):
    tensors = _write_verifier(tmp_path)
    transformer_config = Qwen3Config(
        vocab_size=4,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        max_position_embeddings=32,
        layer_types=["full_attention"],
        _attn_implementation="eager",
    )
    config = DFlashSpeculatorConfig(
        transformer_layer_config=transformer_config,
        draft_vocab_size=2,
        block_size=2,
        aux_hidden_state_layer_ids=[0],
        mask_token_id=0,
        speculators_config=SpeculatorsConfig(
            algorithm="dflash",
            proposal_methods=[GreedyTokenProposalConfig(speculative_tokens=1)],
            default_proposal_method="greedy",
            verifier=VerifierConfig(
                name_or_path=str(tmp_path),
                architectures=["Qwen3ForCausalLM"],
            ),
        ),
    )
    model = DFlashDraftModel(config)
    selector = torch.tensor([True, False, True, False])
    model.load_vocab_mappings(selector, torch.tensor([0, 2]))
    model.load_verifier_weights()

    expected = _dequantize_unswizzled_nvfp4(
        tensors["lm_head.weight"],
        tensors["lm_head.weight_scale"],
        tensors["lm_head.weight_scale_2"],
    )[selector]
    assert torch.equal(model.lm_head.weight, expected)
    assert torch.equal(model.verifier_lm_head.weight, expected)
    assert model.lm_head.weight.requires_grad is False
    assert model.verifier_lm_head.weight.requires_grad is False

    hidden_states = torch.randn(3, 32, requires_grad=True)
    model.lm_head(hidden_states).float().square().mean().backward()

    assert hidden_states.grad is not None
    assert torch.isfinite(hidden_states.grad).all()
    assert model.lm_head.weight.grad is None
    assert model.verifier_lm_head.weight.grad is None
