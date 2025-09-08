from __future__ import annotations

from typing import Any

import pytest
import torch

from tests.unit.mock.hf_factory import (
    MockPretrainedTransformersFactory,
    PretrainedBundle,
)

__all__ = ["mock_eagle3_config_dict", "mock_eagle3_state_dict"]


def mock_eagle3_config_dict(
    verifier_dict: dict[str, Any], name_or_path: str | None
) -> dict[str, Any]:
    return {
        "_name_or_path": name_or_path,
        "architectures": ["LlamaForCausalLM"],
        "bos_token_id": verifier_dict.get("bos_token_id", 128000),
        "eos_token_id": verifier_dict.get("eos_token_id", 128001),
        "hidden_act": "silu",
        "hidden_size": verifier_dict.get("hidden_size", 4096),
        "initializer_range": 0.02,
        "intermediate_size": verifier_dict.get("intermediate_size", 14336),
        "max_position_embeddings": 2048,
        "model_type": "llama",
        "num_attention_heads": verifier_dict.get("num_attention_heads", 32),
        "num_key_value_heads": verifier_dict.get("num_key_value_heads", 8),
        "num_hidden_layers": 1,
        "pad_token_id": 0,
        "rms_norm_eps": 1e-05,
        "tie_word_embeddings": False,
        "torch_dtype": "float16",
        "transformers_version": "4.28.1",
        "use_cache": True,
        "vocab_size": verifier_dict.get("vocab_size", 128256),
        "draft_vocab_size": verifier_dict.get("vocab_size", 128256) // 4,
    }


def mock_eagle3_state_dict(verifier_dict: dict[str, Any]) -> dict[str, torch.Tensor]:
    eagle3_dict = mock_eagle3_config_dict(verifier_dict, name_or_path=None)

    return {
        "d2t": torch.randint(
            0,
            eagle3_dict["vocab_size"] + 1,
            (eagle3_dict["draft_vocab_size"],),
            dtype=torch.long,
        ),
        "t2d": torch.randint(0, 2, (eagle3_dict["vocab_size"],), dtype=torch.bool),
        "embed_tokens.weight": torch.randn(
            eagle3_dict["vocab_size"], eagle3_dict["hidden_size"]
        ),
        "fc.weight": torch.randn(
            eagle3_dict["hidden_size"], 3 * eagle3_dict["hidden_size"]
        ),
        "fc.bias": torch.randn(eagle3_dict["hidden_size"]),
        "midlayer.hidden_norm.weight": torch.randn(eagle3_dict["hidden_size"]),
        "midlayer.input_layernorm.weight": torch.randn(eagle3_dict["hidden_size"]),
        "midlayer.self_attn.q_proj.weight": torch.randn(
            eagle3_dict["hidden_size"],
            2 * eagle3_dict["hidden_size"],
        ),
        "midlayer.self_attn.k_proj.weight": torch.randn(
            eagle3_dict["num_key_value_heads"]
            * (eagle3_dict["hidden_size"] // eagle3_dict["num_attention_heads"]),
            2 * eagle3_dict["hidden_size"],
        ),
        "midlayer.self_attn.v_proj.weight": torch.randn(
            eagle3_dict["num_key_value_heads"]
            * (eagle3_dict["hidden_size"] // eagle3_dict["num_attention_heads"]),
            2 * eagle3_dict["hidden_size"],
        ),
        "midlayer.self_attn.o_proj.weight": torch.randn(
            eagle3_dict["hidden_size"], eagle3_dict["hidden_size"]
        ),
        "midlayer.post_attention_layernorm.weight": torch.randn(
            eagle3_dict["hidden_size"]
        ),
        "midlayer.mlp.gate_proj.weight": torch.randn(
            eagle3_dict["intermediate_size"], eagle3_dict["hidden_size"]
        ),
        "midlayer.mlp.up_proj.weight": torch.randn(
            eagle3_dict["intermediate_size"], eagle3_dict["hidden_size"]
        ),
        "midlayer.mlp.down_proj.weight": torch.randn(
            eagle3_dict["hidden_size"], eagle3_dict["intermediate_size"]
        ),
        "norm.weight": torch.randn(eagle3_dict["hidden_size"]),
        "lm_head.weight": torch.randn(
            eagle3_dict["draft_vocab_size"], eagle3_dict["hidden_size"]
        ),
    }


@pytest.fixture
def mock_safeai_eagle3(
    mock_pretrained_factory: MockPretrainedTransformersFactory,
) -> tuple[PretrainedBundle, PretrainedBundle, MockPretrainedTransformersFactory]:
    # Register the verifier with default llama 2m settings
    verifier = mock_pretrained_factory.register()

    # Register the speculator
    name_or_path = "safeai/eagle3-llama-2m"
    speculator = mock_pretrained_factory.register(
        aliases=["eagle3-llama-2m", "eagle3"],
        config_dict=mock_eagle3_config_dict(
            verifier.config.to_dict(), name_or_path=name_or_path
        ),
        state_dict=mock_eagle3_state_dict(verifier.config.to_dict()),
    )

    return speculator, verifier, mock_pretrained_factory
