"""Eagle3 verifier weight loading against local verifier checkpoints."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest
import torch
from safetensors.torch import save_file
from transformers.models.llama.configuration_llama import LlamaConfig

from speculators import SpeculatorsConfig, VerifierConfig
from speculators.models.eagle3 import Eagle3DraftModel, Eagle3SpeculatorConfig
from speculators.proposals.greedy import GreedyTokenProposalConfig

if TYPE_CHECKING:
    from pathlib import Path

VOCAB = 64
HIDDEN = 32


def _make_model(verifier_path: Path, hidden_size: int = HIDDEN) -> Eagle3DraftModel:
    config = Eagle3SpeculatorConfig(
        transformer_layer_config=LlamaConfig(
            vocab_size=VOCAB,
            hidden_size=hidden_size,
            intermediate_size=64,
            num_hidden_layers=1,
            num_attention_heads=4,
            num_key_value_heads=4,
            head_dim=hidden_size // 4,
            _attn_implementation="eager",  # type: ignore[call-arg]
        ),
        draft_vocab_size=VOCAB,
        norm_before_residual=False,
        speculators_config=SpeculatorsConfig(
            algorithm="eagle3",
            proposal_methods=[GreedyTokenProposalConfig(speculative_tokens=1)],
            default_proposal_method="greedy",
            verifier=VerifierConfig(
                name_or_path=str(verifier_path), architectures=["TinyOmniModel"]
            ),
        ),
    )
    return Eagle3DraftModel(config)


def _write_remote_code_verifier(path: Path) -> dict[str, torch.Tensor]:
    """Verifier whose config loads only via remote code and nests the text
    model under ``llm_config``, as in Nemotron Nano Omni."""
    path.mkdir()
    (path / "config.json").write_text(
        json.dumps(
            {
                "model_type": "tiny_omni",
                "architectures": ["TinyOmniModel"],
                "auto_map": {"AutoConfig": "configuration_tiny_omni.TinyOmniConfig"},
                "llm_config": {"hidden_size": HIDDEN, "vocab_size": VOCAB},
            }
        )
    )
    weights = {
        "language_model.backbone.embeddings.weight": torch.randn(VOCAB, HIDDEN),
        "language_model.lm_head.weight": torch.randn(VOCAB, HIDDEN),
        "language_model.backbone.norm_f.weight": torch.ones(HIDDEN),
    }
    save_file(weights, path / "model.safetensors")
    return weights


def test_remote_code_verifier_loads_without_its_config(tmp_path: Path):
    verifier = tmp_path / "verifier"
    weights = _write_remote_code_verifier(verifier)
    model = _make_model(verifier)

    model.load_verifier_weights()

    assert torch.equal(
        model.embed_tokens.weight, weights["language_model.backbone.embeddings.weight"]
    )
    assert torch.equal(
        model.verifier_lm_head.weight, weights["language_model.lm_head.weight"]
    )


def test_hidden_size_mismatch_raises(tmp_path: Path):
    verifier = tmp_path / "verifier"
    _write_remote_code_verifier(verifier)
    model = _make_model(verifier, hidden_size=HIDDEN * 2)

    with pytest.raises(ValueError, match="does not match draft hidden size"):
        model.load_verifier_weights()
