"""Native ASR checkpoints use the existing multimodal decoder/weight loader."""

import pytest
import torch
from safetensors.torch import save_file
from transformers import Qwen3Config

from speculators import SpeculatorsConfig, VerifierConfig
from speculators.models.eagle3 import Eagle3DraftModel, Eagle3SpeculatorConfig
from speculators.proposals.greedy import GreedyTokenProposalConfig
from speculators.train.cli import create_transformer_layer_config


def test_asr_decoder_config_and_frozen_tied_weights(tmp_path):
    native = pytest.importorskip(
        "transformers.models.qwen3_asr.configuration_qwen3_asr"
    )
    text = Qwen3Config(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=28,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        max_position_embeddings=1024,
        tie_word_embeddings=True,
    )
    verifier = native.Qwen3ASRConfig(text_config=text.to_dict())
    verifier.save_pretrained(tmp_path)
    embeddings = torch.arange(64 * 32, dtype=torch.float32).reshape(64, 32)
    norm = torch.full((32,), 0.5)
    weights = {
        "model.language_model.embed_tokens.weight": embeddings,
        "model.language_model.norm.weight": norm,
        "model.audio_tower.layers.0.final_layer_norm.weight": torch.ones(128),
    }
    save_file(weights, str(tmp_path / "model.safetensors"))

    decoder = create_transformer_layer_config(
        str(tmp_path),
        num_layers=1,
        draft_arch="qwen3",
        hidden_act=None,
        sliding_window=0,
        full_attention_indices=[0],
    )
    assert decoder.hidden_size == text.hidden_size
    assert decoder.num_attention_heads == text.num_attention_heads
    assert decoder.vocab_size == text.vocab_size
    decoder._attn_implementation = "eager"
    draft_vocab_size = 32
    model = Eagle3DraftModel(
        Eagle3SpeculatorConfig(
            transformer_layer_config=decoder,
            draft_vocab_size=draft_vocab_size,
            eagle_aux_hidden_state_layer_ids=[2, 14, 25],
            speculators_config=SpeculatorsConfig(
                algorithm="eagle3",
                proposal_methods=[GreedyTokenProposalConfig(speculative_tokens=3)],
                default_proposal_method="greedy",
                verifier=VerifierConfig(
                    name_or_path=str(tmp_path),
                    architectures=["Qwen3ASRForConditionalGeneration"],
                ),
            ),
        )
    )
    t2d = torch.arange(text.vocab_size) < draft_vocab_size
    model.load_vocab_mappings(t2d, torch.arange(draft_vocab_size))
    model.load_verifier_weights()
    torch.testing.assert_close(model.embed_tokens.weight, embeddings)
    torch.testing.assert_close(
        model.verifier_lm_head.weight, embeddings[:draft_vocab_size]
    )
    torch.testing.assert_close(model.verifier_norm.weight, norm)
    assert not model.embed_tokens.weight.requires_grad
    assert not model.verifier_lm_head.weight.requires_grad
    assert not model.verifier_norm.weight.requires_grad
    assert model.fc.weight.requires_grad
    assert all("audio_tower" not in name for name, _ in model.named_parameters())
