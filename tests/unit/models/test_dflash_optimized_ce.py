"""Focused tests for DFlash hard-label CE preparation and target selection."""

import pytest
import torch
from transformers import Qwen3Config

from speculators.config import SpeculatorsConfig, VerifierConfig
from speculators.models.dflash import DFlashSpeculatorConfig
from speculators.models.dflash.core import DFlashDraftModel
from speculators.proposals.greedy import GreedyTokenProposalConfig


def _model(
    *, sample_from_anchor: bool = False, draft_vocab_size: int = 17
) -> DFlashDraftModel:
    transformer_config = Qwen3Config(
        vocab_size=17,
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=4,
        max_position_embeddings=32,
    )
    config = DFlashSpeculatorConfig(
        transformer_layer_config=transformer_config,
        draft_vocab_size=draft_vocab_size,
        block_size=4,
        aux_hidden_state_layer_ids=[0],
        mask_token_id=1,
        sample_from_anchor=sample_from_anchor,
        speculators_config=SpeculatorsConfig(
            algorithm="dflash",
            proposal_methods=[GreedyTokenProposalConfig(speculative_tokens=3)],
            default_proposal_method="greedy",
            verifier=VerifierConfig(name_or_path=None, architectures=[]),
        ),
    )
    model = DFlashDraftModel(config)
    with torch.no_grad():
        head = torch.randn_like(model.lm_head.weight)
        model.lm_head.weight.copy_(head)
        model.verifier_lm_head.weight.copy_(head)
        model.verifier_norm.weight.fill_(1.0)
    return model


def test_chunked_verifier_argmax_matches_materialized_argmax():
    torch.manual_seed(11)
    model = _model()
    hidden = torch.randn(1, 11, model.hidden_size)
    input_ids = torch.randint(0, model.draft_vocab_size, (1, 11))
    indices = torch.tensor([1, 2, 5, 7, 8])

    materialized = model._ce_target_ids(
        input_ids,
        hidden,
        indices,
        label_source="verifier_argmax",
        verifier_argmax_chunk_size=0,
    )
    chunked = model._ce_target_ids(
        input_ids,
        hidden,
        indices,
        label_source="verifier_argmax",
        verifier_argmax_chunk_size=3,
    )

    assert torch.equal(chunked, materialized)


def test_input_id_labels_are_explicit_and_position_aligned():
    model = _model()
    input_ids = torch.arange(11).unsqueeze(0)
    hidden = torch.randn(1, 11, model.hidden_size)
    indices = torch.tensor([1, 2, 5, 7, 8])

    labels = model._ce_target_ids(
        input_ids,
        hidden,
        indices,
        label_source="input_ids",
        verifier_argmax_chunk_size=0,
    )

    assert torch.equal(labels, input_ids[:, indices])


def test_sample_from_anchor_input_labels_select_the_next_token():
    model = _model(sample_from_anchor=True)
    input_ids = torch.arange(11).unsqueeze(0)
    hidden = torch.randn(1, 11, model.hidden_size)
    indices = torch.tensor([1, 2, 5, 7, 8])

    labels = model._ce_target_ids(
        input_ids,
        hidden,
        indices,
        label_source="input_ids",
        verifier_argmax_chunk_size=0,
    )

    assert torch.equal(labels, input_ids[:, indices + 1])


def test_reduced_draft_vocab_rejects_input_id_labels():
    model = _model(draft_vocab_size=5)
    input_ids = torch.tensor([[0, 16]])
    hidden = torch.randn(1, 2, model.hidden_size)

    with pytest.raises(ValueError, match="requires the full verifier vocabulary"):
        model._ce_target_ids(
            input_ids,
            hidden,
            torch.tensor([0, 1]),
            label_source="input_ids",
            verifier_argmax_chunk_size=0,
        )


def test_reduced_vocab_verifier_argmax_returns_direct_draft_boundary_ids():
    model = _model(sample_from_anchor=True, draft_vocab_size=5)
    model.verifier_norm = torch.nn.Identity()
    with torch.no_grad():
        model.verifier_lm_head.weight.zero_()
        model.verifier_lm_head.weight[0, 0] = 1
        model.verifier_lm_head.weight[-1, 1] = 1
    hidden = torch.zeros(1, 2, model.hidden_size)
    hidden[0, 0, 0] = 2
    hidden[0, 1, 1] = 2

    labels = model._ce_target_ids(
        torch.tensor([[0, 16]]),
        hidden,
        torch.tensor([0, 1]),
        label_source="verifier_argmax",
        verifier_argmax_chunk_size=1,
    )

    assert torch.equal(labels, torch.tensor([[0, model.draft_vocab_size - 1]]))


def test_fused_ce_preparation_reuses_ignored_head_at_compute_dtype():
    model = _model()
    model.prepare_fused_linear_cross_entropy(torch.bfloat16)

    assert model.lm_head.weight.dtype == torch.float32
    assert model.verifier_lm_head.weight.dtype == torch.bfloat16


def test_fused_ce_preparation_rejects_mismatched_heads():
    model = _model()
    with torch.no_grad():
        model.verifier_lm_head.weight[0, 0].add_(1)

    with pytest.raises(RuntimeError, match="LM heads"):
        model.prepare_fused_linear_cross_entropy(torch.bfloat16)
