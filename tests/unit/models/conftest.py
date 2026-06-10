"""Shared fixtures for model unit tests."""

import pytest
from torch import nn

from speculators import SpeculatorsConfig, VerifierConfig
from speculators.models.mtp import MTPDraftModel, MTPSpeculatorConfig
from speculators.proposals import GreedyTokenProposalConfig


@pytest.fixture
def mtp_model_config(qwen3_5_pretrained_config):
    return MTPSpeculatorConfig(
        transformer_layer_config=qwen3_5_pretrained_config,
        speculators_config=SpeculatorsConfig(
            algorithm="mtp",
            proposal_methods=[
                GreedyTokenProposalConfig(speculative_tokens=3),
            ],
            default_proposal_method="greedy",
            verifier=VerifierConfig(
                name_or_path=None,
                architectures=["Qwen3_5ForCausalLM"],
            ),
        ),
    )


@pytest.fixture
def mtp_model(mtp_model_config):
    model = MTPDraftModel(mtp_model_config)
    nn.init.normal_(model.embed_tokens.weight, std=0.02)
    nn.init.normal_(model.lm_head.weight, std=0.02)
    model.eval()
    return model
