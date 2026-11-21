"""Shared fixtures for model unit tests."""

import importlib

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


@pytest.fixture
def narrowest_tile(monkeypatch):
    """Pin every autotuned fused-loss kernel to its narrowest tile.

    A cold autotune sweep costs minutes per vocab and term set; correctness
    tests need one tile, and a vocab wider than it still spans several.
    """
    try:
        fused = importlib.import_module("speculators.losses.fused")
    except ImportError:
        return
    autotuner = importlib.import_module("triton.runtime.autotuner").Autotuner
    for kernel in vars(fused).values():
        if isinstance(kernel, autotuner):
            floor = min(c.kwargs["BLOCK_SIZE"] for c in kernel.configs)
            narrow = [c for c in kernel.configs if c.kwargs["BLOCK_SIZE"] == floor]
            monkeypatch.setattr(kernel, "configs", narrow[:1])
