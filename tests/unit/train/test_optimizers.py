"""Tests for optimizer parameter partitioning."""

import torch
from torch import nn

from speculators.train.optimizers import split_named_params_for_muon


def test_generic_embeddings_heads_and_codebooks_use_base_adamw():
    model = nn.Module()
    model.token_lookup = nn.Embedding(32, 8)
    model.lm_head = nn.Linear(8, 32, bias=False)
    model.aux_codebook = nn.Parameter(torch.empty(32, 8))
    model.projection = nn.Linear(8, 16, bias=False)

    muon, adamw, transition = split_named_params_for_muon(model)

    assert {name for name, _ in muon} == {"projection.weight"}
    assert {name for name, _ in adamw} == {
        "token_lookup.weight",
        "lm_head.weight",
        "aux_codebook",
    }
    assert not transition
