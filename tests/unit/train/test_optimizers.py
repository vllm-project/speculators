"""Tests for optimizer parameter partitioning."""

from torch import nn

from speculators.train.optimizers import split_named_params_for_muon


def test_embeddings_route_to_adamw_regardless_of_name():
    # Muon orthogonalizes a matrix as a linear map between feature spaces; an embedding
    # is a lookup table indexed by token id, so the update is meaningless for it. The
    # probe mirrors DSpark's Markov head, whose embedding no name hint would catch.
    model = nn.Module()
    model.markov_head = nn.Module()
    model.markov_head.markov_w1 = nn.Embedding(512, 16)
    model.proj = nn.Linear(16, 32, bias=False)

    muon, adamw = split_named_params_for_muon(model)

    assert "markov_head.markov_w1.weight" in {n for n, _ in adamw}
    assert "markov_head.markov_w1.weight" not in {n for n, _ in muon}
    assert "proj.weight" in {n for n, _ in muon}  # ordinary matrices still reach Muon
