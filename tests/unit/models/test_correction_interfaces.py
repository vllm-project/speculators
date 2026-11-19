"""Correction operations accept generated context without training targets."""

import pytest
import torch

from tests.unit.models.test_dflash_family_contracts import RECIPES, make_contract_model


@pytest.mark.parametrize("recipe", RECIPES)
def test_correction_has_one_owner_and_a_target_free_interface(recipe):
    model = make_contract_model(recipe, False)
    assert (
        not {"markov_head", "candidate_selector", "refiner_head"}
        & model._modules.keys()
    )
    features = model.draft_features(
        torch.randn(1, 12, 32),
        torch.arange(12).view(1, -1),
        torch.ones(1, 12),
        torch.zeros(1, 12, dtype=torch.long),
        max_anchors=2,
    )
    predecessors = features.layout.predecessor_ids(torch.arange(12).view(1, -1))
    if recipe == "dflash2":
        candidates = features.logits.topk(4, dim=-1).indices
        actual = model.score_candidates(features, predecessors, candidates)
        assert actual.candidate_ids is candidates
        assert actual.scores.shape == candidates.shape
        assert torch.isfinite(actual.scores).all()
        with pytest.raises(TypeError, match="candidate scoring"):
            model.correct(features, predecessors)
    else:
        actual = model.correct(features, predecessors)
        assert actual.logits.shape == features.logits.shape
        assert torch.isfinite(actual.logits).all()
        if recipe == "dflash":
            assert actual.logits is features.logits
        if recipe == "dspark":
            assert actual.confidence_features is not None
            assert actual.confidence_features.shape == (2, 4, 4)
