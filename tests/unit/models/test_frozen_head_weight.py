"""Tests for DraftVocabMixin._frozen_head_weight autocast cast caching."""

import pytest
import torch
from torch import nn

from speculators.model import DraftVocabMixin

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="autocast dtype caching tested on CUDA"
)


class _Heads(DraftVocabMixin):
    def __init__(self):
        super().__init__()
        self.lm_head = nn.Linear(16, 32, bias=False).cuda().requires_grad_(False)


def test_returns_weight_outside_autocast():
    model = _Heads()
    assert model._frozen_head_weight(model.lm_head) is model.lm_head.weight


def test_trainable_weight_is_not_cached():
    model = _Heads()
    model.lm_head.weight.requires_grad_(True)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        assert model._frozen_head_weight(model.lm_head) is model.lm_head.weight


def test_frozen_weight_cast_matches_autocast_and_is_reused():
    model = _Heads()
    x = torch.randn(4, 16, device="cuda")
    with torch.autocast("cuda", dtype=torch.bfloat16):
        first = model._frozen_head_weight(model.lm_head)
        second = model._frozen_head_weight(model.lm_head)
        expected = model.lm_head(x)
        actual = nn.functional.linear(x, first)
    assert first.dtype == torch.bfloat16
    assert first is second
    assert torch.equal(actual, expected)


def test_in_place_update_invalidates_cache():
    model = _Heads()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        stale = model._frozen_head_weight(model.lm_head)
        with torch.no_grad():
            model.lm_head.weight.add_(1.0)
        fresh = model._frozen_head_weight(model.lm_head)
    assert fresh is not stale
    assert torch.equal(fresh, model.lm_head.weight.to(torch.bfloat16))
