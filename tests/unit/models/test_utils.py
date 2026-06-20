"""Unit tests for speculators.models.utils config-resolution helpers."""

from types import SimpleNamespace
from typing import cast

import pytest
from transformers import PretrainedConfig

from speculators.models import utils as model_utils
from speculators.models.utils import resolve_draft_intermediate_size


def _fake_verifier(**fields) -> PretrainedConfig:
    """Lightweight stand-in verifier config (the resolver only reads attributes)."""
    return cast("PretrainedConfig", SimpleNamespace(**fields))


# ---------------------------------------------------------------------------
# target layer resolution
# ---------------------------------------------------------------------------


@pytest.mark.regression
def test_resolve_target_layer_ids_preserves_explicit_checkpoint_layers(monkeypatch):
    """Explicit checkpoint layer IDs should load without verifier config access."""

    def raise_if_called(_name_or_path):
        raise AssertionError(
            "explicit checkpoint layers should not load verifier config"
        )

    monkeypatch.setattr(
        model_utils,
        "get_verifier_config",
        raise_if_called,
    )

    layer_ids = model_utils.resolve_target_layer_ids(
        [2, 18, 33, 36], "unused-verifier-path"
    )

    assert layer_ids == [2, 18, 33, 36]


@pytest.mark.regression
def test_strip_verifier_final_layer_id_keeps_aux_layers_only(monkeypatch):
    """Training config keeps final verifier hidden state out of aux layers."""
    monkeypatch.setattr(
        model_utils,
        "get_verifier_config",
        lambda _name_or_path: SimpleNamespace(num_hidden_layers=36),
    )

    with pytest.warns(UserWarning, match="Stripping the verifier's final layer"):
        layer_ids = model_utils.strip_verifier_final_layer_id(
            [2, 18, 33, 36], "unused-verifier-path"
        )

    assert layer_ids == [2, 18, 33]


# ---------------------------------------------------------------------------
# resolve_draft_intermediate_size
# ---------------------------------------------------------------------------


@pytest.mark.smoke
def test_resolve_uses_dense_intermediate_size_directly():
    # A dense verifier's intermediate_size is mirrored verbatim, even when a
    # hidden_size is also present (dense takes precedence over the 3x fallback).
    verifier = _fake_verifier(intermediate_size=11008, hidden_size=4096)

    assert resolve_draft_intermediate_size(verifier) == 11008


@pytest.mark.smoke
def test_resolve_moe_falls_back_to_3x_hidden_size():
    # MoE verifier: no dense intermediate_size -> draft uses 3 * hidden_size.
    verifier = _fake_verifier(hidden_size=2048)

    with pytest.warns(UserWarning, match="3 x hidden_size"):
        assert resolve_draft_intermediate_size(verifier) == 6144


@pytest.mark.smoke
def test_resolve_ignores_moe_expert_fields():
    # Expert fields are irrelevant now: with no dense intermediate_size the draft
    # width is purely 3 * hidden_size regardless of the MoE routing config.
    verifier = _fake_verifier(
        hidden_size=1024,
        moe_intermediate_size=768,
        num_experts_per_tok=8,
        num_experts=128,
        shared_expert_intermediate_size=2048,
    )

    with pytest.warns(UserWarning, match="3 x hidden_size"):
        assert resolve_draft_intermediate_size(verifier) == 3072


@pytest.mark.smoke
def test_resolve_requires_intermediate_or_hidden_size():
    # Degenerate config with neither field -> explicit error pointing at --draft-config.
    verifier = _fake_verifier()

    with pytest.raises(ValueError, match="--draft-config"):
        resolve_draft_intermediate_size(verifier)
