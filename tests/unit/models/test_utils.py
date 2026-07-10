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
def test_get_verifier_config_falls_back_when_text_config_is_none(monkeypatch):
    """A top-level text config remains usable when the nested field is null."""
    verifier_config = SimpleNamespace(text_config=None, num_hidden_layers=36)
    monkeypatch.setattr(
        model_utils.AutoConfig,
        "from_pretrained",
        lambda _name_or_path: verifier_config,
    )

    assert model_utils.get_verifier_config("unused-verifier-path") is verifier_config


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


@pytest.mark.regression
def test_strip_verifier_final_layer_id_allows_variable_aux_count(monkeypatch):
    """Removing the final layer does not impose a fixed three-aux-layer count."""
    monkeypatch.setattr(
        model_utils,
        "get_verifier_config",
        lambda _name_or_path: SimpleNamespace(num_hidden_layers=36),
    )

    with pytest.warns(UserWarning, match="Stripping the verifier's final layer"):
        layer_ids = model_utils.strip_verifier_final_layer_id(
            [2, 18, 36], "unused-verifier-path"
        )

    assert layer_ids == [2, 18]


@pytest.mark.regression
def test_strip_verifier_final_layer_id_rejects_empty_aux_layers(monkeypatch):
    """An explicit final-layer-only list must not silently restore defaults."""
    monkeypatch.setattr(
        model_utils,
        "get_verifier_config",
        lambda _name_or_path: SimpleNamespace(num_hidden_layers=36),
    )

    with pytest.raises(ValueError, match="at least one auxiliary verifier layer"):
        model_utils.strip_verifier_final_layer_id(
            [36], "unused-verifier-path"
        )


@pytest.mark.parametrize("invalid_layer_id", [0, -1, 37])
def test_strip_verifier_final_layer_id_rejects_out_of_range_training_layers(
    monkeypatch,
    invalid_layer_id,
):
    """Non-MTP training must reject layer IDs the verifier cannot produce."""
    monkeypatch.setattr(
        model_utils,
        "get_verifier_config",
        lambda _name_or_path: SimpleNamespace(num_hidden_layers=36),
    )

    with pytest.raises(ValueError, match="inclusive range"):
        model_utils.strip_verifier_final_layer_id(
            [2, invalid_layer_id], "unused-verifier-path"
        )


def test_strip_verifier_final_layer_id_rejects_duplicate_training_layers(monkeypatch):
    monkeypatch.setattr(
        model_utils,
        "get_verifier_config",
        lambda _name_or_path: SimpleNamespace(num_hidden_layers=36),
    )

    with pytest.raises(ValueError, match="must not contain duplicate"):
        model_utils.strip_verifier_final_layer_id(
            [2, 2, 36], "unused-verifier-path"
        )


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


@pytest.mark.parametrize(
    ("num_layers", "expected"),
    [
        (2, [1]),
        (3, [1, 2]),
        (4, [1, 2, 3]),
        (5, [1, 2, 4]),
        (36, [2, 18, 33]),
    ],
)
def test_default_auxiliary_target_layers_are_unique_and_valid(num_layers, expected):
    assert model_utils.default_auxiliary_target_layer_ids(num_layers) == expected


def test_default_auxiliary_target_layers_reject_single_layer_verifier():
    with pytest.raises(ValueError, match="at least two hidden layers"):
        model_utils.default_auxiliary_target_layer_ids(1)
