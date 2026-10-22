"""Unit tests for target layer id resolution."""

from types import SimpleNamespace

import pytest

from speculators.models import utils as model_utils
from speculators.models.utils import resolve_target_layer_ids


def _verifier_with(num_layers, monkeypatch):
    monkeypatch.setattr(
        model_utils,
        "get_verifier_config",
        lambda *_a, **_k: SimpleNamespace(num_hidden_layers=num_layers),
    )


@pytest.mark.parametrize(
    ("num_layers", "expected"), [(3, [2, 1, 0]), (7, [2, 3, 4]), (28, [2, 14, 25])]
)
def test_default_ids_when_distinct(num_layers, expected, monkeypatch):
    _verifier_with(num_layers, monkeypatch)
    with pytest.warns(UserWarning, match="not explicitly set"):
        assert resolve_target_layer_ids(None, "verifier") == expected


@pytest.mark.parametrize("num_layers", [0, 1, 2, 4, 5, 6])
def test_default_ids_fail_when_negative_or_repeated(num_layers, monkeypatch):
    _verifier_with(num_layers, monkeypatch)
    with pytest.raises(ValueError, match="invalid for a verifier"):
        resolve_target_layer_ids(None, "verifier")


def test_explicit_ids_pass_through(monkeypatch):
    _verifier_with(28, monkeypatch)
    assert resolve_target_layer_ids([1, 9, 17], "verifier") == [1, 9, 17]


def test_explicit_last_layer_id_is_valid(monkeypatch):
    """Id ``num_hidden_layers`` is the final hidden state (--include-last-layer)."""
    _verifier_with(28, monkeypatch)
    assert resolve_target_layer_ids([2, 14, 28], "verifier") == [2, 14, 28]


def test_explicit_duplicate_ids_are_rejected(monkeypatch):
    _verifier_with(28, monkeypatch)
    with pytest.raises(ValueError, match="distinct"):
        resolve_target_layer_ids([2, 2, 1], "verifier")


@pytest.mark.parametrize("ids", [[-1, 2, 3], [2, 14, 29], []])
def test_explicit_out_of_bounds_ids_are_rejected(ids, monkeypatch):
    _verifier_with(28, monkeypatch)
    with pytest.raises(ValueError, match="distinct and within"):
        resolve_target_layer_ids(ids, "verifier")
