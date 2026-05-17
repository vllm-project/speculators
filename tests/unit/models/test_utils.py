from types import SimpleNamespace

import pytest

from speculators.models import utils


def test_resolve_target_layer_ids_keeps_aux_layers_only(monkeypatch):
    monkeypatch.setattr(
        utils,
        "get_verifier_config",
        lambda _name_or_path: SimpleNamespace(num_hidden_layers=36),
    )

    with pytest.warns(UserWarning, match="Stripping the verifier's final layer"):
        layer_ids = utils.resolve_target_layer_ids(
            [2, 18, 33, 36], "unused-verifier-path"
        )

    assert layer_ids == [2, 18, 33]


def test_resolve_target_layer_ids_preserves_custom_aux_layers(monkeypatch):
    monkeypatch.setattr(
        utils,
        "get_verifier_config",
        lambda _name_or_path: SimpleNamespace(num_hidden_layers=36),
    )

    assert utils.resolve_target_layer_ids([2, 18, 33], "unused-verifier-path") == [
        2,
        18,
        33,
    ]
