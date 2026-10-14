"""Unit tests: the CLI convert algorithm choices match the backend."""

import importlib

from typer.testing import CliRunner

from speculators.cli import app
from speculators.convert import SUPPORTED_ALGORITHMS

# speculators.cli re-exports the ``convert`` command function, which shadows the
# submodule of the same name on attribute access, so fetch the module directly.
cli_convert = importlib.import_module("speculators.cli.convert")
runner = CliRunner()


def test_cli_offers_exactly_the_backend_algorithms(monkeypatch):
    calls = []
    monkeypatch.setattr(
        cli_convert, "convert_model", lambda **kwargs: calls.append(kwargs["algorithm"])
    )
    for algorithm in SUPPORTED_ALGORITHMS:
        result = runner.invoke(
            app,
            ["convert", "model", "--verifier", "v", "--algorithm", algorithm],
        )
        assert result.exit_code == 0, result.output
    assert calls == list(SUPPORTED_ALGORITHMS)


def test_cli_rejects_unsupported_eagle_v1(monkeypatch):
    calls = []
    monkeypatch.setattr(
        cli_convert, "convert_model", lambda **kwargs: calls.append(kwargs["algorithm"])
    )
    result = runner.invoke(
        app,
        ["convert", "model", "--verifier", "v", "--algorithm", "eagle"],
    )
    # Rejected by the CLI parser instead of exploding inside convert_model.
    assert result.exit_code != 0
    assert calls == []
