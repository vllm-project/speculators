"""Unit tests: the CLI convert algorithm choices match the backend."""

from typer.testing import CliRunner

import speculators.__main__ as cli
from speculators.convert import SUPPORTED_ALGORITHMS

runner = CliRunner()


def test_cli_offers_exactly_the_backend_algorithms(monkeypatch):
    calls = []
    monkeypatch.setattr(
        cli, "convert_model", lambda **kwargs: calls.append(kwargs["algorithm"])
    )
    for algorithm in SUPPORTED_ALGORITHMS:
        result = runner.invoke(
            cli.app,
            ["convert", "model", "--verifier", "v", "--algorithm", algorithm],
        )
        assert result.exit_code == 0, result.output
    assert calls == list(SUPPORTED_ALGORITHMS)


def test_cli_rejects_unsupported_eagle_v1(monkeypatch):
    monkeypatch.setattr(
        cli,
        "convert_model",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError("must not be called")),
    )
    result = runner.invoke(
        cli.app,
        ["convert", "model", "--verifier", "v", "--algorithm", "eagle"],
    )
    # Rejected by the CLI parser instead of exploding inside convert_model.
    assert result.exit_code != 0
