"""Help and argument-parsing tests for ``generate-agentic-traces``.

The command body imports the agentic runtime lazily, so these tests exercise the
CLI surface only -- no docker, network, or model downloads.
"""

import click
from typer.testing import CliRunner

from speculators.cli import app

runner = CliRunner(env={"COLUMNS": "200"})


def unstyled_output(result):
    return click.unstyle(result.output)


class TestGenerateAgenticTracesCommand:
    def test_help(self):
        result = runner.invoke(app, ["generate-agentic-traces", "--help"])
        assert result.exit_code == 0
        output = unstyled_output(result)
        for flag in (
            "--model",
            "--data",
            "--output",
            "--render-endpoint",
            "--pi-runtime",
            "--pi-endpoint",
            "--pi-provider",
            "--pi-model",
            "--workers",
            "--timeout",
            "--seq-length",
            "--limit",
            "--max-samples",
            "--tools",
            "--trace-dir",
            "--keep-traces",
        ):
            assert flag in output

    def test_missing_required_args(self):
        result = runner.invoke(app, ["generate-agentic-traces"])
        assert result.exit_code != 0
        assert "Missing option" in unstyled_output(result)

    def test_missing_single_required_arg(self):
        result = runner.invoke(
            app,
            [
                "generate-agentic-traces",
                "--model",
                "some/model",
                "--data",
                "tasks.jsonl",
                "--output",
                "out.jsonl",
                "--render-endpoint",
                "http://localhost:8000",
            ],
        )
        assert result.exit_code != 0
        assert "--pi-runtime" in unstyled_output(result)
