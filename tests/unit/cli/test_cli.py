"""Smoke tests for the speculators CLI."""

from typer.testing import CliRunner

from speculators.cli import app

runner = CliRunner()


class TestRootApp:
    def test_no_args_shows_help(self):
        result = runner.invoke(app, [])
        assert "Usage" in result.output

    def test_help(self):
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        assert "Tools" in result.output

    def test_version(self):
        result = runner.invoke(app, ["--version"])
        assert result.exit_code == 0
        assert "speculators version:" in result.output

    def test_pipeline_commands_in_help(self):
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        assert "regenerate-responses" in result.output

    def test_tools_commands_in_help(self):
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        assert "convert" in result.output


class TestConvertCommand:
    def test_help(self):
        result = runner.invoke(app, ["convert", "--help"])
        assert result.exit_code == 0
        assert "--verifier" in result.output
        assert "--algorithm" in result.output

    def test_algorithm_choices_in_help(self):
        result = runner.invoke(app, ["convert", "--help"])
        assert result.exit_code == 0
        for algo in ("eagle3", "mtp", "dflash"):
            assert algo in result.output

    def test_missing_required_args(self):
        result = runner.invoke(app, ["convert"])
        assert result.exit_code != 0


class TestRegenerateResponsesCommand:
    def test_help(self):
        result = runner.invoke(app, ["regenerate-responses", "--help"])
        assert result.exit_code == 0
        assert "--endpoint" in result.output
        assert "--dataset" in result.output
        assert "--concurrency" in result.output
        assert "--max-tokens" in result.output

    def test_invalid_max_retries(self):
        result = runner.invoke(app, ["regenerate-responses", "--max-retries", "-1"])
        assert result.exit_code != 0

    def test_invalid_sampling_params(self):
        result = runner.invoke(
            app, ["regenerate-responses", "--sampling-params", "not-json"]
        )
        assert result.exit_code != 0

    def test_sampling_params_must_be_object(self):
        result = runner.invoke(
            app, ["regenerate-responses", "--sampling-params", "[1,2,3]"]
        )
        assert result.exit_code != 0
