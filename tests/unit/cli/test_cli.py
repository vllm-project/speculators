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
        assert "prepare-data" in result.output

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


class TestPrepareDataCommand:
    def test_help(self):
        result = runner.invoke(app, ["prepare-data", "--help"])
        assert result.exit_code == 0
        assert "--model" in result.output
        assert "--data" in result.output
        assert "--output" in result.output
        assert "--seq-length" in result.output

    def test_missing_required_args(self):
        result = runner.invoke(app, ["prepare-data"])
        assert result.exit_code != 0

    def test_allow_empty_output_in_help(self):
        result = runner.invoke(app, ["prepare-data", "--help"])
        assert result.exit_code == 0
        assert "--allow-empty-output" in result.output

    def test_overwrite_in_help(self):
        result = runner.invoke(app, ["prepare-data", "--help"])
        assert result.exit_code == 0
        assert "--overwrite" in result.output

    def test_render_endpoint_in_help(self):
        result = runner.invoke(app, ["prepare-data", "--help"])
        assert result.exit_code == 0
        assert "--render-endpoint" in result.output
