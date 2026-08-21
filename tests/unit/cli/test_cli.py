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
        assert "train" in result.output

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


class TestTrainCommand:
    def test_help(self):
        result = runner.invoke(app, ["train", "--help"])
        assert result.exit_code == 0
        assert "--verifier-name-or-path" in result.output
        assert "--config" in result.output
        assert "--speculator-type" in result.output

    def test_train_appears_in_pipeline_panel(self):
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        assert "train" in result.output
