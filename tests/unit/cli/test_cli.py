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
        assert "Pipeline" in result.output
        assert "Tools" in result.output

    def test_version(self):
        result = runner.invoke(app, ["--version"])
        assert result.exit_code == 0
        assert "speculators version:" in result.output

    def test_pipeline_commands_in_help(self):
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        assert "prepare-data" in result.output
        assert "generate-data" in result.output
        assert "stitch" in result.output

    def test_pipeline_train_in_help(self):
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        assert "train" in result.output

    def test_pipeline_regenerate_responses_in_help(self):
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

    def test_missing_required_args(self):
        result = runner.invoke(app, ["convert"])
        assert result.exit_code != 0


class TestStitchCommand:
    def test_help(self):
        result = runner.invoke(app, ["stitch", "--help"])
        assert result.exit_code == 0
        assert "finetuned_checkpoint" in result.output
        assert "verifier_path" in result.output

    def test_missing_required_args(self):
        result = runner.invoke(app, ["stitch"])
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


class TestTrainCommand:
    def test_help(self):
        result = runner.invoke(app, ["train", "--help"])
        assert result.exit_code == 0
        assert "Train a speculator model" in result.output

    def test_train_appears_in_pipeline_panel(self):
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        assert "train" in result.output


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


class TestGenerateDataCommand:
    def test_help(self):
        result = runner.invoke(app, ["generate-data", "--help"])
        assert result.exit_code == 0
        assert "--endpoint" in result.output
        assert "--preprocessed-data" in result.output
        assert "--concurrency" in result.output

    def test_invalid_rank(self):
        result = runner.invoke(
            app, ["generate-data", "--rank", "5", "--world-size", "2"]
        )
        assert result.exit_code != 0
