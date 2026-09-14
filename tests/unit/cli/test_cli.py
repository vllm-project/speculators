"""Smoke tests for the speculators CLI."""

import click
from typer.testing import CliRunner

from speculators.cli import app

runner = CliRunner()


def unstyled_output(result):
    """Return CLI output without terminal styling for stable assertions."""
    return click.unstyle(result.output)


class TestRootApp:
    def test_no_args_shows_help(self):
        result = runner.invoke(app, [])
        assert "Usage" in unstyled_output(result)

    def test_help(self):
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        assert "Pipeline" in unstyled_output(result)
        assert "Tools" in unstyled_output(result)

    def test_version(self):
        result = runner.invoke(app, ["--version"])
        assert result.exit_code == 0
        assert "speculators version:" in unstyled_output(result)

    def test_pipeline_commands_in_help(self):
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        output = unstyled_output(result)
        assert "prepare-data" in output
        assert "stitch-mtp" in output
        assert "generate-offline-data" in output
        assert "regenerate-responses" in output
        assert "train" in output

    def test_tools_commands_in_help(self):
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        assert "convert" in unstyled_output(result)


class TestConvertCommand:
    def test_help(self):
        result = runner.invoke(app, ["convert", "--help"])
        assert result.exit_code == 0
        output = unstyled_output(result)
        assert "--verifier" in output
        assert "--algorithm" in output

    def test_algorithm_choices_in_help(self):
        result = runner.invoke(app, ["convert", "--help"])
        assert result.exit_code == 0
        for algo in ("eagle3", "mtp", "dflash"):
            assert algo in unstyled_output(result)

    def test_missing_required_args(self):
        result = runner.invoke(app, ["convert"])
        assert result.exit_code != 0


class TestPrepareDataCommand:
    def test_help(self):
        result = runner.invoke(app, ["prepare-data", "--help"])
        assert result.exit_code == 0
        output = unstyled_output(result)
        assert "--model" in output
        assert "--data" in output
        assert "--output" in output
        assert "--seq-length" in output

    def test_missing_required_args(self):
        result = runner.invoke(app, ["prepare-data"])
        assert result.exit_code != 0

    def test_allow_empty_output_in_help(self):
        result = runner.invoke(app, ["prepare-data", "--help"])
        assert result.exit_code == 0
        assert "--allow-empty-output" in unstyled_output(result)

    def test_overwrite_in_help(self):
        result = runner.invoke(app, ["prepare-data", "--help"])
        assert result.exit_code == 0
        assert "--overwrite" in unstyled_output(result)

    def test_render_endpoint_in_help(self):
        result = runner.invoke(app, ["prepare-data", "--help"])
        assert result.exit_code == 0
        assert "--render-endpoint" in unstyled_output(result)


class TestStitchCommand:
    def test_help(self):
        result = runner.invoke(app, ["stitch-mtp", "--help"])
        assert result.exit_code == 0
        output = unstyled_output(result)
        assert "finetuned_checkpoint" in output
        assert "verifier_path" in output

    def test_missing_required_args(self):
        result = runner.invoke(app, ["stitch-mtp"])
        assert result.exit_code != 0


class TestGenerateOfflineDataCommand:
    def test_help(self):
        result = runner.invoke(app, ["generate-offline-data", "--help"])
        assert result.exit_code == 0
        output = unstyled_output(result)
        assert "--endpoint" in output
        assert "--preprocessed-data" in output
        assert "--concurrency" in output
        assert "--world-size" in output
        assert "--rank" in output

    def test_fail_on_error_in_help(self):
        result = runner.invoke(app, ["generate-offline-data", "--help"])
        assert result.exit_code == 0
        output = unstyled_output(result)
        assert "--fail-on-error" in output
        assert "--max-retries" in output
        assert "--validate-outputs" in output

    def test_invalid_rank(self):
        result = runner.invoke(
            app, ["generate-offline-data", "--rank", "5", "--world-size", "2"]
        )
        assert result.exit_code != 0

    def test_invalid_concurrency(self):
        result = runner.invoke(app, ["generate-offline-data", "--concurrency", "0"])
        assert result.exit_code != 0


class TestRegenerateResponsesCommand:
    def test_help(self):
        result = runner.invoke(app, ["regenerate-responses", "--help"])
        assert result.exit_code == 0
        output = unstyled_output(result)
        assert "--endpoint" in output
        assert "--dataset" in output
        assert "--concurrency" in output
        assert "--max-tokens" in output

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

    def test_split_only_applies_to_presets(self, tmp_path):
        dataset = tmp_path / "prompts.jsonl"
        dataset.touch()
        result = runner.invoke(
            app,
            [
                "regenerate-responses",
                "--dataset",
                str(dataset),
                "--split",
                "custom",
            ],
        )
        assert result.exit_code != 0
        assert "only apply to dataset presets" in unstyled_output(result)

    def test_invalid_temperature_cycle(self):
        result = runner.invoke(
            app, ["regenerate-responses", "--temperature-cycle", "0.6,notnum"]
        )
        assert result.exit_code != 0


class TestTrainCommand:
    def test_help(self):
        result = runner.invoke(app, ["train", "--help"])
        assert result.exit_code == 0
        output = unstyled_output(result)
        assert "--verifier-name-or-path" in output
        assert "--config" in output
        assert "--speculator-type" in output

    def test_train_appears_in_pipeline_panel(self):
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        assert "train" in unstyled_output(result)
