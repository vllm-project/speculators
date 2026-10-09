"""Smoke tests for the speculators CLI."""

import json

import click
from typer.testing import CliRunner

from speculators.cli import app
from speculators.cli.regenerate_responses import (
    _cumulative_counts,
    _load_error_ids,
    load_seen,
)

# Rich uses COLUMNS when rendering help, so keep assertions independent of the
# terminal width provided by the environment running the tests.
runner = CliRunner(env={"COLUMNS": "200"})


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

    def test_invalid_temperature(self):
        result = runner.invoke(
            app, ["regenerate-responses", "--temperature", "not json"]
        )
        assert result.exit_code != 0

    def test_reasoning_effort_weights_must_sum_to_one(self):
        dist = '{"low": 0.2, "high": 0.3}'
        result = runner.invoke(
            app, ["regenerate-responses", "--reasoning-effort", dist]
        )
        assert result.exit_code != 0
        assert "must sum to 1.0" in unstyled_output(result)

    def test_temperature_weights_must_sum_to_one(self):
        result = runner.invoke(
            app, ["regenerate-responses", "--temperature", '{"0.6": 0.8, "0.8": 0.8}']
        )
        assert result.exit_code != 0
        assert "must sum to 1.0" in unstyled_output(result)


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


class TestLoadSeen:
    """Unit tests for the resume-seen scanner (RFC1 cumulative counters)."""

    def test_missing_file_returns_empty_sets(self, tmp_path):
        """A missing output file yields two empty sets."""
        seen, truncated = load_seen(str(tmp_path / "missing.jsonl"))
        assert seen == set()
        assert truncated == set()

    def test_collects_primary_ids_and_truncations(self, tmp_path):
        """primary_id keys are collected; length rows mark truncation."""
        outfile = tmp_path / "out.jsonl"
        rows = [
            {
                "id": "c1_gen1",
                "primary_id": "c1",
                "metadata": {"finish_reason": "stop"},
            },
            {
                "id": "c2_gen1",
                "primary_id": "c2",
                "metadata": {"finish_reason": "length"},
            },
        ]
        outfile.write_text(
            "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
        )
        seen, truncated = load_seen(str(outfile))
        assert seen == {"c1", "c2"}
        assert truncated == {"c2"}

    def test_multi_row_conversation_counts_once(self, tmp_path):
        """Rows sharing one primary_id count once in both sets."""
        # A multi-turn conversation fans out to multiple rows sharing one
        # primary_id; both the seen set and the truncation set deduplicate it.
        outfile = tmp_path / "out.jsonl"
        rows = [
            {
                "id": "c1_gen1",
                "primary_id": "c1",
                "metadata": {"finish_reason": "length"},
            },
            {
                "id": "c1_gen2",
                "primary_id": "c1",
                "metadata": {"finish_reason": "length"},
            },
            {
                "id": "c1_gen3",
                "primary_id": "c1",
                "metadata": {"finish_reason": "stop"},
            },
        ]
        outfile.write_text(
            "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
        )
        seen, truncated = load_seen(str(outfile))
        assert seen == {"c1"}
        assert truncated == {"c1"}

    def test_legacy_rows_fall_back_to_top_level_id(self, tmp_path):
        """Rows without primary_id resume on the legacy top-level id."""
        # Output files written before the fan-out carry only a top-level "id"
        outfile = tmp_path / "out.jsonl"
        outfile.write_text(
            json.dumps({"id": "legacy1", "metadata": {}}) + "\n", encoding="utf-8"
        )
        seen, truncated = load_seen(str(outfile))
        assert seen == {"legacy1"}
        assert truncated == set()

    def test_escaped_payload_text_cannot_spoof_ids(self, tmp_path):
        """Payload text cannot inject ids through the fast-path regexes."""
        # The regex fast path must not be confused by escaped quotes inside
        # row payloads; only the real top-level keys are picked up.
        outfile = tmp_path / "out.jsonl"
        row = {
            "id": "c1_gen1",
            "primary_id": "c1",
            "text": 'some text with a nested {"id": "spoofed"} inside',
            "metadata": {"finish_reason": "stop"},
        }
        outfile.write_text(json.dumps(row) + "\n", encoding="utf-8")
        seen, truncated = load_seen(str(outfile))
        assert seen == {"c1"}
        assert truncated == set()


class TestLoadErrorIds:
    """Unit tests for the cumulative error-id collector (RFC1)."""

    def test_missing_file_returns_empty_set(self, tmp_path):
        """A missing errors file yields an empty set."""
        assert _load_error_ids(str(tmp_path / "missing.errors.jsonl")) == set()

    def test_dedups_failures_repeated_across_sessions(self, tmp_path):
        """Failures repeated across sessions deduplicate to one id."""
        # The errors file appends one row per failed attempt, so a
        # conversation failing in several sessions appears multiple times.
        errorfile = tmp_path / "out.errors.jsonl"
        rows = [
            {"id": "conv1", "metadata": {"error": "HTTP 400 ..."}},
            {"id": "conv1", "metadata": {"error": "HTTP 400 ..."}},
            {"id": "conv2", "metadata": {"error": "timeout"}},
        ]
        errorfile.write_text(
            "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
        )
        assert _load_error_ids(str(errorfile)) == {"conv1", "conv2"}

    def test_ignores_rows_without_id(self, tmp_path):
        """Error rows without an id contribute nothing."""
        errorfile = tmp_path / "out.errors.jsonl"
        errorfile.write_text(
            json.dumps({"metadata": {"error": "no id here"}}) + "\n",
            encoding="utf-8",
        )
        assert _load_error_ids(str(errorfile)) == set()


class TestCumulativeCounts:
    """Unit tests for the cumulative ok/err/trunc postfix math (RFC1)."""

    @staticmethod
    def _stats(**overrides):
        """Build a worker stats dict with zeroed counters and empty id sets."""
        stats = {
            "ok": 0,
            "errors": 0,
            "truncated": 0,
            "resumed_skipped": 0,
            "seen_ids": set(),
            "ok_ids": set(),
            "trunc_ids": set(),
            "error_ids": set(),
        }
        stats.update(overrides)
        return stats

    def test_no_history_matches_session_counters(self):
        """With no history, cumulative counts equal the session scalars."""
        # Runtime invariant: the scalar counters and the id sets stay in sync
        # (workers update both), so counts derive from the sets alone.
        counts = _cumulative_counts(
            self._stats(
                ok=5,
                errors=1,
                truncated=2,
                ok_ids={"c1", "c2", "c3", "c4", "c5"},
                error_ids={"e1"},
                trunc_ids={"t1", "t2"},
            )
        )
        assert counts == {"ok": 5, "err": 1, "trunc": 2}

    def test_resumed_history_is_included(self):
        """Historical ids pre-seed the cumulative err/trunc counters."""
        # trunc_ids/error_ids start pre-populated with historical ids
        counts = _cumulative_counts(
            self._stats(
                resumed_skipped=100,
                ok=3,
                truncated=1,
                trunc_ids={"t1"},
                error_ids={"e1"},
            )
        )
        assert counts == {"ok": 103, "err": 1, "trunc": 1}

    def test_errors_exclude_retried_successes(self):
        """Ids that failed then succeeded count once, under ok."""
        # e1 failed in a previous session and succeeded in this one (ok_ids);
        # e2/e3 failed this session only. err must not double-count e1.
        counts = _cumulative_counts(
            self._stats(
                ok=2,
                ok_ids={"e1"},
                error_ids={"e1", "e2", "e3"},
            )
        )
        assert counts == {"ok": 2, "err": 2, "trunc": 0}

    def test_errors_exclude_conversations_completed_before_session(self):
        """Failures already completed before this session are not errors."""
        # e0 failed historically and completed before this session (seen_ids)
        counts = _cumulative_counts(
            self._stats(resumed_skipped=7, error_ids={"e0"}, seen_ids={"e0"})
        )
        assert counts == {"ok": 7, "err": 0, "trunc": 0}

    def test_trunc_union_dedups_across_sessions(self):
        """Truncation in any session counts the conversation once."""
        # conv t1 truncated in a previous session and again in this session
        counts = _cumulative_counts(self._stats(trunc_ids={"t1", "t2"}, truncated=2))
        assert counts == {"ok": 0, "err": 0, "trunc": 2}
