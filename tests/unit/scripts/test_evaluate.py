"""Check native invocation and pooled acceptance reporting from saved records."""

from __future__ import annotations

import csv
import json
import shlex
import sys
from pathlib import Path
from unittest.mock import Mock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts" / "evaluate"))

import evaluate  # type: ignore[import-not-found]


def request_record(histogram=None, *, tokens=100, conversation="q1", phase="profiling"):
    histogram = histogram if histogram is not None else {"0": 1, "3": 1}
    steps = sum(histogram.values())
    return {
        "metadata": {"benchmark_phase": phase, "conversation_id": conversation},
        "metrics": {"input_sequence_length": {"value": tokens}},
        "spec_decode_acceptance": {
            "num_spec_steps": steps,
            "num_accepted_draft_tokens": sum(int(k) * n for k, n in histogram.items()),
            "num_draft_tokens": 7 * steps,
            "num_spec_tokens": 7,
            "acceptance_histogram": histogram,
        },
    }


def write_report(directory, records=None, **updates):
    directory.mkdir(parents=True, exist_ok=True)
    records = records if records is not None else [request_record()]
    report = {
        "request_count": {"avg": len(records)},
        "is_complete": True,
        "was_cancelled": False,
        "spec_decode_acceptance_length": {"count": len(records)},
        "input_config": {"datasets": [{"dataset": "fixture"}]},
    }
    report.update(updates)
    (directory / "profile_export_aiperf.json").write_text(json.dumps(report))
    (directory / "profile_export.jsonl").write_text(
        "".join(json.dumps(record) + "\n" for record in records)
    )


def read_csv(directory, name="acceptance.csv"):
    with (directory / name).open() as handle:
        return list(csv.DictReader(handle))


@pytest.fixture
def native_run(monkeypatch):
    runner = Mock(return_value=Mock(returncode=0))
    monkeypatch.setattr(evaluate.subprocess, "run", runner)
    monkeypatch.setattr(evaluate, "save_eval_provenance", Mock())
    monkeypatch.setattr(evaluate, "summarize_output_directory", Mock(return_value=True))
    return runner


@pytest.mark.parametrize("mode", ["acceptance", "speedup"])
def test_native_command_preserves_overrides(tmp_path, native_run, mode):
    output = tmp_path / mode
    body = '{"max_completion_tokens":1024}'
    assert (
        evaluate.main(
            [
                mode,
                "--output-artifact-dir",
                str(output),
                "--dataset",
                "humaneval",
                "--model",
                "test-model",
                "--url",
                "http://localhost:9125",
                "--request-count",
                "100",
                "--concurrency=2",
                "--extra-inputs",
                body,
            ]
        )
        == 0
    )
    command = native_run.call_args.args[0]
    assert command[:2] == ["aiperf", "profile"]
    assert command[2:11] == [
        "--model",
        "test-model",
        "--url",
        "http://localhost:9125",
        "--request-count",
        "100",
        "--concurrency=2",
        "--extra-inputs",
        body,
    ]
    assert native_run.call_args.kwargs == {"check": False}
    assert "--concurrency" not in command
    assert command.count("--extra-inputs") == 1
    assert command[command.index("--extra-inputs") + 1] == body
    assert command[command.index("--public-dataset") + 1] == "spec_al_humaneval"
    assert ("--streaming" in command) == (mode == "speedup")
    assert ("--auto-plot" in command) == (mode == "speedup")
    assert (
        shlex.split((output / "humaneval" / "aiperf_command.txt").read_text())
        == command
    )


@pytest.mark.parametrize(
    ("mode", "native_args", "expected"),
    [
        ("acceptance", [], "qualitative"),
        ("speedup", [], "humaneval"),
        ("acceptance", ["--input-file", "prompts.jsonl"], "qualitative"),
    ],
)
def test_default_dataset(
    tmp_path, native_run, monkeypatch, mode, native_args, expected
):
    selection = Mock(return_value=["--public-dataset", "selected"])
    monkeypatch.setattr(evaluate, "dataset_args", selection)
    assert (
        evaluate.main(
            [mode, "--output-artifact-dir", str(tmp_path / "out"), *native_args]
        )
        == 0
    )
    selection.assert_called_once()
    name, options = selection.call_args.args
    assert name == expected
    assert options.mode == mode
    assert options.input_file == (
        Path("prompts.jsonl") if "--input-file" in native_args else None
    )


@pytest.mark.parametrize(
    "flag", ["--config", "-f", "--conversation-num", "--num-sessions"]
)
def test_native_only_options_fail_before_launch(tmp_path, native_run, flag):
    output = tmp_path / "out"
    with pytest.raises(SystemExit):
        evaluate.main(["speedup", flag, "value", "--output-artifact-dir", str(output)])
    native_run.assert_not_called()
    assert not output.exists()


@pytest.mark.parametrize("runs", [1, 2])
def test_warmup_override_only_applies_to_multiple_trials(tmp_path, native_run, runs):
    assert (
        evaluate.main(
            [
                "speedup",
                "--output-artifact-dir",
                str(tmp_path / "out"),
                "--num-profile-runs",
                str(runs),
            ]
        )
        == 0
    )
    command = native_run.call_args.args[0]
    assert command.count("--num-profile-runs") == 1
    assert ("--no-profile-run-disable-warmup-after-first" in command) == (runs > 1)


@pytest.mark.parametrize(
    "budget",
    [
        [],
        ["--request-count", "8"],
        ["--num-conversations", "2"],
        ["--benchmark-duration", "10"],
    ],
)
def test_qualitative_is_a_single_native_profile(
    tmp_path, native_run, monkeypatch, budget
):
    source = tmp_path / "qualitative.jsonl"
    source.write_text(
        '{"question_id":"q1","category":"coding"}\n'
        '{"question_id":"q2","category":"math"}\n'
    )
    monkeypatch.setattr(
        "eval_datasets.speedbench.prepare_speedbench", Mock(return_value=source)
    )
    output = tmp_path / "out"
    assert (
        evaluate.main(
            [
                "acceptance",
                "--dataset",
                "qualitative",
                "--output-artifact-dir",
                str(output),
                *budget,
            ]
        )
        == 0
    )
    native_run.assert_called_once()
    command = native_run.call_args.args[0]
    assert (
        command[command.index("--custom-dataset-type") + 1] == "speed_bench_qualitative"
    )
    assert command[command.index("--output-artifact-dir") + 1] == str(
        output / "qualitative"
    )
    if not budget or budget[0] == "--num-conversations":
        assert command.count("--num-conversations") == 1
        assert command[command.index("--num-conversations") + 1] == "2"
    else:
        assert "--num-conversations" not in command
    if budget:
        assert command.count(budget[0]) == 1
        assert command[command.index(budget[0]) + 1] == budget[1]


def test_multiple_datasets_continue_after_failure(tmp_path, native_run):
    native_run.side_effect = [Mock(returncode=7), Mock(returncode=0)]
    output = tmp_path / "out"
    assert (
        evaluate.main(
            [
                "acceptance",
                "--dataset",
                "gsm8k",
                "math500",
                "--output-artifact-dir",
                str(output),
            ]
        )
        == 7
    )
    assert native_run.call_count == 2
    for call, name in zip(native_run.call_args_list, ("gsm8k", "math500"), strict=True):
        command = call.args[0]
        assert command[command.index("--public-dataset") + 1] == f"spec_al_{name}"
        assert command[command.index("--output-artifact-dir") + 1] == str(output / name)
    evaluate.summarize_output_directory.assert_called_once_with(output, "acceptance")
    evaluate.save_eval_provenance.assert_called_once_with(output)


def test_invalid_saved_results_fail_successful_native_run(tmp_path, native_run):
    evaluate.summarize_output_directory.return_value = False
    assert (
        evaluate.main(
            [
                "acceptance",
                "--dataset",
                "humaneval",
                "--output-artifact-dir",
                str(tmp_path / "out"),
            ]
        )
        == 1
    )


def test_refuses_existing_directory(tmp_path):
    with pytest.raises(FileExistsError):
        evaluate.main(["speedup", "--output-artifact-dir", str(tmp_path)])


@pytest.mark.parametrize(
    "updates",
    [
        {"error_request_count": {"avg": 1}},
        {"spec_decode_acceptance_length": {"count": 0}},
        {"request_count": {"avg": 0}},
        {"is_complete": False},
        {"was_cancelled": True},
    ],
)
def test_reject_partial_acceptance_but_preserve_results(tmp_path, updates):
    write_report(tmp_path, **updates)
    native = (tmp_path / "profile_export.jsonl").read_text()
    assert not evaluate.summarize_output_directory(tmp_path, "acceptance")
    assert (tmp_path / "profile_export.jsonl").read_text() == native
    assert read_csv(tmp_path)


def test_speedup_checks_all_trials_without_requiring_spec_metrics(tmp_path):
    for trial in ("trial_0001", "trial_0002"):
        point = tmp_path / trial / "concurrency_1"
        write_report(point, spec_decode_acceptance_length={})
        write_report(point / "phases" / "profiling", is_complete=False)
    assert evaluate.summarize_output_directory(tmp_path, "speedup")
    write_report(point, error_request_count={"avg": 1})
    assert not evaluate.summarize_output_directory(tmp_path, "speedup")


def test_missing_or_malformed_reports(tmp_path):
    assert not evaluate.summarize_output_directory(tmp_path, "acceptance")
    (tmp_path / "profile_export_aiperf.json").write_text("{")
    assert not evaluate.summarize_output_directory(tmp_path, "acceptance")


def test_acceptance_pools_counters_and_keeps_zero_positions(tmp_path):
    records = [request_record({"0": 2, "1": 1}), request_record({"3": 1})]
    write_report(
        tmp_path, records, spec_decode_token_weighted_acceptance_length={"avg": 99}
    )
    assert evaluate.summarize_output_directory(tmp_path, "acceptance")
    assert read_csv(tmp_path) == [
        {
            "dataset": ".",
            "acceptance_length": "2.0",
            "acceptance_rate_pct": "14.29",
            **{
                f"acceptance_rate_pos_{i}_pct": value
                for i, value in enumerate(
                    ("50.0", "25.0", "25.0", "0.0", "0.0", "0.0", "0.0"),
                    start=1,
                )
            },
        }
    ]


def test_native_category_report_uses_aiperf_server_metrics(tmp_path):
    native_report = pytest.importorskip("aiperf.analysis.speed_bench_report")

    for category, accepted, steps, drafts in (
        ("coding", 20, 10, 70),
        ("math", 10, 10, 70),
    ):
        directory = tmp_path / category
        write_report(
            directory,
            input_config={
                "datasets": [{"format": f"speed_bench_{category}"}],
                "models": {"items": [{"name": "test-model"}]},
            },
        )
        metrics = {
            name: {"series": [{"stats": {"total": value}}]}
            for name, value in (
                ("vllm:spec_decode_num_accepted_tokens", accepted),
                ("vllm:spec_decode_num_drafts", steps),
                ("vllm:spec_decode_num_draft_tokens", drafts),
            )
        }
        (directory / "server_metrics_export.json").write_text(
            json.dumps({"metrics": metrics})
        )
    for metric, expected in (
        ("accept_length", ("3.00", "2.00")),
        ("accept_rate", ("0.29", "0.14")),
    ):
        output = tmp_path / f"{metric}.csv"
        native_report.generate_report(
            [tmp_path], output=output, output_format="csv", metric=metric
        )
        with output.open() as handle:
            row = next(csv.DictReader(handle))
        assert (row["coding"], row["math"]) == expected


def test_summarize_existing_native_runs_without_launching(tmp_path, native_run):
    (tmp_path / "eval_command.txt").write_text("original profiling command")
    assert evaluate.main(["summarize", "--output-artifact-dir", str(tmp_path)]) == 0
    evaluate.summarize_output_directory.assert_called_once_with(tmp_path, "acceptance")
    native_run.assert_not_called()
    evaluate.save_eval_provenance.assert_not_called()
    assert (tmp_path / "eval_command.txt").read_text() == "original profiling command"


def test_prompt_bins_include_errors_but_exclude_their_metrics(tmp_path):
    tokens = [0, 8192, 8193, 16384, 16385, 1048576, 1048577]
    records = [request_record(tokens=n) for n in tokens]
    failed = request_record({"7": 1}, tokens=None)
    failed["error"] = {"code": 400}
    failed["metrics"] = {}
    known_length_failure = request_record({"7": 1})
    known_length_failure["error"] = {"code": 400}
    known_length_failure["metrics"] = {"error_isl": {"value": 16000}}
    records.extend([failed, known_length_failure])
    write_report(
        tmp_path,
        records,
        request_count={"avg": len(tokens)},
        spec_decode_acceptance_length={"count": len(tokens)},
        error_request_count={"avg": 2},
    )
    assert not evaluate.summarize_output_directory(tmp_path, "acceptance")
    rows = read_csv(tmp_path, "acceptance_by_prompt_length.csv")
    assert [(row["prompt_length"], row["requests"]) for row in rows] == [
        ("0-8192", "2"),
        ("8193-16384", "3"),
        ("16385-32768", "1"),
        ("524289-1048576", "1"),
        (">1048576", "1"),
        ("unknown", "1"),
    ]
    assert rows[-1]["acceptance_length"] == ""
    assert read_csv(tmp_path)[0]["acceptance_length"] == "2.5"


def test_missing_metrics_and_zero_steps_have_blank_rates(tmp_path):
    missing = request_record()
    missing.pop("spec_decode_acceptance")
    write_report(
        tmp_path,
        [missing, request_record({"0": 0})],
        spec_decode_acceptance_length={"count": 1},
    )
    assert not evaluate.summarize_output_directory(tmp_path, "acceptance")
    row = read_csv(tmp_path)[0]
    assert row["acceptance_length"] == row["acceptance_rate_pct"] == ""
    assert row["acceptance_rate_pos_7_pct"] == ""


def test_truncated_request_export_fails_but_keeps_partial_csv(tmp_path):
    write_report(
        tmp_path,
        request_count={"avg": 2},
        spec_decode_acceptance_length={"count": 2},
    )
    assert not evaluate.summarize_output_directory(tmp_path, "acceptance")
    assert read_csv(tmp_path)[0]["acceptance_length"] == "2.5"


def test_bad_record_does_not_discard_valid_records(tmp_path):
    write_report(tmp_path)
    with (tmp_path / "profile_export.jsonl").open("a") as handle:
        handle.write("{\n")
    assert not evaluate.summarize_output_directory(tmp_path, "acceptance")
    assert read_csv(tmp_path)[0]["acceptance_length"] == "2.5"
