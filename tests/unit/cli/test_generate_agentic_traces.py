"""Help and argument-parsing tests for ``generate-agentic-traces``.

The Docker SDK loads only when a task runs, so these tests exercise just the CLI
surface, with no docker, network, or model downloads.
"""

from pathlib import Path

import click
import pytest
import typer
from typer.testing import CliRunner

import speculators.cli.generate_agentic_traces as gat
from speculators.cli import app

runner = CliRunner(env={"COLUMNS": "200"})


def unstyled_output(result):
    return click.unstyle(result.output)


def run_generate(tmp_path, monkeypatch, *, on_generate=None, **overrides):
    output = Path(overrides.pop("output", tmp_path / "out.jsonl"))
    params = {
        "model": "Qwen/Qwen3-32B",
        "data": ["tasks.jsonl"],
        "output": str(output),
        "render_endpoint": "http://localhost:8000",
        "pi_runtime": "pi-runtime",
        "pi_endpoint": "http://localhost:8000/v1",
        "pi_provider": "vllm",
        "pi_model": None,
        "workers": 1,
        "timeout": 10,
        "seq_length": 128,
        "limit": None,
        "max_samples": None,
        "tools": "read",
        "trace_dir": None,
        "keep_traces": True,
    }
    params.update(overrides)

    monkeypatch.setattr(
        gat, "load_trace_tasks", lambda data: [{"instance_id": "task-1"}]
    )
    monkeypatch.setattr(gat, "load_processor", lambda model: object())
    monkeypatch.setattr(
        gat, "rows_from_pi_trace", lambda trace_path, **kwargs: [{"id": "row"}]
    )
    calls = []

    def fake_generate(tasks, *, out_dir, **kwargs):
        calls.append({"out_dir": Path(out_dir), **kwargs})
        if on_generate is not None:
            on_generate(Path(out_dir))
        return [
            {
                "instance_id": "task-1",
                "trace_path": str(Path(out_dir) / "task-1" / "pi.jsonl"),
                "error": None,
            }
        ]

    monkeypatch.setattr(gat, "generate_traces", fake_generate)
    gat._run_generate(**params)
    return calls, output


def fake_which(rg_path=None):
    def which(name, path=None):
        if name == "npm":
            return "/usr/bin/npm"
        return rg_path

    return which


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
        assert "setup-pi" in output

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


class TestGenerateAgenticTracesRun:
    def test_pi_model_defaults_to_target_model(self, tmp_path, monkeypatch):
        calls, _ = run_generate(tmp_path, monkeypatch)

        assert calls[0]["pi_model"] == "Qwen/Qwen3-32B"

    def test_explicit_pi_model_is_preserved(self, tmp_path, monkeypatch):
        calls, _ = run_generate(tmp_path, monkeypatch, pi_model="served-name")

        assert calls[0]["pi_model"] == "served-name"

    @pytest.mark.parametrize(
        ("option", "value"),
        [("limit", -1), ("max_samples", -1)],
    )
    def test_negative_limits_are_rejected(self, tmp_path, monkeypatch, option, value):
        with pytest.raises(typer.BadParameter, match=f"--{option.replace('_', '-')}"):
            run_generate(tmp_path, monkeypatch, **{option: value})

    def test_no_keep_traces_removes_only_created_dirs(self, tmp_path, monkeypatch):
        trace_dir = tmp_path / "traces"
        trace_dir.mkdir()
        (trace_dir / "notes.txt").write_text("keep")

        def on_generate(out_dir):
            (out_dir / "task-1").mkdir(parents=True)
            (out_dir / "task-1" / "pi.jsonl").write_text("{}")

        run_generate(
            tmp_path,
            monkeypatch,
            trace_dir=str(trace_dir),
            keep_traces=False,
            on_generate=on_generate,
        )

        assert trace_dir.is_dir()
        assert (trace_dir / "notes.txt").read_text() == "keep"
        assert not (trace_dir / "task-1").exists()

    def test_no_keep_traces_keeps_output_under_trace_dir(self, tmp_path, monkeypatch):
        trace_dir = tmp_path / "traces"
        trace_dir.mkdir()
        output = trace_dir / "out.jsonl"

        def on_generate(out_dir):
            (out_dir / "task-1").mkdir(parents=True)

        run_generate(
            tmp_path,
            monkeypatch,
            output=output,
            trace_dir=str(trace_dir),
            keep_traces=False,
            on_generate=on_generate,
        )

        assert output.is_file()

    def test_no_keep_traces_removes_new_trace_dir(self, tmp_path, monkeypatch):
        trace_dir = tmp_path / "traces"

        def on_generate(out_dir):
            (out_dir / "task-1").mkdir(parents=True)

        run_generate(
            tmp_path,
            monkeypatch,
            trace_dir=str(trace_dir),
            keep_traces=False,
            on_generate=on_generate,
        )

        assert not trace_dir.exists()


class TestSetupPiCommand:
    def test_help(self):
        result = runner.invoke(app, ["generate-agentic-traces", "setup-pi", "--help"])
        assert result.exit_code == 0
        output = unstyled_output(result)
        for flag in ("--path", "-p", "--node-version", "--force", "--skip-rg", "--yes"):
            assert flag in output


class TestSetupPiConfirmation:
    def test_defaults_to_no(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            gat, "_install_pi", lambda *a: pytest.fail("should not install")
        )
        result = runner.invoke(
            app,
            ["generate-agentic-traces", "setup-pi", "--path", str(tmp_path / "rt")],
            input="n\n",
        )
        assert result.exit_code != 0
        assert "Aborted." in unstyled_output(result)

    def test_yes_skips_prompt(self, tmp_path, monkeypatch):
        calls = []
        monkeypatch.setattr(gat, "_install_pi", lambda *a: calls.append(a))
        result = runner.invoke(
            app,
            [
                "generate-agentic-traces",
                "setup-pi",
                "--path",
                str(tmp_path / "rt"),
                "--yes",
            ],
        )
        assert result.exit_code == 0
        assert calls


class TestInstallPi:
    def test_installed_runtime_is_left_alone(self, tmp_path, monkeypatch):
        pi_launcher = tmp_path / "bin" / "pi"
        pi_launcher.parent.mkdir()
        pi_launcher.touch()
        monkeypatch.setattr(
            gat, "_download_node", lambda *a: pytest.fail("should not download")
        )
        monkeypatch.setattr(
            gat.subprocess,
            "run",
            lambda *a, **k: gat.subprocess.CompletedProcess(a[0], 0, "1.0.0", ""),
        )

        gat._install_pi(tmp_path, "v24.22.2", force=False, skip_rg=True)

        assert pi_launcher.exists()

    def test_unusable_installed_runtime_exits(self, tmp_path, monkeypatch):
        pi_launcher = tmp_path / "bin" / "pi"
        pi_launcher.parent.mkdir()
        pi_launcher.touch()
        monkeypatch.setattr(
            gat.subprocess,
            "run",
            lambda *a, **k: gat.subprocess.CompletedProcess(a[0], 1, "", "boom"),
        )

        with pytest.raises(typer.Exit):
            gat._install_pi(tmp_path, "v24.22.2", force=False, skip_rg=True)

    def test_non_runtime_directory_requires_force(self, tmp_path):
        (tmp_path / "notes.txt").write_text("keep me")

        with pytest.raises(typer.BadParameter, match="--force"):
            gat._install_pi(tmp_path, "v24.22.2", force=False, skip_rg=True)

        assert (tmp_path / "notes.txt").exists()

    def test_force_replaces_directory_and_links_pi(self, tmp_path, monkeypatch):
        (tmp_path / "old.txt").write_text("old")

        def fake_download(version, destination):
            (destination / "bin").mkdir()
            (destination / "bin" / "node").touch()

        monkeypatch.setattr(gat, "_download_node", fake_download)
        monkeypatch.setattr(gat.shutil, "which", fake_which())
        monkeypatch.setattr(
            gat.subprocess,
            "run",
            lambda *a, **k: gat.subprocess.CompletedProcess(a[0], 0, "", ""),
        )

        gat._install_pi(tmp_path, "v24.22.2", force=True, skip_rg=True)

        assert not (tmp_path / "old.txt").exists()
        launcher = tmp_path / "bin" / "pi"
        assert launcher.is_symlink()
        assert launcher.readlink().as_posix() == "../node_modules/.bin/pi"
        assert not (tmp_path / "bin" / "rg").exists()

    def test_copies_rg_when_available(self, tmp_path, monkeypatch):
        rg = tmp_path / "rg-source"
        rg.write_text("rg")
        runtime = tmp_path / "runtime"

        def fake_download(version, destination):
            (destination / "bin").mkdir()

        monkeypatch.setattr(gat, "_download_node", fake_download)
        monkeypatch.setattr(gat.shutil, "which", fake_which(str(rg)))
        monkeypatch.setattr(
            gat.subprocess,
            "run",
            lambda *a, **k: gat.subprocess.CompletedProcess(a[0], 0, "", ""),
        )

        gat._install_pi(runtime, "v24.22.2", force=False, skip_rg=False)

        assert (runtime / "bin" / "rg").read_text() == "rg"

    def test_version_failure_exits(self, tmp_path, monkeypatch):
        def fake_download(version, destination):
            (destination / "bin").mkdir()

        commands = []

        def fake_run(command, *args, **kwargs):
            commands.append(command)
            returncode = 0 if "--version" not in command else 1
            return gat.subprocess.CompletedProcess(command, returncode, "", "boom")

        monkeypatch.setattr(gat, "_download_node", fake_download)
        monkeypatch.setattr(gat.shutil, "which", fake_which())
        monkeypatch.setattr(gat.subprocess, "run", fake_run)

        with pytest.raises(typer.Exit):
            gat._install_pi(tmp_path, "v24.22.2", force=False, skip_rg=True)

        assert any("--version" in command for command in commands)
