"""Help and argument-parsing tests for ``generate-agentic-traces``.

The Docker SDK loads only when a task runs, so these tests exercise just the CLI
surface, with no docker, network, or model downloads.
"""

import click
import pytest
import typer
from typer.testing import CliRunner

import speculators.cli.generate_agentic_traces as gat
from speculators.cli import app

runner = CliRunner(env={"COLUMNS": "200"})


def unstyled_output(result):
    return click.unstyle(result.output)


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


class TestSetupPiCommand:
    def test_help(self):
        result = runner.invoke(app, ["generate-agentic-traces", "setup-pi", "--help"])
        assert result.exit_code == 0
        output = unstyled_output(result)
        for flag in ("--path", "-p", "--node-version", "--force", "--skip-rg"):
            assert flag in output


class TestInstallPi:
    def test_installed_runtime_is_left_alone(self, tmp_path, monkeypatch):
        pi_launcher = tmp_path / "bin" / "pi"
        pi_launcher.parent.mkdir()
        pi_launcher.touch()
        monkeypatch.setattr(
            gat, "_download_node", lambda *a: pytest.fail("should not download")
        )

        gat._install_pi(tmp_path, "v24.22.2", force=False, skip_rg=True)

        assert pi_launcher.exists()

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

        monkeypatch.setattr(gat, "_download_node", fake_download)
        monkeypatch.setattr(gat.shutil, "which", fake_which())
        monkeypatch.setattr(
            gat.subprocess,
            "run",
            lambda *a, **k: gat.subprocess.CompletedProcess(a[0], 1, "", "boom"),
        )

        with pytest.raises(typer.Exit):
            gat._install_pi(tmp_path, "v24.22.2", force=False, skip_rg=True)
