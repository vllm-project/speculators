import json
import shlex
import subprocess
import sys
import types

import pytest

from speculators.data_generation import agentic_runner
from speculators.data_generation.agentic_runner import (
    build_pi_prompt,
    generate_traces,
    load_trace_tasks,
    run_pi_trace,
)

VALID = {
    "instance_id": "acme__pkg.abc123.1",
    "image_name": "acme-pkg:latest",
    "problem_statement": "Fix the bug.",
    "repo": "acme/pkg",
    "FAIL_TO_PASS": ["tests/test_x.py::test_x"],
}


def test_load_trace_tasks_from_json_skips_malformed(tmp_path):
    path = tmp_path / "tasks.json"
    path.write_text(
        json.dumps(
            [
                VALID,
                {"instance_id": "no-image", "problem_statement": "x"},
                {"image_name": "img", "problem_statement": "x"},
            ]
        )
    )

    tasks = load_trace_tasks([str(path)])

    assert tasks == [VALID]


def test_load_trace_tasks_from_jsonl_and_directory(tmp_path):
    valid = {**VALID, "instance_id": "other"}
    (tmp_path / "a.jsonl").write_text(
        json.dumps(VALID) + "\n" + "{not json}\n" + json.dumps(valid) + "\n"
    )

    tasks = load_trace_tasks([str(tmp_path)])

    assert [t["instance_id"] for t in tasks] == [VALID["instance_id"], "other"]
    assert tasks[0]["FAIL_TO_PASS"] == VALID["FAIL_TO_PASS"]


def test_load_trace_tasks_from_hf_spec(monkeypatch):
    captured = {}

    def fake_load_dataset(hf_id, name=None, split=None):
        captured.update(hf_id=hf_id, name=name, split=split)
        return [dict(VALID)]

    monkeypatch.setitem(
        sys.modules,
        "datasets",
        types.SimpleNamespace(load_dataset=fake_load_dataset),
    )

    tasks = load_trace_tasks(["hf:org/data:subset:validation"])

    assert tasks == [VALID]
    assert captured == {"hf_id": "org/data", "name": "subset", "split": "validation"}


def test_load_trace_tasks_skips_duplicate_instance_ids(tmp_path):
    duplicate = {**VALID, "problem_statement": "A different bug."}
    path = tmp_path / "tasks.json"
    path.write_text(json.dumps([VALID, duplicate]))

    tasks = load_trace_tasks([str(path)])

    assert tasks == [VALID]


@pytest.mark.parametrize(
    "instance_id",
    ["/etc/passwd", "../escape", "a/b", "a\\b", ".", "..", "", "trailing/"],
)
def test_load_trace_tasks_skips_unsafe_instance_ids(tmp_path, instance_id):
    path = tmp_path / "tasks.json"
    path.write_text(json.dumps([{**VALID, "instance_id": instance_id}, VALID]))

    tasks = load_trace_tasks([str(path)])

    assert tasks == [VALID]


def test_load_trace_tasks_keeps_swe_smith_instance_id(tmp_path):
    task = {
        **VALID,
        "instance_id": "pytest-dev__iniconfig.16793ead.combine_file__abc123",
    }
    path = tmp_path / "tasks.json"
    path.write_text(json.dumps([task]))

    assert load_trace_tasks([str(path)]) == [task]


def test_build_pi_prompt_includes_problem_and_workdir():
    prompt = build_pi_prompt(VALID)

    assert "/testbed" in prompt
    assert VALID["problem_statement"] in prompt


class _FakeContainer:
    def __init__(self):
        self.started = False
        self.removed = False

    def start(self):
        self.started = True

    def remove(self, force=False):
        self.removed = force


class _FakeContainers:
    def __init__(self, container):
        self.container = container
        self.create_kwargs = None

    def create(self, **kwargs):
        self.create_kwargs = kwargs
        return self.container


def _fake_docker(monkeypatch):
    container = _FakeContainer()
    containers = _FakeContainers(container)
    module = types.SimpleNamespace(
        from_env=lambda: types.SimpleNamespace(containers=containers)
    )
    monkeypatch.setitem(sys.modules, "docker", module)
    monkeypatch.setattr(agentic_runner.shutil, "which", lambda _: "/usr/bin/docker")
    return container, containers


def _run_kwargs(tmp_path, **overrides):
    runtime = tmp_path / "runtime"
    runtime.mkdir(exist_ok=True)
    kwargs = {
        "pi_runtime": runtime,
        "out_dir": tmp_path / "out",
        "pi_endpoint": "http://host.docker.internal:8000/v1",
        "pi_provider": "vllm",
        "pi_model": "test-model",
        "tools": "read,bash,edit,write",
        "timeout": 10,
    }
    kwargs.update(overrides)
    return kwargs


def _fake_exec(container_name, argv, *, timeout=None):
    if argv[0] == "cat":
        outputs = {
            "/tmp/pi.jsonl": b'{"type":"agent_end"}\n',
            "/tmp/pi.err": b"",
            "/tmp/pi.rc": b"0\n",
        }
        return subprocess.CompletedProcess(argv, 0, outputs[argv[1]], b"")
    if argv[:2] == ["git", "-C"] and argv[-1] == "diff":
        return subprocess.CompletedProcess(argv, 0, b"diff --git a/f b/f\n", b"")
    return subprocess.CompletedProcess(argv, 0, b"", b"")


def test_run_pi_trace_without_docker(tmp_path, monkeypatch):
    container, containers = _fake_docker(monkeypatch)
    monkeypatch.setattr(agentic_runner, "_docker_exec", _fake_exec)

    result = run_pi_trace(VALID, **_run_kwargs(tmp_path))

    run_dir = tmp_path / "out" / VALID["instance_id"]
    assert result["error"] is None
    assert result["instance_id"] == VALID["instance_id"]
    assert (run_dir / "prompt.txt").read_text() == build_pi_prompt(VALID)
    models = json.loads((run_dir / "pi-home" / "models.json").read_text())
    provider = models["providers"]["vllm"]
    assert provider["baseUrl"] == "http://host.docker.internal:8000/v1"
    assert provider["models"][0]["id"] == "test-model"
    settings = json.loads((run_dir / "pi-home" / "settings.json").read_text())
    assert settings["defaultProjectTrust"] == "always"
    assert settings["defaultTools"] == ["read", "bash", "edit", "write"]
    assert (run_dir / "pi.jsonl").read_text() == '{"type":"agent_end"}\n'
    assert result["patch_path"] == str(run_dir / "patch.diff")
    assert (run_dir / "patch.diff").read_text() == "diff --git a/f b/f\n"
    assert container.started
    assert container.removed
    assert containers.create_kwargs["extra_hosts"] == {
        "host.docker.internal": "host-gateway"
    }


def test_run_pi_trace_requires_docker_cli(tmp_path, monkeypatch):
    _fake_docker(monkeypatch)
    monkeypatch.setattr(agentic_runner.shutil, "which", lambda _: None)

    with pytest.raises(RuntimeError, match="docker.*not on PATH"):
        run_pi_trace(VALID, **_run_kwargs(tmp_path))


@pytest.mark.parametrize(
    "instance_id", ["/etc/passwd", "../escape", "a/b", "a\\b", ".", "..", ""]
)
def test_run_pi_trace_rejects_unsafe_instance_id(tmp_path, monkeypatch, instance_id):
    container, containers = _fake_docker(monkeypatch)
    monkeypatch.setattr(agentic_runner, "_docker_exec", _fake_exec)
    task = {**VALID, "instance_id": instance_id}
    out_dir = tmp_path / "out"

    with pytest.raises(ValueError, match="instance_id"):
        run_pi_trace(task, **_run_kwargs(tmp_path))

    assert containers.create_kwargs is None
    assert not container.started
    assert not out_dir.exists()


def test_run_pi_trace_quotes_shell_arguments(tmp_path, monkeypatch):
    _fake_docker(monkeypatch)
    calls = []

    def recording_exec(container_name, argv, *, timeout=None):
        calls.append(argv)
        return _fake_exec(container_name, argv, timeout=timeout)

    monkeypatch.setattr(agentic_runner, "_docker_exec", recording_exec)

    run_pi_trace(
        VALID,
        **_run_kwargs(
            tmp_path,
            pi_provider="my provider",
            pi_model="model; rm -rf /",
            tools="read, bash",
        ),
    )

    pi_cmd = next(argv[-1] for argv in calls if argv[:2] == ["/bin/sh", "-lc"])
    tokens = shlex.split(pi_cmd)
    assert "my provider" in tokens
    assert "model; rm -rf /" in tokens
    assert tokens[tokens.index("--tools") + 1] == "read, bash"
    assert "@/run-input/prompt.txt" in tokens
    assert pi_cmd.count("> /tmp/pi.jsonl") == 1


def test_run_pi_trace_sets_error_on_nonzero_rc(tmp_path, monkeypatch):
    _fake_docker(monkeypatch)

    def failing_exec(container_name, argv, *, timeout=None):
        if argv[0] == "cat" and argv[1] == "/tmp/pi.rc":
            return subprocess.CompletedProcess(argv, 0, b"3\n", b"")
        return _fake_exec(container_name, argv, timeout=timeout)

    monkeypatch.setattr(agentic_runner, "_docker_exec", failing_exec)

    result = run_pi_trace(VALID, **_run_kwargs(tmp_path))

    assert result["error"] == "Pi exited with code 3"
    assert (tmp_path / "out" / VALID["instance_id"] / "pi.rc").read_text() == "3\n"


def test_run_pi_trace_keeps_timeout_error_on_nonzero_rc(tmp_path, monkeypatch):
    _fake_docker(monkeypatch)

    def timed_out_exec(container_name, argv, *, timeout=None):
        if argv[:2] == ["/bin/sh", "-lc"]:
            raise subprocess.TimeoutExpired(argv, timeout)
        return _fake_exec(container_name, argv, timeout=timeout)

    monkeypatch.setattr(agentic_runner, "_docker_exec", timed_out_exec)

    result = run_pi_trace(VALID, **_run_kwargs(tmp_path))

    assert result["error"] == "TimeoutExpired: Pi exceeded 10s"


def test_generate_traces_isolates_failures(tmp_path, monkeypatch):
    def fake_run(task, **kwargs):
        if task["instance_id"] == "boom":
            raise RuntimeError("container failed")
        return {"instance_id": task["instance_id"], "error": None}

    monkeypatch.setattr(agentic_runner, "run_pi_trace", fake_run)
    tasks = [VALID, {**VALID, "instance_id": "boom"}]

    results = generate_traces(tasks, out_dir=tmp_path, workers=2, timeout=1)

    by_id = {r["instance_id"]: r for r in results}
    assert by_id[VALID["instance_id"]]["error"] is None
    assert "container failed" in by_id["boom"]["error"]
    for task in tasks:
        assert (tmp_path / task["instance_id"] / "result.json").exists()


def test_generate_traces_reports_result_write_failure(tmp_path, monkeypatch):
    def fake_run(task, **kwargs):
        return {"instance_id": task["instance_id"], "error": None}

    monkeypatch.setattr(agentic_runner, "run_pi_trace", fake_run)
    blocked = tmp_path / VALID["instance_id"]
    blocked.write_text("not a directory")
    other = {**VALID, "instance_id": "other"}

    results = generate_traces([VALID, other], out_dir=tmp_path, workers=1, timeout=1)

    by_id = {r["instance_id"]: r for r in results}
    assert "result.json" in by_id[VALID["instance_id"]]["error"]
    assert by_id["other"]["error"] is None
    assert (tmp_path / "other" / "result.json").exists()
