import json
import subprocess
import sys
import types

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
    return container, containers


def _fake_exec(container_name, argv, *, timeout=None):
    if argv[0] == "cat":
        outputs = {
            "/tmp/pi.jsonl": b'{"type":"agent_end"}\n',
            "/tmp/pi.err": b"",
            "/tmp/pi.rc": b"0\n",
        }
        return subprocess.CompletedProcess(argv, 0, outputs[argv[1]], b"")
    if argv[0] == "/bin/sh" and "git diff" in argv[-1]:
        return subprocess.CompletedProcess(argv, 0, b"diff --git a/f b/f\n", b"")
    return subprocess.CompletedProcess(argv, 0, b"", b"")


def test_run_pi_trace_without_docker(tmp_path, monkeypatch):
    container, containers = _fake_docker(monkeypatch)
    monkeypatch.setattr(agentic_runner, "_docker_exec", _fake_exec)
    runtime = tmp_path / "runtime"
    runtime.mkdir()

    result = run_pi_trace(
        VALID,
        pi_runtime=runtime,
        out_dir=tmp_path / "out",
        pi_endpoint="http://host.docker.internal:8000/v1",
        pi_provider="vllm",
        pi_model="test-model",
        tools="read,bash,edit,write",
        timeout=10,
    )

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
