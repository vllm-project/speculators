"""Run the Pi coding agent inside SWE-smith containers.

Each task row describes a SWE-smith instance: a Docker image checked out at a
buggy revision, plus the problem statement to fix. The runner starts one
container per task, mounts a prepared Pi runtime read-only, points Pi at an
OpenAI-compatible endpoint, and collects the raw Pi event trace and the final
``git diff`` as the candidate patch.

The module imports Docker and Hugging Face ``datasets`` lazily, so it stays
importable and testable without either dependency.
"""

import json
import logging
import shlex
import shutil
import subprocess
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

__all__ = ["build_pi_prompt", "generate_traces", "load_trace_tasks", "run_pi_trace"]

_REQUIRED_FIELDS = ("instance_id", "image_name", "problem_statement")
_OPTIONAL_FIELDS = ("repo", "FAIL_TO_PASS", "PASS_TO_PASS")
_WORKDIR = "/testbed"
_PI_HOME = "/root/.pi/agent"
_PROMPT_PATH = "/run-input/prompt.txt"


def build_pi_prompt(task: dict) -> str:
    """Build the agent prompt for a SWE-smith task row."""
    return (
        "You are working in /testbed, a repository checked out at a commit "
        "that contains a bug.\n"
        "Fix the issue described below with minimal changes to non-test source "
        "files. Do not modify tests.\n"
        "You may inspect files, run commands, and edit files. Stop when the "
        "fix is complete. The harness collects `git diff`.\n\n"
        "<problem_statement>\n"
        f"{task['problem_statement']}\n"
        "</problem_statement>\n"
    )


def _models_config(*, endpoint: str, provider: str, model: str) -> dict:
    return {
        "providers": {
            provider: {
                "name": provider,
                "baseUrl": endpoint,
                "api": "openai-completions",
                "apiKey": "EMPTY",
                "models": [
                    {
                        "id": model,
                        "name": model,
                        "input": ["text"],
                        "cost": {
                            "input": 0,
                            "output": 0,
                            "cacheRead": 0,
                            "cacheWrite": 0,
                        },
                    }
                ],
            }
        }
    }


def _settings_config(tools: str) -> dict:
    return {
        "defaultProjectTrust": "always",
        "defaultTools": [t.strip() for t in tools.split(",") if t.strip()],
    }


def _normalize_row(row: Any, *, source: str, index: int) -> dict | None:
    if not isinstance(row, dict):
        logger.warning("Skipping %s row %d: not a JSON object", source, index)
        return None
    normalized: dict[str, Any] = {}
    for field in _REQUIRED_FIELDS:
        value = row.get(field)
        if not isinstance(value, str) or not value.strip():
            logger.warning(
                "Skipping %s row %d: missing or empty '%s'", source, index, field
            )
            return None
        normalized[field] = value.strip()
    for field in _OPTIONAL_FIELDS:
        if field in row:
            normalized[field] = row[field]
    return normalized


def _rows_from_json_file(path: Path) -> list[Any]:
    text = path.read_text(encoding="utf-8")
    if path.suffix == ".jsonl":
        rows = []
        for line in text.splitlines():
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                logger.warning("Skipping invalid JSON line in %s: %s", path, exc)
        return rows
    payload = json.loads(text)
    if isinstance(payload, list):
        return payload
    logger.warning("Skipping %s: expected a JSON array of task rows", path)
    return []


def _rows_from_local_path(path: Path) -> list[Any]:
    if path.is_dir():
        rows: list[Any] = []
        files = sorted((*path.rglob("*.json"), *path.rglob("*.jsonl")))
        for file in files:
            rows.extend(_rows_from_json_file(file))
        return rows
    return _rows_from_json_file(path)


def _parse_hf_spec(spec: str) -> tuple[str, str | None, str]:
    """Split an ``hf:<id>[:<subset>:<split>]`` spec, or treat it as an id."""
    if not spec.startswith("hf:"):
        return spec, None, "train"
    parts = spec.removeprefix("hf:").split(":")
    match parts:
        case [hf_id]:
            return hf_id, None, "train"
        case [hf_id, split]:
            return hf_id, None, split
        case [hf_id, subset, split]:
            return hf_id, subset, split
        case _:
            raise ValueError(
                f"Invalid hf: spec '{spec}'. "
                f"Expected hf:<dataset_id>[:<subset>:<split>]."
            )


def _rows_from_hf(spec: str) -> list[Any]:
    from datasets import load_dataset  # noqa: PLC0415

    hf_id, subset, split = _parse_hf_spec(spec)
    dataset = load_dataset(hf_id, name=subset, split=split)
    return [dict(row) for row in dataset]


def load_trace_tasks(specs: list[str]) -> list[dict]:
    """Load and normalize SWE-smith trace tasks from local or HF sources.

    Args:
        specs: Local ``.json``/``.jsonl`` files, directories of them, or
            ``hf:<id>[:<subset>:<split>]`` / bare Hugging Face dataset ids.

    Returns:
        Rows with ``instance_id``, ``image_name`` and ``problem_statement``
        (plus any of ``repo``/``FAIL_TO_PASS``/``PASS_TO_PASS``). Malformed
        rows and repeated ``instance_id`` values are skipped with a warning.
    """
    tasks: list[dict] = []
    seen_ids: set[str] = set()
    for spec in specs:
        path = Path(spec)
        rows = _rows_from_local_path(path) if path.exists() else _rows_from_hf(spec)
        for index, row in enumerate(rows):
            normalized = _normalize_row(row, source=spec, index=index)
            if normalized is None:
                continue
            instance_id = normalized["instance_id"]
            if instance_id in seen_ids:
                logger.warning(
                    "Skipping %s row %d: duplicate instance_id '%s'",
                    spec,
                    index,
                    instance_id,
                )
                continue
            seen_ids.add(instance_id)
            tasks.append(normalized)
    return tasks


def _docker_exec(
    container_name: str, argv: list[str], *, timeout: float | None = None
) -> subprocess.CompletedProcess:
    """Run one command in a container, capturing bytes."""
    return subprocess.run(  # noqa: S603
        ["docker", "exec", container_name, *argv],  # noqa: S607
        capture_output=True,
        timeout=timeout,
        stdin=subprocess.DEVNULL,
        check=False,
    )


def _collect_file(
    container_name: str, remote: str, dest: Path, *, timeout: float | None = None
) -> None:
    proc = _docker_exec(container_name, ["cat", remote], timeout=timeout)
    if proc.returncode == 0:
        dest.write_bytes(proc.stdout)


def run_pi_trace(
    task: dict,
    *,
    pi_runtime: str | Path,
    out_dir: str | Path,
    pi_endpoint: str,
    pi_provider: str,
    pi_model: str,
    tools: str,
    timeout: int,
) -> dict:
    """Run one Pi trace for a SWE-smith task in its Docker image.

    Writes ``prompt.txt``, ``pi-home/{models.json,settings.json}``,
    ``pi.jsonl``, ``pi.err``, ``pi.rc`` and ``patch.diff`` under
    ``<out_dir>/<instance_id>/``.

    Returns:
        Dict with ``instance_id``, ``trace_path``, ``patch_path``, ``seconds``
        and ``error`` (``None`` on success).
    """
    import docker  # noqa: PLC0415

    if shutil.which("docker") is None:
        raise RuntimeError(
            "The 'docker' CLI is not on PATH. Install Docker and make the "
            "command available before generating traces."
        )

    instance_id = task["instance_id"]
    run_dir = Path(out_dir) / instance_id
    run_dir.mkdir(parents=True, exist_ok=True)
    pi_home = run_dir / "pi-home"
    pi_home.mkdir(exist_ok=True)
    (run_dir / "prompt.txt").write_text(build_pi_prompt(task))
    (pi_home / "models.json").write_text(
        json.dumps(
            _models_config(endpoint=pi_endpoint, provider=pi_provider, model=pi_model),
            indent=2,
        )
    )
    (pi_home / "settings.json").write_text(
        json.dumps(_settings_config(tools), indent=2)
    )

    trace_path = run_dir / "pi.jsonl"
    patch_path = run_dir / "patch.diff"
    result: dict[str, Any] = {
        "instance_id": instance_id,
        "trace_path": str(trace_path),
        "patch_path": str(patch_path),
        "seconds": 0.0,
        "error": None,
    }

    client = docker.from_env()
    name = f"pi-trace-{uuid.uuid4().hex[:12]}"
    create_kwargs: dict[str, Any] = {
        "image": task["image_name"],
        "name": name,
        "detach": True,
        "command": "tail -f /dev/null",
        "platform": "linux/x86_64",
        "mem_limit": "10g",
        "volumes": {
            str(Path(pi_runtime).resolve()): {
                "bind": "/opt/pi-runtime",
                "mode": "ro",
            },
            str(pi_home.resolve()): {"bind": _PI_HOME, "mode": "rw"},
            str(run_dir.resolve()): {"bind": "/run-input", "mode": "ro"},
        },
        "extra_hosts": {"host.docker.internal": "host-gateway"},
        "environment": {
            "HOME": "/root",
            "PI_CODING_AGENT_DIR": _PI_HOME,
            "PATH": "/opt/pi-runtime/bin:/opt/pi-runtime/node_modules/.bin:"
            "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
        },
    }
    try:
        container = client.containers.create(**create_kwargs)
    except (docker.errors.ImageNotFound, docker.errors.NotFound):
        logger.info("Image %s missing locally, pulling", task["image_name"])
        client.images.pull(task["image_name"])
        container = client.containers.create(**create_kwargs)

    started = time.time()
    deadline = started + timeout

    def remaining() -> float:
        return max(deadline - time.time(), 1.0)

    try:
        container.start()
        checkout = _docker_exec(
            name, ["git", "-C", _WORKDIR, "checkout", instance_id], timeout=timeout
        )
        if checkout.returncode != 0:
            stderr = checkout.stderr.decode(errors="replace")
            raise RuntimeError(f"git checkout {instance_id} failed: {stderr}")

        pi_cmd = (
            f"cd {shlex.quote(_WORKDIR)} && pi "
            f"--provider {shlex.quote(pi_provider)} "
            f"--model {shlex.quote(pi_model)} --mode json --no-session "
            f"--tools {shlex.quote(tools)} "
            f"{shlex.quote('@' + _PROMPT_PATH)} "
            "> /tmp/pi.jsonl 2>/tmp/pi.err; echo $? > /tmp/pi.rc"
        )
        try:
            _docker_exec(name, ["/bin/sh", "-lc", pi_cmd], timeout=timeout)
        except subprocess.TimeoutExpired:
            result["error"] = f"TimeoutExpired: Pi exceeded {timeout}s"

        _collect_file(name, "/tmp/pi.jsonl", trace_path, timeout=remaining())  # noqa: S108
        _collect_file(name, "/tmp/pi.err", run_dir / "pi.err", timeout=remaining())  # noqa: S108
        rc_path = run_dir / "pi.rc"
        _collect_file(name, "/tmp/pi.rc", rc_path, timeout=remaining())  # noqa: S108
        if result["error"] is None and rc_path.exists():
            try:
                pi_rc = int(rc_path.read_text(encoding="utf-8").strip())
            except ValueError:
                pi_rc = 0
            if pi_rc != 0:
                result["error"] = f"Pi exited with code {pi_rc}"

        diff = _docker_exec(name, ["git", "-C", _WORKDIR, "diff"], timeout=remaining())
        patch_path.write_bytes(diff.stdout)
    except Exception as exc:
        if result["error"] is None:
            result["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        result["seconds"] = time.time() - started
        try:
            container.remove(force=True)
        except Exception as exc:
            logger.warning("Failed to remove container %s: %s", name, exc)

    return result


def generate_traces(
    tasks: list[dict], *, out_dir: str | Path, workers: int, **run_kwargs: Any
) -> list[dict]:
    """Run Pi traces for many tasks, isolating per-task failures.

    The function also writes each result to ``<out_dir>/<instance_id>/result.json``.
    """
    results: list[dict | None] = [None] * len(tasks)

    def _run(index: int, task: dict) -> dict:
        try:
            result = run_pi_trace(task, out_dir=out_dir, **run_kwargs)
        except Exception as exc:
            result = {
                "instance_id": task.get("instance_id"),
                "trace_path": None,
                "patch_path": None,
                "seconds": 0.0,
                "error": f"{type(exc).__name__}: {exc}",
            }
        logger.info(
            "Trace %d/%d %s finished (error=%s)",
            index + 1,
            len(tasks),
            result.get("instance_id"),
            result.get("error"),
        )
        instance_id = result.get("instance_id")
        if instance_id:
            try:
                run_dir = Path(out_dir) / instance_id
                run_dir.mkdir(parents=True, exist_ok=True)
                (run_dir / "result.json").write_text(json.dumps(result, indent=2))
            except OSError as exc:
                logger.warning(
                    "Failed to write result.json for %s: %s", instance_id, exc
                )
                message = f"Failed to write result.json: {exc}"
                if result.get("error"):
                    message = f"{result['error']}; {message}"
                result["error"] = message
        return result

    if workers <= 1:
        for index, task in enumerate(tasks):
            results[index] = _run(index, task)
    else:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(_run, index, task): index
                for index, task in enumerate(tasks)
            }
            for future in as_completed(futures):
                results[futures[future]] = future.result()

    return [result for result in results if result is not None]
