"""
Generate agentic traces with Pi and convert them to speculator-format rows.

For each SWE-smith-style dataset instance, a Pi agent runs inside the task
container against a vLLM endpoint and writes a raw execution trace. The command
then renders and tokenizes each trace into speculator-format rows with
``input_ids`` and ``loss_mask``, ready for training.

Usage::

    speculators generate-agentic-traces setup-pi --path ./pi-runtime

    speculators generate-agentic-traces \\
        --model meta-llama/Llama-3.1-8B-Instruct \\
        --data ./swe_smith_tasks.jsonl \\
        --render-endpoint http://localhost:8000 \\
        --pi-runtime ./pi-runtime \\
        --output ./traces.jsonl
"""

import json
import logging
import os
import platform
import shutil
import subprocess
import tarfile
import tempfile
import urllib.request
from pathlib import Path
from typing import Annotated

import typer

from speculators.data_generation.agentic_runner import (
    generate_traces,
    load_trace_tasks,
)
from speculators.data_generation.agentic_traces import rows_from_pi_trace
from speculators.data_generation.logging_utils import PipelineLogger
from speculators.data_generation.preprocessing import load_processor

log = PipelineLogger(__name__)

DEFAULT_NODE_VERSION = "v24.22.2"
PI_PACKAGE = "@earendil-works/pi-coding-agent"

generate_agentic_traces_app = typer.Typer(
    help="Generate agentic traces with Pi and convert them to speculator rows.",
)


def _node_platform() -> str:
    system = platform.system().lower()
    machine = platform.machine().lower()
    if system != "linux":
        raise typer.BadParameter(
            f"unsupported platform {system!r}; only Linux is supported"
        )
    if machine in ("x86_64", "amd64"):
        return "linux-x64"
    if machine in ("aarch64", "arm64"):
        return "linux-arm64"
    raise typer.BadParameter(
        f"unsupported architecture {machine!r}; only x86_64 and arm64 are supported"
    )


def _download_node(version: str, destination: Path) -> None:
    node_platform = _node_platform()
    filename = f"node-{version}-{node_platform}.tar.xz"
    url = f"https://nodejs.org/dist/{version}/{filename}"
    log.info(f"Downloading {url}")

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        archive = tmp_path / filename
        with urllib.request.urlopen(url) as response, archive.open("wb") as handle:
            shutil.copyfileobj(response, handle)

        extract_dir = tmp_path / "node"
        extract_dir.mkdir()
        with tarfile.open(archive) as tar:
            tar.extractall(extract_dir, filter="data")

        roots = list(extract_dir.iterdir())
        if len(roots) != 1 or not roots[0].is_dir():
            raise typer.BadParameter(f"unexpected layout in {filename}")
        for entry in roots[0].iterdir():
            shutil.move(str(entry), str(destination / entry.name))


def _verify_pi(pi_launcher: Path, env: dict[str, str]) -> None:
    log.info(f"Verifying {pi_launcher}")
    version = subprocess.run(  # noqa: S603
        [str(pi_launcher), "--version"],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    if version.returncode != 0:
        log.error(f"{pi_launcher} --version failed with exit code {version.returncode}")
        if version.stderr:
            log.error(version.stderr.strip())
        raise typer.Exit(1)


def _install_pi(
    path: Path,
    node_version: str,
    force: bool,
    skip_rg: bool,
) -> None:
    pi_launcher = path / "bin" / "pi"
    env = os.environ.copy()
    env["PATH"] = f"{path / 'bin'}{os.pathsep}{env.get('PATH', '')}"

    if pi_launcher.exists() and not force:
        _verify_pi(pi_launcher, env)
        log.info(f"Pi runtime already installed at {path} (use --force to replace)")
        return

    if path.exists() and any(path.iterdir()) and not pi_launcher.exists() and not force:
        raise typer.BadParameter(
            f"{path} exists and is not an empty Pi runtime; pass --force to replace it"
        )

    if force and path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)

    _download_node(node_version, path)

    log.info(f"Installing {PI_PACKAGE}")
    npm = shutil.which("npm", path=env["PATH"])
    if npm is None:
        log.error("npm was not found in the downloaded Node runtime")
        raise typer.Exit(1)
    result = subprocess.run(  # noqa: S603
        [npm, "install", "--prefix", str(path), PI_PACKAGE],
        env=env,
        check=False,
    )
    if result.returncode != 0:
        log.error(f"npm install failed with exit code {result.returncode}")
        raise typer.Exit(1)

    bin_dir = path / "bin"
    bin_dir.mkdir(exist_ok=True)
    pi_package_bin = path / "node_modules" / ".bin" / "pi"
    pi_launcher.unlink(missing_ok=True)
    pi_launcher.symlink_to(os.path.relpath(pi_package_bin, bin_dir))

    if not skip_rg:
        rg = shutil.which("rg")
        if rg is None:
            log.warning("ripgrep (rg) not found on PATH; skipping copy")
        else:
            shutil.copy2(rg, bin_dir / "rg")
            log.info(f"Copied rg from {rg}")

    _verify_pi(pi_launcher, env)

    log.success(f"Pi runtime installed at {path}")
    log.info(f"Next: speculators generate-agentic-traces --pi-runtime {path} ...")


@generate_agentic_traces_app.command("setup-pi")
def setup_pi(
    path: Annotated[
        Path,
        typer.Option(
            "--path",
            "-p",
            help="Directory to install the Pi runtime into",
        ),
    ] = Path("./pi-runtime"),
    node_version: Annotated[
        str,
        typer.Option(help="Node.js version to download"),
    ] = DEFAULT_NODE_VERSION,
    force: Annotated[
        bool,
        typer.Option(help="Replace an existing installation"),
    ] = False,
    skip_rg: Annotated[
        bool,
        typer.Option(help="Do not copy ripgrep into the runtime"),
    ] = False,
    yes: Annotated[
        bool,
        typer.Option("--yes", "-y", help="Skip the confirmation prompt"),
    ] = False,
) -> None:
    """Install a Pi runtime directory for --pi-runtime."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    target = path.resolve()
    if not yes:
        rg_note = (
            "do not copy ripgrep" if skip_rg else "copy ripgrep from PATH when found"
        )
        replace_note = " and replace the existing directory" if force else ""
        typer.echo(
            f"Install the Pi runtime in {target}{replace_note}:\n"
            f"  Node.js {node_version} from nodejs.org\n"
            f"  @earendil-works/pi-coding-agent from npm\n"
            f"  {rg_note}\n"
        )
        if not typer.confirm("Continue?", default=False):
            typer.echo("Aborted.")
            raise typer.Exit(1)
    _install_pi(target, node_version, force, skip_rg)


def _remove_created_trace_dirs(
    trace_path: Path,
    existing_names: set[str],
    created_ids: set[str],
    output_path: Path,
) -> None:
    for name in created_ids - existing_names:
        entry = trace_path / name
        if not entry.is_dir():
            continue
        if output_path.resolve().is_relative_to(entry.resolve()):
            continue
        shutil.rmtree(entry, ignore_errors=True)


def _cleanup_traces(
    trace_path: Path,
    existing_names: set[str],
    results: list[dict],
    output_path: Path,
    trace_path_existed: bool,
) -> None:
    created_ids = {
        result["instance_id"] for result in results if result.get("instance_id")
    }
    _remove_created_trace_dirs(trace_path, existing_names, created_ids, output_path)
    if not trace_path_existed and trace_path.is_dir() and not any(trace_path.iterdir()):
        trace_path.rmdir()


def _validate_limits(limit: int | None, max_samples: int | None) -> None:
    if limit is not None and limit < 0:
        raise typer.BadParameter(f"--limit must be non-negative, got {limit}")
    if max_samples is not None and max_samples < 0:
        raise typer.BadParameter(
            f"--max-samples must be non-negative, got {max_samples}"
        )


def _run_generate(  # noqa: PLR0917
    model: str,
    data: list[str],
    output: str,
    render_endpoint: str,
    pi_runtime: str,
    pi_endpoint: str,
    pi_provider: str,
    pi_model: str | None,
    workers: int,
    timeout: int,
    seq_length: int,
    limit: int | None,
    max_samples: int | None,
    tools: str,
    trace_dir: str | None,
    keep_traces: bool,
) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    _validate_limits(limit, max_samples)

    output_path = Path(output)
    trace_path = Path(trace_dir) if trace_dir is not None else Path(f"{output}.traces")
    resolved_pi_model = pi_model or model
    trace_path_existed = trace_path.is_dir()
    existing_trace_names = (
        {entry.name for entry in trace_path.iterdir()} if trace_path_existed else set()
    )

    log.section("Generating agentic traces")
    log.config(
        {
            "Target Model": model,
            "Datasets": data,
            "Output": str(output_path),
            "Trace Dir": str(trace_path),
            "Render Endpoint": render_endpoint,
            "Pi Endpoint": pi_endpoint,
            "Pi Model": resolved_pi_model,
            "Workers": workers,
            "Tools": tools,
        }
    )

    tasks = load_trace_tasks(data)
    if limit is not None:
        tasks = tasks[:limit]
    log.info(f"Loaded {len(tasks)} task(s)")
    if not tasks:
        log.warning("No tasks to run")
        raise typer.Exit(1)

    results = generate_traces(
        tasks,
        out_dir=trace_path,
        workers=workers,
        pi_runtime=pi_runtime,
        pi_endpoint=pi_endpoint,
        pi_provider=pi_provider,
        pi_model=resolved_pi_model,
        tools=tools,
        timeout=timeout,
    )

    succeeded = [r for r in results if not r.get("error") and r.get("trace_path")]
    failed = len(results) - len(succeeded)
    log.info(f"{len(succeeded)} trace(s) succeeded, {failed} failed")

    processor = load_processor(model)

    rows: list[dict] = []
    for result in succeeded:
        try:
            rows.extend(
                rows_from_pi_trace(
                    result["trace_path"],
                    processor=processor,
                    render_endpoint=render_endpoint,
                    seq_length=seq_length,
                    primary_id=result["instance_id"],
                )
            )
        except Exception as exc:  # noqa: BLE001
            log.warning(
                f"Failed to convert trace for {result.get('instance_id')}: {exc!r}"
            )
        if max_samples is not None and len(rows) >= max_samples:
            break

    if max_samples is not None:
        rows = rows[:max_samples]

    if not rows:
        log.warning("No training rows were produced")
        raise typer.Exit(1)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    if not keep_traces:
        _cleanup_traces(
            trace_path,
            existing_trace_names,
            results,
            output_path,
            trace_path_existed,
        )

    log.section("Done")
    log.success(f"Wrote {len(rows)} row(s) from {len(succeeded)} trace(s)")
    log.info(f"Output: {output_path}")


@generate_agentic_traces_app.callback(invoke_without_command=True)
def generate_agentic_traces(  # noqa: PLR0917
    ctx: typer.Context,
    model: Annotated[
        str | None,
        typer.Option(
            help="HuggingFace model ID or local path used for tokenizer/processor",
        ),
    ] = None,
    data: Annotated[
        list[str] | None,
        typer.Option(
            "--data",
            help=(
                "SWE-smith-style task dataset: local JSON/JSONL file or directory, "
                "HuggingFace id, or hf: prefix (repeatable)"
            ),
        ),
    ] = None,
    output: Annotated[
        str | None,
        typer.Option(help="Output JSONL path, e.g. traces.jsonl"),
    ] = None,
    render_endpoint: Annotated[
        str | None,
        typer.Option(
            help=(
                "Base URL of a running vLLM server used to render traces "
                "(e.g. http://localhost:8000)"
            ),
        ),
    ] = None,
    pi_runtime: Annotated[
        str | None,
        typer.Option(help="Path to the prepared Pi runtime"),
    ] = None,
    pi_endpoint: Annotated[
        str,
        typer.Option(
            help="OpenAI-compatible endpoint as seen from inside the task container",
        ),
    ] = "http://host.docker.internal:8000/v1",
    pi_provider: Annotated[
        str,
        typer.Option(help="Pi model provider name"),
    ] = "vllm",
    pi_model: Annotated[
        str | None,
        typer.Option(
            help="Model name exposed to Pi by the endpoint (defaults to --model)"
        ),
    ] = None,
    workers: Annotated[
        int,
        typer.Option(help="Number of tasks to run concurrently"),
    ] = 4,
    timeout: Annotated[
        int,
        typer.Option(help="Per-task timeout in seconds"),
    ] = 900,
    seq_length: Annotated[
        int,
        typer.Option(help="Maximum sequence length of an output row"),
    ] = 32768,
    limit: Annotated[
        int | None,
        typer.Option(help="Maximum number of tasks to run"),
    ] = None,
    max_samples: Annotated[
        int | None,
        typer.Option(help="Maximum number of output rows to write"),
    ] = None,
    tools: Annotated[
        str,
        typer.Option(help="Comma-separated list of tools enabled for the agent"),
    ] = "read,bash,edit,write",
    trace_dir: Annotated[
        str | None,
        typer.Option(
            help="Directory for raw traces (defaults to <output>.traces)",
        ),
    ] = None,
    keep_traces: Annotated[
        bool,
        typer.Option(
            "--keep-traces/--no-keep-traces",
            help="Keep the raw trace directory after conversion",
        ),
    ] = True,
) -> None:
    """Generate agentic traces and convert them to speculator-format rows.

    Runs a Pi agent for each task in ``--data`` against the vLLM endpoint, then
    renders and tokenizes each trace into rows with ``input_ids``/``loss_mask``
    and writes them to ``--output`` as JSONL. Run ``setup-pi`` first to install
    the Pi runtime.
    """
    if ctx.invoked_subcommand is not None:
        return

    required = {
        "--model": model,
        "--data": data,
        "--output": output,
        "--render-endpoint": render_endpoint,
        "--pi-runtime": pi_runtime,
    }
    missing = [name for name, value in required.items() if not value]
    if missing:
        typer.echo(f"Missing option: {', '.join(missing)}")
        raise typer.Exit(1)

    assert model is not None  # noqa: S101
    assert data is not None  # noqa: S101
    assert output is not None  # noqa: S101
    assert render_endpoint is not None  # noqa: S101
    assert pi_runtime is not None  # noqa: S101

    _run_generate(
        model=model,
        data=data,
        output=output,
        render_endpoint=render_endpoint,
        pi_runtime=pi_runtime,
        pi_endpoint=pi_endpoint,
        pi_provider=pi_provider,
        pi_model=pi_model,
        workers=workers,
        timeout=timeout,
        seq_length=seq_length,
        limit=limit,
        max_samples=max_samples,
        tools=tools,
        trace_dir=trace_dir,
        keep_traces=keep_traces,
    )
