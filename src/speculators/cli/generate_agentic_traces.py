"""
Generate agentic traces with Pi and convert them to speculator-format rows.

For each SWE-smith-style dataset instance, a Pi agent runs inside the task
container against a vLLM endpoint, producing a raw execution trace. Each trace
is then rendered and tokenized into speculator-format rows (``input_ids`` and
``loss_mask``) ready for training.

Usage::

    speculators generate-agentic-traces \\
        --model meta-llama/Llama-3.1-8B-Instruct \\
        --data ./swe_smith_tasks.jsonl \\
        --render-endpoint http://localhost:8000 \\
        --pi-runtime ./pi-runtime \\
        --output ./traces.jsonl
"""

import json
import logging
import shutil
from pathlib import Path
from typing import Annotated, Any

import typer

from speculators.data_generation.logging_utils import PipelineLogger

log = PipelineLogger(__name__)
logger = logging.getLogger(__name__)


def generate_agentic_traces(  # noqa: PLR0917
    model: Annotated[
        str,
        typer.Option(
            help="HuggingFace model ID or local path used for tokenizer/processor",
        ),
    ],
    data: Annotated[
        list[str],
        typer.Option(
            "--data",
            help=(
                "SWE-smith-style task dataset: local JSON/JSONL file or directory, "
                "HuggingFace id, or hf: prefix (repeatable)"
            ),
        ),
    ],
    output: Annotated[
        str,
        typer.Option(help="Output JSONL path, e.g. traces.jsonl"),
    ],
    render_endpoint: Annotated[
        str,
        typer.Option(
            help=(
                "Base URL of a running vLLM server used to render traces "
                "(e.g. http://localhost:8000)"
            ),
        ),
    ],
    pi_runtime: Annotated[
        str,
        typer.Option(help="Path to the prepared Pi runtime"),
    ],
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
        str,
        typer.Option(help="Model name exposed to Pi by the endpoint"),
    ] = "qwen3.8-27b",
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
    renders and tokenizes each trace into rows with ``input_ids``/``loss_mask``,
    writing them to ``--output`` as JSONL.
    """
    # Imported here so `--help` works without the (optional) agentic runtime.
    from speculators.data_generation.agentic_runner import (  # noqa: PLC0415
        generate_traces,
        load_trace_tasks,
    )
    from speculators.data_generation.agentic_traces import (  # noqa: PLC0415
        rows_from_pi_trace,
    )
    from speculators.data_generation.preprocessing import (  # noqa: PLC0415
        load_processor,
    )

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    output_path = Path(output)
    trace_path = Path(trace_dir) if trace_dir is not None else Path(f"{output}.traces")

    log.section("Generating agentic traces")
    log.config(
        {
            "Target Model": model,
            "Datasets": data,
            "Output": str(output_path),
            "Trace Dir": str(trace_path),
            "Render Endpoint": render_endpoint,
            "Pi Endpoint": pi_endpoint,
            "Pi Model": pi_model,
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
        pi_model=pi_model,
        tools=tools,
        timeout=timeout,
    )

    succeeded = [r for r in results if not r.get("error") and r.get("trace_path")]
    failed = len(results) - len(succeeded)
    log.info(f"{len(succeeded)} trace(s) succeeded, {failed} failed")

    processor = load_processor(model)

    rows: list[dict[str, Any]] = []
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
            logger.warning(
                "Failed to convert trace for %s: %r", result.get("instance_id"), exc
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
        shutil.rmtree(trace_path, ignore_errors=True)

    log.section("Done")
    log.success(f"Wrote {len(rows)} row(s) from {len(succeeded)} trace(s)")
    log.info(f"Output: {output_path}")
