"""
Offline Hidden States Generation Pipeline

This module generates hidden states and saves them to disk for offline training.

Usage::

    speculators generate-offline-data \
        --model meta-llama/Llama-3.1-8B-Instruct \
        --preprocessed-data sharegpt \
        --output ./training_data \
        --max-samples 5000
"""

import asyncio
import logging
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Annotated, Any

import openai
import typer
from datasets import load_from_disk
from safetensors.torch import load_file
from tqdm import tqdm

from speculators.data_generation.offline import (
    check_hidden_states,
    get_existing_hidden_state_indices,
    get_indices_to_process,
)
from speculators.data_generation.vllm_client import (
    DEFAULT_MAX_RETRIES,
    DEFAULT_REQUEST_TIMEOUT,
    generate_hidden_states_async,
    wait_for_lock_async,
)
from speculators.train.data import build_client_item
from speculators.train.logger import setup_root_logger

logger = logging.getLogger(__name__)


class _FailureTracker:
    """Tracks consecutive sample failures across async workers.

    When the number of consecutive failures (with no successes in between)
    reaches ``threshold``, the tracker signals that the run should abort.
    Because asyncio is single-threaded, no locking is needed.
    """

    def __init__(self, threshold: int):
        self.threshold = threshold
        self._consecutive = 0

    def record_success(self) -> None:
        self._consecutive = 0

    def record_failure(self) -> bool:
        """Record a failure. Returns True when the threshold is reached."""
        self._consecutive += 1
        return self._consecutive >= self.threshold


async def _worker(  # noqa: C901
    client,
    model: str,
    queue: "asyncio.Queue[dict[str, Any]]",
    pbar: tqdm,
    vllm_semaphore: asyncio.Semaphore,
    write_semaphore: asyncio.Semaphore,
    hidden_states_output_dir: Path,
    validate_outputs: bool,
    request_timeout: float | None,
    max_retries: int,
    fail_on_error: bool,
    skipped_indices: list[int],
    cancel_event: asyncio.Event,
    failure_tracker: _FailureTracker | None,
    stats: dict[str, Any],
):
    """Worker that pulls items from queue and sends them to the vLLM endpoint."""
    while True:
        item = await queue.get()
        if item is None:
            queue.task_done()
            return

        idx = item["idx"]

        if cancel_event.is_set():
            queue.task_done()
            continue

        target_hidden_states_path = hidden_states_output_dir / f"hs_{idx}.safetensors"

        try:
            async with vllm_semaphore:
                t_vllm = time.perf_counter()
                hidden_states_path = await generate_hidden_states_async(
                    client,
                    model,
                    item,
                    timeout=request_timeout,
                    max_retries=max_retries,
                )
                vllm_s = time.perf_counter() - t_vllm
            lock_path = hidden_states_path + ".lock"
            if Path(lock_path).exists():  # noqa: ASYNC240
                await wait_for_lock_async(lock_path)

            async with write_semaphore:
                t_write = time.perf_counter()
                await asyncio.to_thread(
                    shutil.move, hidden_states_path, target_hidden_states_path
                )
                write_s = time.perf_counter() - t_write
                if validate_outputs:

                    def _load_and_check(
                        path=target_hidden_states_path,
                        tokens=item["input_ids"],
                    ):
                        loaded = load_file(path)
                        check_hidden_states(loaded, tokens)

                    await asyncio.to_thread(_load_and_check)
        except Exception as e:
            if fail_on_error:
                logger.exception(
                    "Fatal: sample %d aborted with --fail-on-error: %s", idx, e
                )
                logging.shutdown()
                os._exit(1)
            logger.warning("Skipping sample %d due to error: %s", idx, e)
            skipped_indices.append(idx)
            stats["errors"] += 1
            if failure_tracker is not None and failure_tracker.record_failure():
                cancel_event.set()
                raise RuntimeError(
                    f"Aborting: {failure_tracker.threshold} consecutive samples "
                    "errored out. The vLLM server may be unreachable."
                ) from e
        else:
            stats["ok"] += 1
            stats["total_vllm_s"] += vllm_s
            stats["total_write_s"] += write_s
            logger.debug(
                "Sample %d: vLLM %.0f ms, write %.0f ms",
                idx,
                vllm_s * 1000,
                write_s * 1000,
            )
            if failure_tracker is not None:
                failure_tracker.record_success()
        finally:
            elapsed = time.perf_counter() - stats["start_time"]
            postfix = {"ok": stats["ok"], "err": stats["errors"]}
            if elapsed > 0 and stats["ok"] > 0:
                postfix["rps"] = f"{stats['ok'] / elapsed:.1f}"
                postfix["vllm"] = f"{stats['total_vllm_s'] / stats['ok'] * 1000:.0f}ms"
                postfix["write"] = (
                    f"{stats['total_write_s'] / stats['ok'] * 1000:.0f}ms"
                )
            pbar.set_postfix(postfix, refresh=False)
            pbar.update(1)
            queue.task_done()


async def _feed_queue(to_process, dataset, queue, cancel_event):
    """Feed dataset items into the worker queue, respecting cancellation."""
    for i in to_process:
        if cancel_event.is_set():
            break

        dataset_item = dataset[i]
        client_item = build_client_item(dataset_item) | {"idx": i}

        while not cancel_event.is_set():
            try:
                queue.put_nowait(client_item)
                break
            except asyncio.QueueFull:
                await asyncio.sleep(0.1)


async def _shutdown_workers(workers, queue, cancel_event):
    """Shut down workers and propagate the first real exception."""
    logger.info("Waiting for remaining file saves to complete...")
    if cancel_event.is_set():
        for w in workers:
            if not w.done():
                w.cancel()
    else:
        for _ in range(len(workers)):
            await queue.put(None)
    results = await asyncio.gather(*workers, return_exceptions=True)

    for result in results:
        if isinstance(result, Exception) and not isinstance(
            result, asyncio.CancelledError
        ):
            raise result


async def _generate_and_save_hidden_states(
    model: str | None,
    endpoint: str,
    preprocessed_data: str,
    output: str | None,
    max_samples: int | None,
    concurrency: int,
    validate_outputs: bool,
    request_timeout: float,
    max_retries: int,
    fail_on_error: bool,
    max_consecutive_errors: int | None,
    world_size: int,
    rank: int,
):
    dataset = load_from_disk(preprocessed_data)

    if output is None:
        hidden_states_dir = Path(preprocessed_data) / "hidden_states"
    else:
        hidden_states_dir = Path(output)
    hidden_states_dir.mkdir(parents=True, exist_ok=True)

    existing_file_indices = get_existing_hidden_state_indices(hidden_states_dir)
    num_samples = len(dataset)

    to_process = get_indices_to_process(
        num_samples,
        max_samples,
        existing_file_indices,
        world_size,
        rank,
    )
    if not to_process:
        return

    logger.info(f"Processing {len(to_process)} samples")

    queue: asyncio.Queue = asyncio.Queue(maxsize=concurrency * 4)
    vllm_semaphore = asyncio.Semaphore(concurrency)
    write_semaphore = asyncio.Semaphore(concurrency)

    skipped_indices: list[int] = []
    cancel_event = asyncio.Event()
    stats: dict[str, Any] = {
        "ok": 0,
        "errors": 0,
        "total_vllm_s": 0.0,
        "total_write_s": 0.0,
        "start_time": time.perf_counter(),
    }

    max_consec = max_consecutive_errors
    if max_consec is None:
        max_consec = concurrency
    failure_tracker = _FailureTracker(max_consec) if not fail_on_error else None

    async with openai.AsyncOpenAI(
        base_url=endpoint, api_key="EMPTY", max_retries=0
    ) as client:
        list_models = await client.models.list()
        if not list_models.data:
            raise RuntimeError(
                "No models found on the vLLM server. "
                "Make sure the server is fully loaded."
            )
        model_id = list_models.data[0].id
        if model and model != model_id:
            raise ValueError(
                f"An explicit model name was passed ({model}) which doesn't match"
                f" found model_id {model_id}."
                "Please make sure --endpoint is set to the correct vllm instance."
            )

        with tqdm(total=len(to_process)) as pbar:
            workers = [
                asyncio.create_task(
                    _worker(
                        client,
                        model_id,
                        queue,
                        pbar,
                        vllm_semaphore,
                        write_semaphore,
                        hidden_states_dir,
                        validate_outputs,
                        request_timeout,
                        max_retries,
                        fail_on_error,
                        skipped_indices,
                        cancel_event,
                        failure_tracker,
                        stats,
                    )
                )
                for _ in range(concurrency * 2)
            ]

            await _feed_queue(to_process, dataset, queue, cancel_event)
            await _shutdown_workers(workers, queue, cancel_event)

    elapsed = time.perf_counter() - stats["start_time"]
    if stats["ok"] > 0:
        logger.info(
            "Timing: %.1fs elapsed, %.1f samples/s, "
            "avg vLLM request %.0f ms, avg file write %.0f ms",
            elapsed,
            stats["ok"] / elapsed if elapsed > 0 else 0,
            stats["total_vllm_s"] / stats["ok"] * 1000,
            stats["total_write_s"] / stats["ok"] * 1000,
        )

    num_saved = len(to_process) - len(skipped_indices)
    logger.info(f"Saved {num_saved} new data points to {hidden_states_dir}")
    if skipped_indices:
        logger.warning(
            f"Skipped {len(skipped_indices)} samples due to errors: {skipped_indices}"
        )


def generate_offline_data(
    model: Annotated[
        str | None,
        typer.Option(
            help=(
                "HuggingFace model ID or local path for target model "
                "(default auto select). For verification purposes only."
            ),
        ),
    ] = None,
    endpoint: Annotated[
        str,
        typer.Option(
            help=(
                "The address of the vLLM instance to use for hidden states "
                "generation. The instance must be configured for hidden states "
                "extraction."
            ),
        ),
    ] = "http://localhost:8000/v1",
    preprocessed_data: Annotated[
        str,
        typer.Option(
            help="Path to preprocessed dataset (produced by prepare-data)",
        ),
    ] = "./output",
    max_samples: Annotated[
        int | None,
        typer.Option(help="Maximum number of samples to process"),
    ] = None,
    output: Annotated[
        str | None,
        typer.Option(
            help=(
                "Directory to save generated hidden states files "
                "(default: {preprocessed-data}/hidden_states)"
            ),
        ),
    ] = None,
    concurrency: Annotated[
        int,
        typer.Option(
            help=(
                "Number of active vLLM requests at a time. "
                "Note: number of async workers set to 2*concurrency"
            ),
        ),
    ] = 32,
    validate_outputs: Annotated[
        bool,
        typer.Option(
            "--validate-outputs",
            help=(
                "Load generated safetensor files and check output token ids "
                "match prompt tokens and hidden states seq_len matches num tokens"
            ),
        ),
    ] = False,
    request_timeout: Annotated[
        float,
        typer.Option(
            help="Timeout in seconds for each individual vLLM request",
        ),
    ] = DEFAULT_REQUEST_TIMEOUT,
    max_retries: Annotated[
        int,
        typer.Option(
            help="Maximum number of retry attempts per request on failure",
        ),
    ] = DEFAULT_MAX_RETRIES,
    fail_on_error: Annotated[
        bool,
        typer.Option(
            "--fail-on-error",
            help=(
                "Abort when a request fails after all retries. "
                "By default, failed samples are skipped."
            ),
        ),
    ] = False,
    max_consecutive_errors: Annotated[
        int | None,
        typer.Option(
            help=(
                "Abort after this many consecutive sample failures (each sample "
                "already retried --max-retries times). Prevents silently churning "
                "through the entire dataset when the server is down. "
                "Ignored when --fail-on-error is set. "
                "(default: value of --concurrency)"
            ),
        ),
    ] = None,
    world_size: Annotated[
        int,
        typer.Option(
            help=(
                "World size for multi-node data generation offline. "
                "This is the number of nodes (not GPUs)."
            ),
        ),
    ] = 1,
    rank: Annotated[
        int,
        typer.Option(
            help=(
                "Rank for multi-node data generation offline. "
                "This is the node index, not a GPU index. "
                "Must be in range [0, world_size)."
            ),
        ),
    ] = 0,
) -> None:
    """Generate hidden states offline from a vLLM server.

    Connects to a running vLLM instance, sends preprocessed samples, and saves
    the extracted hidden states to disk for offline training.
    """
    if concurrency < 1:
        raise typer.BadParameter("--concurrency must be >= 1")
    if rank < 0 or rank >= world_size:
        raise typer.BadParameter("--rank must be in range [0, world_size)")
    setup_root_logger()

    logger.info("EAGLE Offline Data Generation")

    try:
        asyncio.run(
            _generate_and_save_hidden_states(
                model=model,
                endpoint=endpoint,
                preprocessed_data=preprocessed_data,
                output=output,
                max_samples=max_samples,
                concurrency=concurrency,
                validate_outputs=validate_outputs,
                request_timeout=request_timeout,
                max_retries=max_retries,
                fail_on_error=fail_on_error,
                max_consecutive_errors=max_consecutive_errors,
                world_size=world_size,
                rank=rank,
            )
        )
    except KeyboardInterrupt:
        sys.exit(130)
    except Exception:
        logger.exception("Data generation failed")
        sys.exit(1)

    logger.info("Data generation complete!")
