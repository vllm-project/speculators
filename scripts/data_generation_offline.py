#!/usr/bin/env python3
"""
Offline Hidden States Generation Pipeline

This script generates hidden states and saves them to disk for offline training.

Usage:
    python data_generation_offline.py \
        --model meta-llama/Llama-3.1-8B-Instruct \
        --preprocessed-data sharegpt \
        --output ./training_data \
        --max-samples 5000
"""

import argparse
import asyncio
import logging
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import openai
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


def parse_args():
    parser = argparse.ArgumentParser(description="Generate EAGLE training data offline")

    # Model arguments
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help=(
            "HuggingFace model ID or local path for target model "
            "(default auto select). For verification purposes only."
        ),
    )
    parser.add_argument(
        "--endpoint",
        type=str,
        default="http://localhost:8000/v1",
        help=(
            "The address of the vLLM instance to use for hidden states generation "
            "(default: 'http://localhost:8000/v1'). "
            "Note: the vLLM instance must be configured for hidden states extraction."
        ),
    )

    # Data arguments
    parser.add_argument(
        "--preprocessed-data",
        type=str,
        default="./output",
        help="Path to preprocessed dataset (dataset produced by prepare_data.py)"
        " (default: ./output)",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum number of samples to process (default: None, process all)",
    )

    # Output arguments
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help=(
            "Directory to generated hidden states files "
            "(default args.preprocessed_data / 'hidden_states')"
        ),
    )

    # Hidden states generation arguments
    parser.add_argument(
        "--concurrency",
        type=int,
        default=32,
        help=(
            "Number of active vLLM requests at a time. "
            "Note: number of async workers set to 2*concurrency"
        ),
    )
    parser.add_argument(
        "--validate-outputs",
        action="store_true",
        help=(
            "Load generated safetensor files and check output token ids match "
            "prompt tokens and hidden states seq_len matches num tokens"
        ),
    )
    parser.add_argument(
        "--request-timeout",
        type=float,
        default=DEFAULT_REQUEST_TIMEOUT,
        help=(
            "Timeout in seconds for each individual vLLM request "
            f"(default: {DEFAULT_REQUEST_TIMEOUT})"
        ),
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=DEFAULT_MAX_RETRIES,
        help=(
            "Maximum number of retry attempts per request on failure "
            f"(default: {DEFAULT_MAX_RETRIES})"
        ),
    )
    parser.add_argument(
        "--fail-on-error",
        action="store_true",
        help=(
            "Abort when a request fails after all retries. "
            "By default, failed samples are skipped."
        ),
    )
    parser.add_argument(
        "--max-consecutive-errors",
        type=int,
        default=None,
        help=(
            "Abort after this many consecutive sample failures (each sample "
            "already retried --max-retries times). Prevents silently churning "
            "through the entire dataset when the server is down. "
            "Ignored when --fail-on-error is set. "
            "(default: value of --concurrency)"
        ),
    )
    parser.add_argument(
        "--world-size",
        type=int,
        default=1,
        help=(
            "World size for multi-node data generation offline. IMPORTANT: this "
            "is the number of nodes (not the number of gpus). Defaults to 1"
        ),
    )
    parser.add_argument(
        "--rank",
        type=int,
        default=0,
        help=(
            "Rank for multi-node data generation offline. IMPORTANT: this is "
            "the node index, not an index for a gpu. Must be in range[0, world_size)."
            " Defaults to 0"
        ),
    )
    return parser.parse_args()


async def worker(  # noqa: C901
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

        # Drain remaining items quickly after cancellation
        if cancel_event.is_set():
            queue.task_done()
            continue

        target_hidden_states_path = hidden_states_output_dir / f"hs_{idx}.safetensors"

        try:
            async with vllm_semaphore:  # Limit number of active generate calls
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

            async with write_semaphore:  # Limit number of active disk writes
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
            stats["requests"] += 1
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
            if elapsed > 0 and stats["requests"] > 0:
                postfix["rps"] = f"{stats['requests'] / elapsed:.1f}"
                postfix["vllm"] = (
                    f"{stats['total_vllm_s'] / stats['requests'] * 1000:.0f}ms"
                )
                postfix["write"] = (
                    f"{stats['total_write_s'] / stats['requests'] * 1000:.0f}ms"
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

        # Check cancel_event while waiting for queue space to avoid
        # deadlocking when all workers have died.
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
        # Workers may be dead or draining — cancel any that are
        # still alive so we don't deadlock on sentinel puts.
        for w in workers:
            if not w.done():
                w.cancel()
    else:
        # Normal shutdown: send sentinel values so workers exit
        for _ in range(len(workers)):
            await queue.put(None)
    results = await asyncio.gather(*workers, return_exceptions=True)

    # Propagate the first real worker exception (skip CancelledError)
    for result in results:
        if isinstance(result, Exception) and not isinstance(
            result, asyncio.CancelledError
        ):
            raise result


async def generate_and_save_hidden_states(args, dataset):
    if args.output is None:
        hidden_states_dir = Path(args.preprocessed_data) / "hidden_states"
    else:
        hidden_states_dir = Path(args.output)
    hidden_states_dir.mkdir(parents=True, exist_ok=True)

    existing_file_indices = get_existing_hidden_state_indices(hidden_states_dir)
    num_samples = len(dataset)

    to_process = get_indices_to_process(
        num_samples,
        args.max_samples,
        existing_file_indices,
        args.world_size,
        args.rank,
    )
    if not to_process:
        return

    logger.info(f"Processing {len(to_process)} samples")

    queue: asyncio.Queue = asyncio.Queue(maxsize=args.concurrency * 4)
    vllm_semaphore = asyncio.Semaphore(args.concurrency)
    write_semaphore = asyncio.Semaphore(args.concurrency)

    skipped_indices: list[int] = []
    cancel_event = asyncio.Event()
    stats: dict[str, Any] = {
        "ok": 0,
        "errors": 0,
        "requests": 0,
        "total_vllm_s": 0.0,
        "total_write_s": 0.0,
        "start_time": time.perf_counter(),
    }

    max_consec = args.max_consecutive_errors
    if max_consec is None:
        max_consec = args.concurrency
    failure_tracker = _FailureTracker(max_consec) if not args.fail_on_error else None

    async with openai.AsyncOpenAI(
        base_url=args.endpoint, api_key="EMPTY", max_retries=0
    ) as client:
        list_models = await client.models.list()
        model_id = list_models.data[0].id
        if args.model and args.model != model_id:
            raise ValueError(
                f"An explicit model name was passed ({args.model}) which doesn't match"
                f" found model_id {model_id}."
                "Please make sure --endpoint is set to the correct vllm instance."
            )

        with tqdm(total=len(to_process)) as pbar:
            workers = [
                asyncio.create_task(
                    worker(
                        client,
                        model_id,
                        queue,
                        pbar,
                        vllm_semaphore,
                        write_semaphore,
                        hidden_states_dir,
                        args.validate_outputs,
                        args.request_timeout,
                        args.max_retries,
                        args.fail_on_error,
                        skipped_indices,
                        cancel_event,
                        failure_tracker,
                        stats,
                    )
                )
                for _ in range(args.concurrency * 2)
            ]

            await _feed_queue(to_process, dataset, queue, cancel_event)
            await _shutdown_workers(workers, queue, cancel_event)

    elapsed = time.perf_counter() - stats["start_time"]
    if stats["requests"] > 0:
        logger.info(
            "Timing: %.1fs elapsed, %.1f samples/s, "
            "avg vLLM request %.0f ms, avg file write %.0f ms",
            elapsed,
            stats["requests"] / elapsed if elapsed > 0 else 0,
            stats["total_vllm_s"] / stats["requests"] * 1000,
            stats["total_write_s"] / stats["requests"] * 1000,
        )

    num_saved = len(to_process) - len(skipped_indices)
    logger.info(f"Saved {num_saved} new data points to {args.output}")
    if skipped_indices:
        logger.warning(
            f"Skipped {len(skipped_indices)} samples due to errors: {skipped_indices}"
        )


def main():
    args = parse_args()
    if int(args.rank) < 0 or int(args.rank) >= int(args.world_size):
        raise ValueError("--rank must be in range [0, world_size)")
    setup_root_logger()

    logger.info("EAGLE Offline Data Generation")

    dataset = load_from_disk(args.preprocessed_data)

    try:
        asyncio.run(generate_and_save_hidden_states(args, dataset))
    except KeyboardInterrupt:
        sys.exit(130)
    except Exception:
        logger.exception("Data generation failed")
        sys.exit(1)

    logger.info("Data generation complete!")


if __name__ == "__main__":
    main()
