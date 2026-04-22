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
from pathlib import Path
from typing import Any

import openai
from datasets import load_from_disk
from safetensors import safe_open
from tqdm import tqdm

from speculators.data_generation.vllm_client import (
    DEFAULT_MAX_RETRIES,
    DEFAULT_REQUEST_TIMEOUT,
    generate_hidden_states_async,
)
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
            "HuggingFace model ID or local path for target model (default auto select)."
            "For verification purposes only."
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
        "--layer-ids",
        type=int,
        nargs="+",
        default=None,
        help=(
            "List of layer IDs from which to capture hidden states "
            "(default: auto-select)"
        ),
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=32,
        help=(
            "Number of active vLLM requests at a time."
            "Note: number of async workers set to 2*concurrency"
        ),
    )
    parser.add_argument(
        "--validate-outputs",
        action="store_true",
        help=(
            "Load generated safetensor files and check output token ids match prompt"
            " tokens and hidden states seq_len matches num tokens"
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

    # Processing arguments
    parser.add_argument(
        "--start-idx",
        type=int,
        default=0,
        help="Starting index for output files (default: 0)",
    )
    return parser.parse_args()


def get_existing_hidden_state_indices(output_path: Path) -> list[int]:
    """Find existing `hs_i.safetensors` files (where i is the file index)"""

    existing_file_indices = []

    if not output_path.exists():
        return existing_file_indices

    for file_path in output_path.iterdir():
        if file_path.name.startswith("hs_") and file_path.name.endswith(".safetensors"):
            index_str = file_path.stem[3:]  # Remove "hs_" prefix
            try:
                file_index = int(index_str)
                existing_file_indices.append(file_index)
            except ValueError:
                continue

    return sorted(existing_file_indices)


def get_indices_to_process(
    num_samples: int, max_samples: int | None, existing: list[int]
) -> list[int]:
    """Determines which indices should be processed. If max_samples is None
    returns all dataset indices not in existing. Otherwise gets the first
    `max_samples - len(existing)` samples not already in existing.

    Args:
        num_samples: Total size of preprocessed dataset
        max_samples: (Optional) limit for number of samples to process
        existing: list of ids that have already been processed

    Returns:
        list of dataset indices to process
    """

    if len(existing) >= num_samples:
        logger.info("All samples already processed!")
        return []
    if max_samples and len(existing) >= max_samples:
        logger.info("At least max_samples already processed!")
        return []

    if len(existing) > 0:
        logger.info(f"Found {len(existing)} existing samples.")

    existing_s = set(existing)
    if max_samples is None:
        return [i for i in range(num_samples) if i not in existing_s]

    num_remaining = min(max_samples, num_samples) - len(existing)
    to_process = []
    cur = 0
    while num_remaining > 0 and cur < num_samples:
        if cur not in existing_s:
            to_process.append(cur)
            num_remaining -= 1

        cur += 1

    return to_process


def check_safetensors_file(path: Path, tokens: list[int]):
    with safe_open(path, "pt") as f:
        t_ids = f.get_tensor("token_ids").tolist()
        if t_ids != tokens:
            raise ValueError(
                f"Token ids in {path} don't match expected token ids {tokens}"
            )

        hs_slice = f.get_slice("hidden_states")
        hs_shape = list(hs_slice.get_shape())
        if len(tokens) != hs_shape[0]:
            raise ValueError(
                f"Sequence length of hidden states {hs_shape[0]} in {path}"
                f" doesn't match num tokens {len(tokens)}"
            )


async def worker(
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

        input_ids = item["input_ids"].tolist()

        target_hidden_states_path = hidden_states_output_dir / f"hs_{idx}.safetensors"

        try:
            async with vllm_semaphore:  # Limit number of active generate calls
                hidden_states_path = await generate_hidden_states_async(
                    client,
                    model,
                    input_ids,
                    timeout=request_timeout,
                    max_retries=max_retries,
                )
            async with write_semaphore:  # Limit number of active disk writes
                await asyncio.to_thread(
                    shutil.move, hidden_states_path, target_hidden_states_path
                )
                if validate_outputs:
                    await asyncio.to_thread(
                        check_safetensors_file, target_hidden_states_path, input_ids
                    )
        except Exception as e:
            if fail_on_error:
                logger.exception(
                    "Fatal: sample %d aborted with --fail-on-error: %s", idx, e
                )
                logging.shutdown()
                os._exit(1)
            logger.warning("Skipping sample %d due to error: %s", idx, e)
            skipped_indices.append(idx)
            if failure_tracker is not None and failure_tracker.record_failure():
                cancel_event.set()
                raise RuntimeError(
                    f"Aborting: {failure_tracker.threshold} consecutive samples "
                    "errored out. The vLLM server may be unreachable."
                ) from e
        else:
            if failure_tracker is not None:
                failure_tracker.record_success()
        finally:
            pbar.update(1)
            queue.task_done()


async def _feed_queue(to_process, dataset, queue, cancel_event):
    """Feed dataset items into the worker queue, respecting cancellation."""
    for i in to_process:
        if cancel_event.is_set():
            break
        item = dataset[i]
        # Check cancel_event while waiting for queue space to avoid
        # deadlocking when all workers have died.
        while not cancel_event.is_set():
            try:
                queue.put_nowait({"idx": i, "input_ids": item["input_ids"]})
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
        num_samples, args.max_samples, existing_file_indices
    )
    if not to_process:
        return

    logger.info(f"Processing {len(to_process)} samples")

    queue: asyncio.Queue = asyncio.Queue(maxsize=args.concurrency * 4)
    vllm_semaphore = asyncio.Semaphore(args.concurrency)
    write_semaphore = asyncio.Semaphore(args.concurrency)

    skipped_indices: list[int] = []
    cancel_event = asyncio.Event()

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
                "found model_id {model_id}."
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
                    )
                )
                for _ in range(args.concurrency * 2)
            ]

            await _feed_queue(to_process, dataset, queue, cancel_event)
            await _shutdown_workers(workers, queue, cancel_event)

    num_saved = len(to_process) - len(skipped_indices)
    logger.info(f"Saved {num_saved} new data points to {args.output}")
    if skipped_indices:
        logger.warning(
            f"Skipped {len(skipped_indices)} samples due to errors: {skipped_indices}"
        )


def main():
    args = parse_args()
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
