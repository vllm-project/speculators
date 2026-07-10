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
import math
import os
import sys
from pathlib import Path
from typing import Any

import openai
from datasets import load_from_disk
from safetensors.torch import load_file
from tqdm import tqdm

from speculators.data_generation.offline import (
    InvalidHiddenStateCacheError,
    align_hidden_states_to_tokens,
    atomic_move_safetensors,
    atomic_save_safetensors,
    durable_unlink_safetensors,
    get_existing_hidden_state_indices,
    get_indices_to_process,
    hidden_states_file_sha256,
    validate_generated_source_ownership,
    validate_hidden_states_file_contents,
    validate_hidden_states_path,
    validate_hidden_states_root,
)
from speculators.data_generation.vllm_client import (
    DEFAULT_MAX_RETRIES,
    DEFAULT_REQUEST_TIMEOUT,
    generate_hidden_states_async,
    wait_for_lock_async,
)
from speculators.train.data import build_client_item
from speculators.train.logger import setup_root_logger
from speculators.utils.argparse_utils import nonnegative_int

logger = logging.getLogger(__name__)


def _positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"expected a positive integer, got {value!r}"
        ) from None
    if parsed <= 0:
        raise argparse.ArgumentTypeError(
            f"expected a positive integer, got {value!r}"
        )
    return parsed


def _positive_finite_float(value: str) -> float:
    try:
        parsed = float(value)
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"expected a finite positive number, got {value!r}"
        ) from None
    if not math.isfinite(parsed) or parsed <= 0:
        raise argparse.ArgumentTypeError(
            f"expected a finite positive number, got {value!r}"
        )
    return parsed


def _align_and_write_hidden_states(
    source_path: Path,
    target_path: Path,
    tokens: list[int],
    *,
    source_root: Path,
    target_root: Path,
    allow_prefix_truncation: bool,
    validate_outputs: bool,
) -> None:
    source_path = validate_hidden_states_path(source_path, source_root)
    source_sha256 = hidden_states_file_sha256(
        source_path,
        allowed_root=source_root,
    )
    # Even the fast path must reject arbitrary/corrupt files before they become a
    # canonical cache entry. Exact token matching is cheap because the structure
    # gate already materializes token_ids; --validate-outputs only controls the
    # full hidden-state value/finite scan.
    validate_hidden_states_file_contents(
        source_path,
        source_root,
        expected_tokens=None if allow_prefix_truncation else tokens,
    )
    source_sha256_after_validation = hidden_states_file_sha256(
        source_path,
        allowed_root=source_root,
    )
    if source_sha256_after_validation != source_sha256:
        raise RuntimeError(
            "Generated hidden-state source changed while it was being validated"
        )
    validate_generated_source_ownership(
        source_path,
        target_path,
        source_root=source_root,
        target_root=target_root,
        allow_current_target=allow_prefix_truncation,
    )
    if not allow_prefix_truncation and not validate_outputs:
        if source_path != target_path:
            atomic_move_safetensors(
                source_path,
                target_path,
                source_root=source_root,
                target_root=target_root,
                expected_source_sha256=source_sha256,
                expected_tokens=tokens,
            )
        return

    source_path = validate_hidden_states_path(source_path, source_root)
    loaded = load_file(source_path)
    source_sha256_after_load = hidden_states_file_sha256(
        source_path,
        allowed_root=source_root,
    )
    if source_sha256_after_load != source_sha256:
        raise RuntimeError(
            "Generated hidden-state source changed while it was being loaded"
        )
    aligned, truncated = align_hidden_states_to_tokens(
        loaded,
        tokens,
        allow_prefix_truncation=allow_prefix_truncation,
    )
    validate_generated_source_ownership(
        source_path,
        target_path,
        source_root=source_root,
        target_root=target_root,
        allow_current_target=truncated,
    )
    if truncated:
        atomic_save_safetensors(
            {key: value.contiguous() for key, value in aligned.items()},
            target_path,
            allowed_root=target_root,
            allow_replace=source_path == target_path,
            expected_existing_sha256=(
                source_sha256 if source_path == target_path else None
            ),
        )
        if source_path != target_path:
            durable_unlink_safetensors(
                source_path,
                allowed_root=source_root,
                expected_sha256=source_sha256,
            )
    elif source_path != target_path:
        atomic_move_safetensors(
            source_path,
            target_path,
            source_root=source_root,
            target_root=target_root,
            expected_source_sha256=source_sha256,
            expected_tokens=tokens,
        )


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
    parser.add_argument(
        "--source-hidden-states-root",
        type=str,
        default=None,
        help=(
            "Existing allowed root for hidden-state source paths returned by vLLM. "
            "Required when vLLM writes outside --output; defaults to the resolved "
            "output hidden-states directory."
        ),
    )

    # Hidden states generation arguments
    parser.add_argument(
        "--concurrency",
        type=_positive_int,
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
        type=_positive_finite_float,
        default=DEFAULT_REQUEST_TIMEOUT,
        help=(
            "Timeout in seconds for each individual vLLM request "
            f"(default: {DEFAULT_REQUEST_TIMEOUT})"
        ),
    )
    parser.add_argument(
        "--max-retries",
        type=nonnegative_int,
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


async def worker(
    client,
    model: str,
    queue: "asyncio.Queue[dict[str, Any]]",
    pbar: tqdm,
    vllm_semaphore: asyncio.Semaphore,
    write_semaphore: asyncio.Semaphore,
    hidden_states_output_dir: Path,
    source_hidden_states_root: Path,
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

        target_hidden_states_path = hidden_states_output_dir / f"hs_{idx}.safetensors"

        try:
            async with vllm_semaphore:  # Limit number of active generate calls
                hidden_states_path = await generate_hidden_states_async(
                    client,
                    model,
                    item,
                    timeout=request_timeout,
                    max_retries=max_retries,
                )
            source_path = validate_hidden_states_path(
                hidden_states_path,
                source_hidden_states_root,
                require_exists=False,
            )
            lock_path = str(source_path) + ".lock"
            if Path(lock_path).is_symlink():  # noqa: ASYNC240
                raise ValueError(f"Hidden-state lock path is a symlink: {lock_path}")
            if Path(lock_path).exists():  # noqa: ASYNC240
                await wait_for_lock_async(lock_path)

            source_path = validate_hidden_states_path(
                source_path, source_hidden_states_root
            )

            async with write_semaphore:  # Limit number of active disk writes
                allow_prefix_truncation = "messages" in item

                await asyncio.to_thread(
                    _align_and_write_hidden_states,
                    source_path,
                    target_hidden_states_path,
                    item["input_ids"],
                    source_root=source_hidden_states_root,
                    target_root=hidden_states_output_dir,
                    allow_prefix_truncation=allow_prefix_truncation,
                    validate_outputs=validate_outputs,
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
    hidden_states_dir = validate_hidden_states_root(hidden_states_dir)
    source_hidden_states_root = validate_hidden_states_root(
        hidden_states_dir
        if args.source_hidden_states_root is None
        else args.source_hidden_states_root
    )

    num_samples = len(dataset)

    def expected_tokens_for_index(file_index: int) -> list[int]:
        if file_index < 0 or file_index >= num_samples:
            raise InvalidHiddenStateCacheError(
                f"cache index {file_index} is outside the current dataset"
            )
        input_ids = dataset[file_index]["input_ids"]
        if hasattr(input_ids, "tolist"):
            input_ids = input_ids.tolist()
        return [int(token) for token in input_ids]

    existing_file_indices = get_existing_hidden_state_indices(
        hidden_states_dir,
        expected_tokens_for_index=expected_tokens_for_index,
    )
    if "messages" in dataset.column_names:
        logger.info("Detected multimodal preprocessed dataset")

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
                        source_hidden_states_root,
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
    logger.info(f"Saved {num_saved} new data points to {hidden_states_dir}")
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
