import asyncio
import fcntl
import functools
import logging
import os
import time
from typing import TYPE_CHECKING, Any, TypedDict

import openai
from openai.types.chat import ChatCompletion, ChatCompletionMessageParam
from openai.types.completion import Completion
from typing_extensions import NotRequired

if TYPE_CHECKING:
    from collections.abc import Coroutine

logger = logging.getLogger(__name__)

DEFAULT_REQUEST_TIMEOUT = 120  # seconds
DEFAULT_MAX_RETRIES = 3
RETRY_BACKOFF_BASE = 2  # seconds


class InvalidResponseError(Exception):
    pass


def _handle_retry_error(
    error: Exception, attempt: int, total_attempts: int
) -> float | None:
    """Handle a retry-eligible error.

    Returns backoff seconds if the caller should retry, or ``None`` on the
    final attempt.  Raises ``InvalidResponseError`` immediately.
    """
    if isinstance(error, InvalidResponseError):
        raise error
    if attempt < total_attempts:
        backoff = RETRY_BACKOFF_BASE ** attempt
        logger.warning(
            "Request aborted (attempt %d/%d): %s. Retrying in %ds...",
            attempt,
            total_attempts,
            error,
            backoff,
        )
        return backoff
    logger.error("Request timed out after %d attempts: %s", total_attempts, error)
    return None


def with_retries(fn):
    """Decorator that adds retry logic with exponential backoff.

    The decorated function gains a ``max_retries`` keyword argument
    (default ``DEFAULT_MAX_RETRIES``). ``InvalidResponseError`` is never
    retried. Works for both sync and async functions.
    """
    if asyncio.iscoroutinefunction(fn):

        @functools.wraps(fn)
        async def async_wrapper(*args, max_retries=DEFAULT_MAX_RETRIES, **kwargs):
            total_attempts = max_retries + 1
            last_error: Exception | None = None
            for attempt in range(1, total_attempts + 1):
                try:
                    return await fn(*args, **kwargs)
                except Exception as e:
                    last_error = e
                    backoff = _handle_retry_error(e, attempt, total_attempts)
                    if backoff is not None:
                        await asyncio.sleep(backoff)
            raise last_error  # type: ignore[misc]

        return async_wrapper

    @functools.wraps(fn)
    def sync_wrapper(*args, max_retries=DEFAULT_MAX_RETRIES, **kwargs):
        total_attempts = max_retries + 1
        last_error: Exception | None = None
        for attempt in range(1, total_attempts + 1):
            try:
                return fn(*args, **kwargs)
            except Exception as e:
                last_error = e
                backoff = _handle_retry_error(e, attempt, total_attempts)
                if backoff is not None:
                    time.sleep(backoff)
        raise last_error  # type: ignore[misc]

    return sync_wrapper


class ClientItem(TypedDict):
    input_ids: list[int]
    """The input token IDs."""

    messages: NotRequired[list[ChatCompletionMessageParam]]
    """If provided, pass `messages` to Chat Completions API
    instead of passing `token_ids` to Completions API."""


def extract_output(
    response: Completion | ChatCompletion,
    token_ids: list[int],
) -> str:
    """Extract hidden states path from vLLM response.

    Merged version:
    - Supports both Completion and ChatCompletion (from incoming)
    - Validates prompt_token_ids match (from HEAD)
    """
    # Extract prompt_token_ids based on response type
    if isinstance(response, Completion):
        prompt_token_ids = getattr(response.choices[0], "prompt_token_ids", None)
    else:
        # ChatCompletion
        prompt_token_ids = getattr(response, "prompt_token_ids", None)

    if prompt_token_ids is None:
        raise InvalidResponseError("Response missing prompt_token_ids")

    if prompt_token_ids != token_ids:
        raise InvalidResponseError(
            f"Prompt token IDs mismatch: expected {token_ids}, got {prompt_token_ids}"
        )

    # Extract hidden_states_path from kv_transfer_params
    kv_transfer_params = getattr(response, "kv_transfer_params", None)
    if kv_transfer_params is None:
        raise InvalidResponseError("Response missing kv_transfer_params")

    hidden_states_path = kv_transfer_params.get("hidden_states_path")
    if not hidden_states_path:
        raise InvalidResponseError("Response missing hidden_states_path")

    return hidden_states_path


async def _poll_lock_async(fd, poll_interval):
    """Async poll for file lock."""
    while True:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            return
        except BlockingIOError:
            await asyncio.sleep(poll_interval)


async def wait_for_lock_async(lock_path, timeout=10.0, poll_interval=0.1):
    """Async wait for and release file lock."""
    fd = os.open(lock_path, os.O_RDONLY)
    try:
        await asyncio.wait_for(_poll_lock_async(fd, poll_interval), timeout=timeout)
    except BaseException:
        os.close(fd)
        raise
    os.close(fd)
    os.remove(lock_path)


def wait_for_lock(lock_path, timeout=10.0, poll_interval=0.1):
    """Sync wait for and release file lock."""
    fd = os.open(lock_path, os.O_RDONLY)
    try:
        deadline = time.monotonic() + timeout
        while True:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if time.monotonic() >= deadline:
                    raise TimeoutError(
                        f"Timed out waiting for lock: {lock_path}"
                    ) from None
                time.sleep(poll_interval)
    except BaseException:
        os.close(fd)
        raise
    os.close(fd)
    os.remove(lock_path)


@with_retries
async def generate_hidden_states_async(
    client: openai.AsyncClient,
    model: str,
    client_item: ClientItem,
    *,
    timeout: float | None = DEFAULT_REQUEST_TIMEOUT,
) -> str:
    """Generate hidden states for a single sample (async).

    Uses Completions API for text-only, Chat Completions API for multimodal.
    """
    token_ids = client_item["input_ids"]
    messages = client_item.get("messages")

    coro: Coroutine[Any, Any, Completion | ChatCompletion]
    if messages is None:
        # Text-only: use Completions API
        coro = client.completions.create(
            model=model,
            prompt=token_ids,
            max_tokens=1,
            extra_body={"return_token_ids": True},
            timeout=timeout,
        )
    else:
        # Multimodal: use Chat Completions API
        coro = client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=1,
            extra_body={"add_generation_prompt": False, "return_token_ids": True},
            timeout=timeout,
        )

    res: Completion | ChatCompletion
    if timeout is not None:
        res = await asyncio.wait_for(coro, timeout=timeout)
    else:
        res = await coro

    return extract_output(res, token_ids)


@with_retries
async def generate_hidden_states_multimodal_async(
    client: openai.AsyncClient,
    model: str,
    messages: list[dict[str, Any]],
    timeout: float | None = DEFAULT_REQUEST_TIMEOUT,
) -> str:
    """Generate hidden states for multimodal chat messages (async).

    Convenience wrapper for multimodal-only generation.
    """
    client_item: ClientItem = {"input_ids": [], "messages": messages}  # type: ignore
    return await generate_hidden_states_async(
        client, model, client_item, timeout=timeout
    )


@with_retries
def generate_hidden_states(
    client: openai.Client,
    model: str,
    client_item: ClientItem,
    *,
    timeout: float | None = DEFAULT_REQUEST_TIMEOUT,
) -> str:
    """Generate hidden states for a single sample (sync).

    Uses Completions API for text-only, Chat Completions API for multimodal.
    """
    token_ids = client_item["input_ids"]
    messages = client_item.get("messages")

    res: Completion | ChatCompletion
    if messages is None:
        # Text-only: use Completions API
        res = client.completions.create(
            model=model,
            prompt=token_ids,
            max_tokens=1,
            extra_body={"return_token_ids": True},
            timeout=timeout,
        )
    else:
        # Multimodal: use Chat Completions API
        res = client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=1,
            extra_body={"add_generation_prompt": False, "return_token_ids": True},
            timeout=timeout,
        )

    return extract_output(res, token_ids)


@with_retries
def generate_hidden_states_multimodal(
    client: openai.Client,
    model: str,
    messages: list[dict[str, Any]],
    timeout: float | None = DEFAULT_REQUEST_TIMEOUT,
) -> str:
    """Generate hidden states for multimodal chat messages (sync).

    Convenience wrapper for multimodal-only generation.
    """
    client_item: ClientItem = {"input_ids": [], "messages": messages}  # type: ignore
    return generate_hidden_states(client, model, client_item, timeout=timeout)
