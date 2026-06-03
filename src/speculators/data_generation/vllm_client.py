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
        backoff = RETRY_BACKOFF_BASE**attempt
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


def extract_output(
    response: Completion | ChatCompletion,
    token_ids: list[int],
    *,
    trust_server_token_ids: bool = False,
) -> str | tuple[str, list[int]]:
    """Extract hidden-states file path from vLLM response.

    Args:
        response: vLLM Completion or ChatCompletion response.
        token_ids: The client-side token IDs sent with the request.
        trust_server_token_ids: If True (used for multimodal Chat Completions),
            skip the strict token-id equality check and return the server's
            ``prompt_token_ids`` alongside the file path.  The caller must use
            the returned token IDs as the authoritative sequence (they match
            the hidden states positionally).

    Returns:
        - When ``trust_server_token_ids=False``: the hidden-states file path (str).
        - When ``trust_server_token_ids=True``: a tuple of
          (hidden-states file path, server prompt_token_ids).

    Strict check rationale (when trust_server_token_ids=False):
        Hidden states are position-indexed.  If the server's prompt
        tokenization differs from the client's by even one token, every
        subsequent hidden-state vector is mis-aligned and the resulting
        training data is silently corrupted.
    """
    if isinstance(response, Completion):
        prompt_token_ids = getattr(response.choices[0], "prompt_token_ids", None)
    else:
        prompt_token_ids = getattr(response, "prompt_token_ids", None)

    if prompt_token_ids is None:
        raise InvalidResponseError("Response missing prompt_token_ids")

    if not trust_server_token_ids and prompt_token_ids != token_ids:
        raise InvalidResponseError(
            f"Prompt token IDs mismatch: expected {len(token_ids)} tokens, "
            f"got {len(prompt_token_ids)} tokens"
        )

    kv_transfer_params = getattr(response, "kv_transfer_params", None)
    if kv_transfer_params is None:
        raise InvalidResponseError("Response missing kv_transfer_params")

    path = kv_transfer_params.get("hidden_states_path")

    if trust_server_token_ids:
        return path, prompt_token_ids
    return path


class ClientItem(TypedDict):
    input_ids: list[int]
    """The input token IDs."""

    multi_modal_data: NotRequired[dict[str, list]]
    """Per-modality URL/path lists (image/video/audio) forwarded to vLLM via
    ``extra_body.multi_modal_data`` so the server can attach media features
    without re-rendering the chat template."""

    messages: NotRequired[list[ChatCompletionMessageParam]]
    """Optional fallback. Only consumed when ``use_chat_completions=True`` is
    passed to :func:`generate_hidden_states`. The default token-id path
    ignores this field."""


async def _poll_lock_async(fd, poll_interval):
    while True:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            return
        except BlockingIOError:
            await asyncio.sleep(poll_interval)


async def wait_for_lock_async(lock_path, timeout=10.0, poll_interval=0.1):
    fd = os.open(lock_path, os.O_RDONLY)
    try:
        await asyncio.wait_for(_poll_lock_async(fd, poll_interval), timeout=timeout)
    except BaseException:
        os.close(fd)
        raise
    os.close(fd)
    os.remove(lock_path)


def wait_for_lock(lock_path, timeout=10.0, poll_interval=0.1):
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
    use_chat_completions: bool = False,
) -> str | tuple[str, list[int]]:
    """Generate hidden states asynchronously and return the safetensors path.

    For multimodal samples, set ``use_chat_completions=True`` so vLLM runs
    its vision encoder.  In that mode the return value is a tuple
    ``(path, server_token_ids)`` — the caller must use ``server_token_ids``
    as the authoritative input_ids (they are positionally aligned with the
    hidden states).

    For text-only samples the default Completions path is used (strict
    token-id parity check) and a plain ``str`` path is returned.
    """
    token_ids = client_item["input_ids"]
    messages = client_item.get("messages")

    is_mm_chat = use_chat_completions and messages is not None

    coro: Coroutine[Any, Any, Completion | ChatCompletion]
    if is_mm_chat:
        coro = client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=1,
            extra_body={"add_generation_prompt": False, "return_token_ids": True},
            timeout=timeout,
        )
    else:
        extra_body: dict[str, Any] = {"return_token_ids": True}
        mm = client_item.get("multi_modal_data")
        if mm:
            extra_body["multi_modal_data"] = mm
        coro = client.completions.create(
            model=model,
            prompt=token_ids,
            max_tokens=1,
            extra_body=extra_body,
            timeout=timeout,
        )

    res: Completion | ChatCompletion
    if timeout is not None:
        res = await asyncio.wait_for(coro, timeout=timeout)
    else:
        res = await coro

    return extract_output(res, token_ids, trust_server_token_ids=is_mm_chat)


@with_retries
def generate_hidden_states(
    client: openai.Client,
    model: str,
    client_item: ClientItem,
    *,
    timeout: float | None = DEFAULT_REQUEST_TIMEOUT,
    use_chat_completions: bool = False,
) -> str | tuple[str, list[int]]:
    """Generate hidden states via vLLM (synchronous version).

    For multimodal samples, set ``use_chat_completions=True`` so vLLM runs
    its vision encoder.  Returns ``(path, server_token_ids)`` in that mode.
    For text-only samples returns a plain ``str`` path.
    """
    token_ids = client_item["input_ids"]
    messages = client_item.get("messages")

    is_mm_chat = use_chat_completions and messages is not None

    res: Completion | ChatCompletion
    if is_mm_chat:
        res = client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=1,
            extra_body={"add_generation_prompt": False, "return_token_ids": True},
            timeout=timeout,
        )
    else:
        extra_body: dict[str, Any] = {"return_token_ids": True}
        mm = client_item.get("multi_modal_data")
        if mm:
            extra_body["multi_modal_data"] = mm
        res = client.completions.create(
            model=model,
            prompt=token_ids,
            max_tokens=1,
            extra_body=extra_body,
            timeout=timeout,
        )

    return extract_output(res, token_ids, trust_server_token_ids=is_mm_chat)
