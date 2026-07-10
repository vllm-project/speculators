import asyncio
import fcntl
import functools
import logging
import os
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypedDict, cast
from urllib.parse import unquote, urlparse

import openai
from openai.types.chat import ChatCompletion, ChatCompletionMessageParam
from openai.types.completion import Completion
from typing_extensions import NotRequired

from speculators.data_generation.media import get_image_ref

if TYPE_CHECKING:
    from collections.abc import Coroutine

logger = logging.getLogger(__name__)

DEFAULT_REQUEST_TIMEOUT = 120  # seconds
DEFAULT_MAX_RETRIES = 3
RETRY_BACKOFF_BASE = 2  # seconds
RETRYABLE_HTTP_STATUS_CODES = {408, 409, 425, 429}


class InvalidResponseError(Exception):
    pass


class RetryableRequestError(RuntimeError):
    """Explicitly mark a non-SDK request failure as safe to retry."""


def _is_retryable_error(error: Exception) -> bool:
    """Return whether a failed request may succeed when retried."""
    if isinstance(error, InvalidResponseError):
        return False
    if isinstance(error, RetryableRequestError):
        return True
    if isinstance(error, openai.APIConnectionError):
        return True
    if isinstance(error, openai.APIStatusError):
        status_code = error.status_code
        return (
            status_code in RETRYABLE_HTTP_STATUS_CODES
            or 500 <= status_code < 600
        )
    if isinstance(error, (TimeoutError, ConnectionError)):
        return True
    # Unknown exceptions are deterministic by default. Callers that know a
    # non-SDK failure is transient must wrap it in RetryableRequestError.
    return False


def _get_field(obj: Any, key: str) -> Any:
    if isinstance(obj, dict):
        return obj.get(key)
    return getattr(obj, key, None)


def _to_token_id_list(token_ids: Any) -> list[int]:
    if hasattr(token_ids, "tolist"):
        token_ids = token_ids.tolist()
    return list(token_ids)


def _image_ref_to_chat_url(image_ref: Any) -> str:
    """Convert a dataset image reference to an OpenAI-compatible image URL."""
    ref = str(image_ref)
    parsed = urlparse(ref)
    if not parsed.scheme and parsed.netloc:
        raise ValueError(
            f"Unsupported schemeless image URI authority: {parsed.netloc}"
        )
    if parsed.scheme in {"http", "https", "data"}:
        return ref

    if parsed.scheme == "file":
        if parsed.netloc not in {"", "localhost"}:
            raise ValueError(f"Unsupported file URI host: {parsed.netloc}")
        if parsed.query or parsed.fragment:
            raise ValueError("Local file URIs must not contain query/fragment")
        decoded_path = unquote(parsed.path)
        if not decoded_path or "\x00" in decoded_path:
            raise ValueError("Local file URI contains an invalid path")
        path = Path(decoded_path)
        if not path.is_absolute():
            raise ValueError("Local file URI path must be absolute")
        return path.resolve().as_uri()

    if parsed.scheme:
        raise ValueError(f"Unsupported image URL scheme: {parsed.scheme}")

    path = Path(ref).expanduser()
    return path.resolve().as_uri()


def _prepare_chat_message_content(content: Any) -> Any:
    if not isinstance(content, list):
        return content

    prepared: list[Any] = []
    for part in content:
        if isinstance(part, str):
            prepared.append({"type": "text", "text": part})
            continue

        if not isinstance(part, dict):
            prepared.append(part)
            continue

        image_ref = get_image_ref(part)
        if image_ref is not None:
            prepared.append(
                {
                    "type": "image_url",
                    "image_url": {"url": _image_ref_to_chat_url(image_ref)},
                }
            )
            continue

        text = part.get("text")
        if text is not None:
            prepared.append({"type": "text", "text": str(text)})
            continue

        prepared.append(part)

    return prepared


def _prepare_chat_messages(
    messages: list[ChatCompletionMessageParam],
) -> list[dict[str, Any]]:
    """Convert processor-style multimodal messages to vLLM chat messages."""
    prepared = []
    for message in messages:
        prepared_message = dict(cast("dict[str, Any]", message))
        prepared_message["content"] = _prepare_chat_message_content(
            prepared_message.get("content", "")
        )
        prepared.append(prepared_message)
    return prepared


def _handle_retry_error(
    error: Exception, attempt: int, total_attempts: int
) -> float | None:
    """Classify and handle a failed request.

    Returns backoff seconds if the caller should retry, or ``None`` on the
    final attempt. Raises deterministic failures immediately.
    """
    if not _is_retryable_error(error):
        logger.error("Request failed with a non-retryable error: %s", error)
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
    logger.error("Request failed after %d attempts: %s", total_attempts, error)
    return None


def _validate_max_retries(max_retries: Any) -> int:
    if (
        isinstance(max_retries, bool)
        or not isinstance(max_retries, int)
        or max_retries < 0
    ):
        raise ValueError("max_retries must be a non-negative integer")
    return max_retries


def with_retries(fn):
    """Decorator that adds retry logic with exponential backoff.

    The decorated function gains a ``max_retries`` keyword argument
    (default ``DEFAULT_MAX_RETRIES``). Only explicitly recognized transient
    transport/status failures and ``RetryableRequestError`` are retried; all
    other exceptions fail fast. Works for both sync and async functions.
    """
    if asyncio.iscoroutinefunction(fn):

        @functools.wraps(fn)
        async def async_wrapper(*args, max_retries=DEFAULT_MAX_RETRIES, **kwargs):
            max_retries = _validate_max_retries(max_retries)
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
        max_retries = _validate_max_retries(max_retries)
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
    allow_prefix_truncation: bool = False,
) -> str:
    token_ids = _to_token_id_list(token_ids)
    prompt_token_ids = _get_field(response, "prompt_token_ids")
    if prompt_token_ids is None:
        choices = _get_field(response, "choices")
        if choices:
            prompt_token_ids = _get_field(choices[0], "prompt_token_ids")

    if prompt_token_ids is None:
        raise InvalidResponseError("Response missing prompt_token_ids")

    kv_transfer_params = _get_field(response, "kv_transfer_params")
    if kv_transfer_params is None:
        raise InvalidResponseError("Response missing kv_transfer_params")

    hidden_states_path = _get_field(kv_transfer_params, "hidden_states_path")
    if hidden_states_path is None:
        raise InvalidResponseError("Response missing hidden_states_path")

    prompt_token_ids = _to_token_id_list(prompt_token_ids)
    if prompt_token_ids == token_ids:
        return hidden_states_path

    if allow_prefix_truncation and prompt_token_ids[: len(token_ids)] == token_ids:
        logger.debug(
            "vLLM returned %d prompt tokens for a %d-token preprocessed prompt; "
            "hidden states will be aligned after the output lock is released.",
            len(prompt_token_ids),
            len(token_ids),
        )
        return hidden_states_path

    raise InvalidResponseError(
        f"Prompt token IDs mismatch: expected {token_ids}, got {prompt_token_ids}"
    )


class ClientItem(TypedDict):
    input_ids: list[int]
    """The input token IDs."""

    messages: NotRequired[list[ChatCompletionMessageParam]]
    """If provided, pass `messages` to Chat Completions API
    instead of passing `token_ids` to Completions API."""

    tools: NotRequired[list[dict[str, Any]]]
    """Tool schemas used to render a multimodal Chat Completions prompt."""


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
) -> str:
    """
    Runs decode w/ max_tokens 1 to generate hidden states and returns path to
    hidden states file.

    Args:
        client: The async OpenAI client.
        model: The model ID.
        client_item: Inputs to send via the client.
        timeout: Timeout in seconds for each request attempt. None for no timeout.
    """
    token_ids = client_item["input_ids"]
    messages = client_item.get("messages")
    tools = client_item.get("tools")

    coro: Coroutine[Any, Any, Completion | ChatCompletion]
    if messages is None:
        coro = client.completions.create(
            model=model,
            prompt=token_ids,
            max_tokens=1,
            extra_body={"return_token_ids": True},
            timeout=timeout,
        )
    else:
        chat_messages = _prepare_chat_messages(messages)
        chat_kwargs: dict[str, Any] = {
            "model": model,
            "messages": chat_messages,
            "max_tokens": 1,
            "extra_body": {
                "add_generation_prompt": False,
                "continue_final_message": True,
                "return_token_ids": True,
            },
            "timeout": timeout,
        }
        if tools:
            chat_kwargs["tools"] = tools
        coro = client.chat.completions.create(**cast("Any", chat_kwargs))

    res: Completion | ChatCompletion
    if timeout is not None:
        res = await asyncio.wait_for(coro, timeout=timeout)
    else:
        res = await coro

    return extract_output(
        res,
        token_ids,
        allow_prefix_truncation=messages is not None,
    )


@with_retries
def generate_hidden_states(
    client: openai.Client,
    model: str,
    client_item: ClientItem,
    *,
    timeout: float | None = DEFAULT_REQUEST_TIMEOUT,
) -> str:
    """
    Runs decode w/ max_tokens 1 to generate hidden states and returns path to
    hidden states file.
    """
    token_ids = client_item["input_ids"]
    messages = client_item.get("messages")
    tools = client_item.get("tools")

    res: Completion | ChatCompletion
    if messages is None:
        res = client.completions.create(
            model=model,
            prompt=token_ids,
            max_tokens=1,
            extra_body={"return_token_ids": True},
            timeout=timeout,
        )
    else:
        chat_messages = _prepare_chat_messages(messages)
        chat_kwargs: dict[str, Any] = {
            "model": model,
            "messages": chat_messages,
            "max_tokens": 1,
            "extra_body": {
                "add_generation_prompt": False,
                "continue_final_message": True,
                "return_token_ids": True,
            },
            "timeout": timeout,
        }
        if tools:
            chat_kwargs["tools"] = tools
        res = client.chat.completions.create(**cast("Any", chat_kwargs))

    return extract_output(
        res,
        token_ids,
        allow_prefix_truncation=messages is not None,
    )
