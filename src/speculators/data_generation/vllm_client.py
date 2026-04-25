import asyncio
import functools
import logging
import time
from typing import Any, cast

import openai

logger = logging.getLogger(__name__)

DEFAULT_REQUEST_TIMEOUT = 15  # seconds
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


def extract_output(completion, token_ids) -> str:
    prompt_token_ids = getattr(completion.choices[0], "prompt_token_ids", None)

    if prompt_token_ids is None:
        raise InvalidResponseError("Response missing prompt_token_ids")

    if prompt_token_ids != token_ids:
        raise InvalidResponseError(
            f"Prompt token IDs mismatch: expected {token_ids}, got {prompt_token_ids}"
        )

    if not hasattr(completion, "kv_transfer_params"):
        raise InvalidResponseError("Response missing kv_transfer_params")

    return completion.kv_transfer_params.get("hidden_states_path")


def _build_request_payload(
    token_ids: list[int],
    prompt: str | None = None,
    multi_modal_data: dict[str, Any] | None = None,
) -> tuple[list[int] | dict[str, Any], dict[str, Any]]:
    """Build a vLLM completion request payload.

    Text-only requests can pass token IDs directly. Requests that need the
    original prompt text or multimodal inputs use vLLM's structured prompt form.
    """
    extra_body: dict[str, Any] = {"return_token_ids": True}

    if prompt is None and multi_modal_data is None:
        return token_ids, extra_body

    request_prompt: dict[str, Any] = {"prompt_token_ids": token_ids}
    if prompt is not None:
        request_prompt["prompt"] = prompt
    if multi_modal_data is not None:
        request_prompt["multi_modal_data"] = multi_modal_data

    return request_prompt, extra_body


@with_retries
async def generate_hidden_states_async(
    client: openai.AsyncClient,
    model: str,
    token_ids: list[int],
    prompt: str | None = None,
    multi_modal_data: dict[str, Any] | None = None,
    timeout: float | None = DEFAULT_REQUEST_TIMEOUT,
) -> str:
    """
    Runs decode w/ max_tokens 1 to generate hidden states and returns path to
    hidden states file.

    Args:
        client: The async OpenAI client.
        model: The model ID.
        token_ids: The input token IDs.
        prompt: Optional prompt text corresponding to ``token_ids``.
        multi_modal_data: Optional multimodal payload expected by vLLM.
        timeout: Timeout in seconds for each request attempt. None for no timeout.
    """
    request_prompt, extra_body = _build_request_payload(
        token_ids, prompt, multi_modal_data
    )
    coro = client.completions.create(
        model=model,
        prompt=cast("Any", request_prompt),
        max_tokens=1,
        extra_body=extra_body,
        timeout=timeout,
    )
    if timeout is not None:
        completion = await asyncio.wait_for(coro, timeout=timeout)
    else:
        completion = await coro

    return extract_output(completion, token_ids)


@with_retries
def generate_hidden_states(
    client: openai.Client,
    model: str,
    token_ids: list[int],
    prompt: str | None = None,
    multi_modal_data: dict[str, Any] | None = None,
    timeout: float | None = DEFAULT_REQUEST_TIMEOUT,
) -> str:
    """
    Runs decode w/ max_tokens 1 to generate hidden states and returns path to
    hidden states file.
    """
    request_prompt, extra_body = _build_request_payload(
        token_ids, prompt, multi_modal_data
    )
    completion = client.completions.create(
        model=model,
        prompt=cast("Any", request_prompt),
        max_tokens=1,
        extra_body=extra_body,
        timeout=timeout,
    )
    return extract_output(completion, token_ids)
