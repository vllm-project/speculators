import asyncio
import logging
import time

import openai

logger = logging.getLogger(__name__)

DEFAULT_REQUEST_TIMEOUT = 15  # seconds
DEFAULT_MAX_RETRIES = 3
RETRY_BACKOFF_BASE = 2  # seconds


class InvalidResponseError(Exception):
    pass


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


async def generate_hidden_states_async(
    client: openai.AsyncClient,
    model: str,
    token_ids: list[int],
    timeout: float | None = DEFAULT_REQUEST_TIMEOUT,
    max_retries: int = DEFAULT_MAX_RETRIES,
) -> str:
    """
    Runs decode w/ max_tokens 1 to generate hidden states and returns path to
    hidden states file.

    Args:
        client: The async OpenAI client.
        model: The model ID.
        token_ids: The input token IDs.
        timeout: Timeout in seconds for each request attempt. None for no timeout.
        max_retries: Maximum number of retry attempts on failure.
    """
    total_attempts = max_retries + 1
    last_error: Exception | None = None
    for attempt in range(1, total_attempts + 1):
        try:
            coro = client.completions.create(
                model=model,
                prompt=token_ids,
                max_tokens=1,
                extra_body={"return_token_ids": True},
                timeout=timeout,
            )
            if timeout is not None:
                completion = await asyncio.wait_for(coro, timeout=timeout)
            else:
                completion = await coro

            return extract_output(completion, token_ids)
        except InvalidResponseError:
            raise
        except Exception as e:
            last_error = e
            if attempt < total_attempts:
                backoff = RETRY_BACKOFF_BASE**attempt
                logger.warning(
                    "Request failed (attempt %d/%d): %s. Retrying in %ds...",
                    attempt,
                    total_attempts,
                    e,
                    backoff,
                )
                await asyncio.sleep(backoff)
            else:
                logger.error(
                    "Request failed after %d attempts: %s",
                    total_attempts,
                    e,
                )

    raise last_error  # type: ignore[misc]


def generate_hidden_states(
    client: openai.Client,
    model: str,
    token_ids: list[int],
    timeout: float | None = DEFAULT_REQUEST_TIMEOUT,
    max_retries: int = DEFAULT_MAX_RETRIES,
) -> str:
    total_attempts = max_retries + 1
    last_error: Exception | None = None
    for attempt in range(1, total_attempts + 1):
        try:
            completion = client.completions.create(
                model=model,
                prompt=token_ids,
                max_tokens=1,
                extra_body={"return_token_ids": True},
                timeout=timeout,
            )
            return extract_output(completion, token_ids)
        except InvalidResponseError:
            raise
        except Exception as e:
            last_error = e
            if attempt < total_attempts:
                backoff = RETRY_BACKOFF_BASE**attempt
                logger.warning(
                    "Request failed (attempt %d/%d): %s. Retrying in %ds...",
                    attempt,
                    total_attempts,
                    e,
                    backoff,
                )
                time.sleep(backoff)
            else:
                logger.error(
                    "Request failed after %d attempts: %s",
                    total_attempts,
                    e,
                )

    raise last_error  # type: ignore[misc]
