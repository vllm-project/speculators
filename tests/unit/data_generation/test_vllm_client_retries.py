"""Retry classification for ``with_retries`` on OpenAI client status errors."""

import asyncio
from types import SimpleNamespace
from typing import Any

import openai
import pytest

from speculators.data_generation import vllm_client
from speculators.data_generation.vllm_client import with_retries


@pytest.fixture(autouse=True)
def _no_retry_backoff(monkeypatch):
    # with_retries sleeps RETRY_BACKOFF_BASE ** attempt between retries; zero it
    # so the retry tests don't actually sleep.
    monkeypatch.setattr(vllm_client, "RETRY_BACKOFF_BASE", 0)


# openai>=3 is built on httpx2 and openai 2.x on httpx; the error constructors
# only read these attributes, so a stand-in keeps the tests version-agnostic.
_REQUEST: Any = SimpleNamespace(method="POST", url="http://x/v1/completions")


def _status_error(status: int) -> openai.APIStatusError:
    response: Any = SimpleNamespace(status_code=status, request=_REQUEST, headers={})
    cls = {
        400: openai.BadRequestError,
        404: openai.NotFoundError,
        429: openai.RateLimitError,
    }.get(status, openai.InternalServerError)
    return cls(f"HTTP {status}", response=response, body=None)


def _counting(status: int):
    calls = {"n": 0}

    @with_retries
    def request():
        calls["n"] += 1
        raise _status_error(status)

    return request, calls


@pytest.mark.parametrize("status", [400, 404])
def test_permanent_status_is_not_retried(status):
    # e.g. vLLM's 400 "max_tokens must be at least 1, got 0" for a prompt that
    # fills max_model_len: retrying cannot change the outcome.
    request, calls = _counting(status)

    with pytest.raises(openai.APIStatusError):
        request(max_retries=3)

    assert calls["n"] == 1


@pytest.mark.parametrize("status", [429, 500, 503])
def test_transient_status_is_retried(status):
    request, calls = _counting(status)

    with pytest.raises(openai.APIStatusError):
        request(max_retries=3)

    assert calls["n"] == 4


def test_async_permanent_status_is_not_retried():
    calls = {"n": 0}

    @with_retries
    async def request():
        calls["n"] += 1
        raise _status_error(400)

    with pytest.raises(openai.BadRequestError):
        asyncio.run(request(max_retries=3))

    assert calls["n"] == 1


def test_connection_error_is_still_retried():
    calls = {"n": 0}

    @with_retries
    def request():
        calls["n"] += 1
        raise openai.APIConnectionError(request=_REQUEST)

    with pytest.raises(openai.APIConnectionError):
        request(max_retries=2)

    assert calls["n"] == 3
