import asyncio

import httpx2
import openai
import pytest

from speculators.data_generation import vllm_client
from speculators.train.recovery import GenerationRecoveryGuard, SampleUnavailable


@pytest.mark.parametrize(
    ("status", "expected"), [(400, 1), (404, 1), (429, 6), (503, 6)]
)
@pytest.mark.parametrize("asynchronous", [False, True])
def test_outer_guard(status, expected, asynchronous, monkeypatch):
    monkeypatch.setattr(vllm_client, "RETRY_BACKOFF_BASE", 0)
    calls = []
    response = httpx2.Response(
        status, request=httpx2.Request("POST", "https://unit.invalid")
    )

    def fail():
        calls.append(1)
        raise openai.APIStatusError("synthetic", response=response, body=None)

    guard = GenerationRecoveryGuard(retries=2, max_consecutive_failures=2)
    if asynchronous:

        @vllm_client.with_retries
        async def op():
            fail()

        result = asyncio.run(
            guard.run_async(lambda: op(max_retries=1), description="unit")
        )
    else:
        op = vllm_client.with_retries(fail)
        result = guard.run(lambda: op(max_retries=1), description="unit")
    assert isinstance(result, SampleUnavailable)
    assert result.consecutive_failures == 1
    assert not result.fatal
    assert len(calls) == expected
