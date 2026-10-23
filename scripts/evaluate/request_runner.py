"""Collect response metrics with bounded concurrency and durable request results."""

from __future__ import annotations

import asyncio
import json
from typing import TextIO

import httpx

MAX_CONSECUTIVE_FAILURES = 3


async def send_request(
    client: httpx.AsyncClient, target: str, request: dict, sequence: int
) -> dict:
    """Collect usage and speculative metrics from a non-streaming response."""
    result = {
        "sample_id": request["id"],
        "dataset": request["dataset"],
        "sequence": sequence,
        "status": "ok",
        "metrics": None,
        "usage": None,
        "finish_reason": None,
        "output_tokens": 0,
        "prompt_tokens": 0,
    }
    try:
        body = {**request["body"], "stream": False}
        body.pop("stream_options", None)
        response = await client.post(target + "/v1/chat/completions", json=body)
        if response.is_error:
            raise ValueError(f"HTTP {response.status_code}: {response.text[:8192]}")
        response.raise_for_status()
        data = response.json()
        result["usage"] = data["usage"]
        result["metrics"] = data.get("metrics")
        result["finish_reason"] = data["choices"][0]["finish_reason"]
        if result["finish_reason"] is None:
            raise ValueError("incomplete response: expected finish reason")
        if not isinstance(result["usage"], dict):
            raise ValueError("usage must be an object")
        for key in ("completion_tokens", "prompt_tokens"):
            value = result["usage"].get(key)
            if type(value) is not int or value < 0:
                raise ValueError(f"invalid token usage: {key}")
        result["output_tokens"] = result["usage"]["completion_tokens"]
        result["prompt_tokens"] = result["usage"]["prompt_tokens"]
    except (httpx.HTTPError, ValueError, KeyError, TypeError, IndexError) as exc:
        result.update(status="error", error=f"{type(exc).__name__}: {exc}")
    return result


async def collect_requests(
    requests: list[dict],
    target: str,
    concurrency: int,
    output: TextIO,
    request_timeout: float,
    consecutive_failures: int = 0,
) -> tuple[list[dict], int]:
    """Send samples once; stop scheduling after three errors in completion order."""
    results = []
    sequence = 0
    stopped = False
    limits = httpx.Limits(
        max_connections=concurrency, max_keepalive_connections=concurrency
    )
    async with httpx.AsyncClient(
        timeout=request_timeout, limits=limits, trust_env=False
    ) as client:

        async def worker() -> None:
            nonlocal sequence, consecutive_failures, stopped
            while sequence < len(requests) and not stopped:
                index = sequence
                sequence += 1
                request = requests[index]
                try:
                    async with asyncio.timeout(request_timeout):
                        result = await send_request(client, target, request, index)
                except TimeoutError:
                    result = {
                        "sample_id": request["id"],
                        "dataset": request["dataset"],
                        "sequence": index,
                        "status": "error",
                        "error": "request exceeded total timeout",
                        "output_tokens": 0,
                    }
                result["category"] = request.get("category")
                output.write(json.dumps(result, allow_nan=False) + "\n")
                output.flush()
                results.append(result)
                consecutive_failures = (
                    0 if result["status"] == "ok" else consecutive_failures + 1
                )
                stopped |= consecutive_failures >= MAX_CONSECUTIVE_FAILURES

        await asyncio.gather(*(worker() for _ in range(concurrency)))
    return results, MAX_CONSECUTIVE_FAILURES if stopped else consecutive_failures
