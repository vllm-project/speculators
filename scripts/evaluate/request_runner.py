"""Collect response metrics with bounded concurrency and durable request results."""

from __future__ import annotations

import asyncio
import json
from typing import TextIO

import httpx


async def send_request(
    client: httpx.AsyncClient, target: str, request: dict, sequence: int
) -> dict:
    """Read SSE through its final usage chunk; never count chunks as tokens."""
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
        async with client.stream(
            "POST", target + "/v1/chat/completions", json=request["body"]
        ) as response:
            if response.is_error:
                await response.aread()
                raise ValueError(f"HTTP {response.status_code}: {response.text[:8192]}")
            response.raise_for_status()
            await _read_stream(response, result)
        result["output_tokens"] = result["usage"]["completion_tokens"]
        result["prompt_tokens"] = result["usage"]["prompt_tokens"]
    except (httpx.HTTPError, ValueError, KeyError, TypeError) as exc:
        result.update(status="error", error=f"{type(exc).__name__}: {exc}")
    return result


async def _read_stream(response: httpx.Response, result: dict) -> None:
    done = False
    async for line in response.aiter_lines():
        if not line.startswith("data:"):
            continue
        data = line[5:].strip()
        if data == "[DONE]":
            done = True
            break
        chunk = _parse_chunk(data)
        if chunk.get("usage") is not None:
            result["usage"] = chunk["usage"]
        if chunk.get("metrics") is not None:
            result["metrics"] = chunk["metrics"]
        for choice in chunk.get("choices", []):
            if choice.get("finish_reason") is not None:
                result["finish_reason"] = choice["finish_reason"]
    _validate_complete(result, done)


def _parse_chunk(data: str) -> dict:
    chunk = json.loads(data)
    if not isinstance(chunk, dict):
        raise ValueError("stream chunk must be an object")
    if chunk.get("error"):
        raise ValueError(f"server stream error: {chunk['error']}")
    return chunk


def _validate_complete(result: dict, done: bool) -> None:
    if not done or result["usage"] is None or result["finish_reason"] is None:
        raise ValueError("incomplete stream: expected finish reason, usage and [DONE]")
    if not isinstance(result["usage"], dict):
        raise ValueError("usage must be an object")
    for key in ("completion_tokens", "prompt_tokens"):
        value = result["usage"].get(key)
        if type(value) is not int or value < 0:
            raise ValueError(f"invalid token usage: {key}")


async def collect_requests(
    requests: list[dict],
    target: str,
    concurrency: int,
    output: TextIO,
    request_timeout: float,
    count: int,
) -> list[dict]:
    """Send a fixed number of requests, preserving each response's metrics."""
    results = []
    sequence = 0
    limits = httpx.Limits(
        max_connections=concurrency, max_keepalive_connections=concurrency
    )
    async with httpx.AsyncClient(
        timeout=request_timeout, limits=limits, trust_env=False
    ) as client:

        async def worker() -> None:
            nonlocal sequence
            while sequence < count:
                index = sequence
                sequence += 1
                request = requests[index % len(requests)]
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

        await asyncio.gather(*(worker() for _ in range(concurrency)))
    return results
