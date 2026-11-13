"""Pool speculative acceptance by phase from the vLLM token stream."""

from __future__ import annotations

import json
import logging
from collections import Counter
from functools import cache
from http import HTTPStatus
from itertools import accumulate, chain
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

PHASES = ("thinking", "final_answer", "boundary", "unresolved")
logger = logging.getLogger("evaluate")


@cache
def _marker_ids(name: str, revision: str | None) -> tuple[int, int]:
    """Return the verifier's opening and closing thinking token IDs."""
    from transformers import AutoTokenizer  # noqa: PLC0415

    vocab = AutoTokenizer.from_pretrained(name, revision=revision).get_vocab()
    return vocab["<think>"], vocab["</think>"]


def _earliest_starts(widths: list[int], chunks: list[int]) -> list[int]:
    """Find the earliest offsets where the rounds fit within stream chunks."""
    ends = iter(accumulate(chunks))
    start, end = 0, next(ends, 0)
    starts = []
    for width in widths:
        while start + width > end:
            start, end = end, next(ends, 0)
            if not end:
                return []
        starts.append(start)
        start += width
    return starts


def _phase(start: int, width: int, boundary: int) -> str:
    """Classify a token span relative to the closing thinking marker."""
    if start + width <= boundary:
        return "thinking"
    return "final_answer" if start > boundary else "boundary"


def _align_phases(widths: list[int], chunks: list[int], boundary: int) -> list[str]:
    """Assign a phase only when the earliest and latest placements agree."""
    if not widths:
        return []
    # A chunk can merge verification rounds. Gaps can be unlogged one-token
    # rounds, and the final verification round may be trimmed to one token.
    packed = [*widths[:-1], 1]
    first = _earliest_starts(packed, chunks)
    reverse = _earliest_starts(packed[::-1], chunks[::-1])
    if not first or not reverse:
        return ["unresolved"] * len(widths)

    total = sum(chunks)
    phases = []
    for start, from_end, width in zip(first, reversed(reverse), packed, strict=True):
        earliest = _phase(start, width, boundary)
        latest = _phase(total - from_end - width, width, boundary)
        phases.append(earliest if earliest == latest else "unresolved")
    if widths[-1] > 1:
        phases[-1] = "unresolved"
    return phases


def _events(record: dict) -> Iterator[dict]:
    """Yield JSON data events from an AIPerf raw response record."""
    for response in record["responses"]:
        for packet in response.get("packets", ()):
            if packet["name"] == "data" and packet["value"] != "[DONE]":
                yield json.loads(packet["value"])


def _request_phases(
    record: dict,
    widths: list[int],
    completion_tokens: int | None,
    markers: tuple[int, int] | None,
) -> list[str]:
    """Classify one request's verification rounds using its streamed token IDs."""
    unresolved = ["unresolved"] * len(widths)
    if not markers or not widths:
        return unresolved
    opener, closer = markers
    chunks, prompt = [], []
    for body in _events(record):
        prompt = body.get("prompt_token_ids") or prompt
        if choices := body.get("choices"):
            if len(choices) != 1 or choices[0].get("index", 0):
                return unresolved
            if ids := choices[0].get("token_ids"):
                chunks.append(ids)
    tokens = list(chain.from_iterable(chunks))
    if (
        not prompt
        or prompt[-1] != opener
        or len(tokens) != completion_tokens
        or opener in tokens
        or tokens.count(closer) > 1
    ):
        return unresolved
    boundary = tokens.index(closer) if closer in tokens else len(tokens)
    return _align_phases(widths, list(map(len, chunks)), boundary)


def phase_rows(
    report: dict, raw_path: Path, dataset: str, specs_by_request: dict
) -> tuple[list[dict], bool]:
    """Return pooled phase metrics and whether the request exports reconcile."""
    tokenizer = report.get("input_config", {}).get("tokenizer") or {}
    try:
        markers = _marker_ids(tokenizer["name"], tokenizer.get("revision"))
    except (ImportError, KeyError, OSError, ValueError):
        markers = None
        logger.warning("%s: use --tokenizer with <think> and </think> tokens", dataset)

    accepted, drafted, steps = Counter(), Counter(), Counter()
    # AIPerf's normalized export selects the profiling requests and validates
    # their counters. The raw export supplies only the token stream.
    pending = specs_by_request.copy()
    detailed = unresolved = 0
    valid = True
    with raw_path.open() as handle:
        for line in handle:
            try:
                record = json.loads(line)
                request_id = record["metadata"].get("x_request_id")
                if (
                    request_id not in specs_by_request
                    or record.get("status") != HTTPStatus.OK
                ):
                    continue
                spec = pending.pop(request_id)
                a, d = spec.get("per_step_accepted"), spec.get("per_step_drafted")
                if spec["engine"] != "vllm" or a is None or d is None:
                    raise ValueError("missing detailed vLLM metrics")
                phases = _request_phases(
                    record, [n + 1 for n in a], spec.get("completion_tokens"), markers
                )
                rounds = list(zip(phases, a, d, strict=True))
            except (AttributeError, KeyError, TypeError, ValueError):
                valid = False
                continue
            detailed += 1
            unresolved += "unresolved" in phases
            for phase, kept, proposed in rounds:
                accepted[phase] += kept
                drafted[phase] += proposed
                steps[phase] += 1

    completed = report["request_count"]["avg"]
    valid = valid and not pending and detailed == completed
    if not valid:
        logger.warning(
            "%s: detailed speculative metrics missing or inconsistent", dataset
        )
    return [
        {
            "dataset": dataset,
            "phase": phase,
            "accepted_draft_tokens": accepted[phase],
            "proposed_draft_tokens": drafted[phase],
            "verification_steps": steps[phase],
            "acceptance_rate_pct": round(100 * accepted[phase] / drafted[phase], 2)
            if drafted[phase]
            else "",
            "acceptance_length": round(1 + accepted[phase] / steps[phase], 4)
            if steps[phase]
            else "",
            "successful_requests": completed,
            "detailed_requests": detailed,
            "unresolved_requests": unresolved,
        }
        for phase in PHASES
    ], valid
