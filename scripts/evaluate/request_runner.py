"""Generic request runner: send a set of chat requests, record everything.

This layer is deliberately benchmark- and metric-agnostic. A caller supplies an
ordered sequence of :class:`Request` objects -- each just chat messages, a
sampling budget, and an opaque ``metadata`` dict -- and, optionally, a stop
predicate evaluated after each batch. The runner renders every request through
the server's own ``/v1/chat/completions/render`` endpoint (which doubles as an
exact-length + context fit-test), generates whatever fits, and streams one flat
row per request to a Parquet dataset. Each row carries the request identity and
metadata, the exact prompt length, the response text, and the *raw* server
metrics blob -- all untouched.

The runner does not know what a "bin", a "context length", or "speculative
decoding" is. Interpreting the recorded table -- binning, acceptance math,
plots -- is the job of a separate analysis layer (see ``acceptance_report.py``),
so the same recording can be re-analysed any number of ways without re-running
inference, and other benchmarks can reuse this runner unchanged.

A benchmark, in this design, is just "a set of requests to make, in some order,
with an optional early-stop rule" -- produced by a thin adapter (see ``mrcr.py``).

Durability: each batch is written as its own Parquet part file
(``part-00000.parquet`` ...) under the table directory. A completed batch is a
complete, readable file, so a crash only loses the batch in flight; the table is
the whole directory, read back with :func:`load_table`.
"""

from __future__ import annotations

import hashlib
import json
import logging
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal, TypeVar
from urllib.error import HTTPError, URLError
from urllib.request import Request as UrlRequest
from urllib.request import urlopen

import pyarrow as pa
import pyarrow.parquet as pq

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Sequence
    from pathlib import Path

logger = logging.getLogger("evaluate")

T = TypeVar("T")

RowStatus = Literal["ok", "oversized", "error"]

_RENDER_TIMEOUT_S = 180
_GENERATE_TIMEOUT_S = 1200

# HTTP statuses the render endpoint uses to reject a prompt that exceeds the
# model's context window (vLLM: 400; some proxies: 413). Every other status is a
# genuine failure, not an oversized prompt.
_CONTEXT_LIMIT_STATUSES = frozenset({400, 413})

# Flat, stable schema for the recorded table. The two ``*_json`` columns keep the
# runner generic: it never interprets server metrics or caller metadata, it just
# records them verbatim for the analysis layer to parse.
TABLE_SCHEMA = pa.schema(
    [
        ("request_id", pa.string()),
        ("status", pa.string()),  # ok | oversized | error
        ("prompt_tokens", pa.int64()),
        ("granted_max_tokens", pa.int64()),
        ("completion_tokens", pa.int64()),
        ("finish_reason", pa.string()),
        ("response_text", pa.string()),
        ("metrics_json", pa.string()),  # verbatim response["metrics"]
        ("metadata_json", pa.string()),  # verbatim Request.metadata
    ]
)

TABLE_DIRNAME = "raw_table"


@dataclass
class Request:
    """One chat request to send, plus opaque metadata echoed into the table."""

    messages: list[dict]
    max_tokens: int = 512
    sampling: dict = field(default_factory=dict)  # temperature, top_p, seed, ...
    metadata: dict = field(default_factory=dict)  # echoed verbatim into the row
    request_id: str = ""


# ---------------------------------------------------------------------------
# Generic request-set helpers (produce/order the batches a runner consumes)
# ---------------------------------------------------------------------------


def stable_hash(value: str) -> str:
    """A stable, process-independent hash usable as a deterministic sort key."""
    return hashlib.sha256(value.encode()).hexdigest()


def _bin_index(length: float, bin_edges: Sequence[float]) -> int:
    for i, edge in enumerate(bin_edges):
        if length <= edge:
            return i
    return len(bin_edges)


def select_stratified(
    items: Iterable[T],
    *,
    bin_edges: Sequence[float],
    samples_per_bin: int,
    key_fn: Callable[[T], str],
    length_fn: Callable[[T], float],
) -> list[list[T]]:
    """Deterministically pick up to *samples_per_bin* items per length bin.

    Items are placed into fixed absolute bins by ``length_fn`` and, within each
    bin, ranked by ``key_fn`` (a stable content hash); the lowest-ranked
    ``samples_per_bin`` per bin are kept. The result is grouped by bin and
    ordered by ascending bin, ready to hand to :func:`run_requests` as batches
    -- render smallest-first and stop once a whole bin no longer fits. The
    selection depends only on the items and these arguments, never on the
    server, so a larger-context run is a superset of a smaller one.
    """
    bins: dict[int, list[T]] = defaultdict(list)
    for item in items:
        bins[_bin_index(length_fn(item), bin_edges)].append(item)
    return [sorted(bins[b], key=key_fn)[:samples_per_bin] for b in sorted(bins)]


# ---------------------------------------------------------------------------
# HTTP
# ---------------------------------------------------------------------------


def _post(root_url: str, path: str, body: dict, timeout: int) -> dict:
    req = UrlRequest(  # noqa: S310
        f"{root_url}{path}",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urlopen(req, timeout=timeout) as resp:  # noqa: S310
        return json.loads(resp.read())


def _render(root_url: str, model: str, req: Request) -> tuple[int, int] | RowStatus:
    """Return ``(prompt_tokens, granted_max_tokens)`` or a failure status.

    Only a context-limit rejection means the prompt genuinely doesn't fit --
    reported as ``"oversized"``. The render endpoint returns 400/413 for that.
    Any other HTTP status (a missing route, or a transient 5xx) is a *failure*,
    not an oversized prompt, and must be reported as ``"error"`` -- otherwise an
    ordered benchmark's ``_all_oversized`` stop rule can halt the whole run on a
    blip and the table would blame prompt length for a server fault.
    """
    try:
        rendered = _post(
            root_url,
            "/v1/chat/completions/render",
            {"model": model, "messages": req.messages, "max_tokens": req.max_tokens},
            timeout=_RENDER_TIMEOUT_S,
        )
        return len(rendered["token_ids"]), rendered["sampling_params"]["max_tokens"]
    except HTTPError as e:
        if e.code in _CONTEXT_LIMIT_STATUSES:
            return "oversized"
        logger.warning(
            "Render endpoint returned HTTP %s for %s: %s", e.code, req.request_id, e
        )
        return "error"
    except (URLError, OSError, json.JSONDecodeError, KeyError, TypeError) as e:
        logger.warning("Render failed for %s: %s", req.request_id, e)
        return "error"


def _row(req: Request, status: RowStatus, **fields) -> dict:
    return {
        "request_id": req.request_id,
        "status": status,
        "prompt_tokens": None,
        "granted_max_tokens": None,
        "completion_tokens": None,
        "finish_reason": None,
        "response_text": None,
        "metrics_json": None,
        "metadata_json": json.dumps(req.metadata),
        **fields,
    }


def _execute(
    root_url: str, model: str, req: Request, *, fit_test: bool, min_room: int
) -> dict:
    """Render (optional), generate, and return one flat table row."""
    max_tokens = req.max_tokens
    prompt_tokens: int | None = None
    granted: int | None = None

    if fit_test:
        rendered = _render(root_url, model, req)
        if isinstance(rendered, str):  # "oversized" | "error"
            return _row(req, rendered)
        prompt_tokens, granted = rendered
        if granted < min_room:
            # Fits, but too little generation room left to be worth measuring.
            return _row(req, "oversized", prompt_tokens=prompt_tokens)
        max_tokens = granted

    try:
        resp = _post(
            root_url,
            "/v1/chat/completions",
            {
                "model": model,
                "messages": req.messages,
                "max_tokens": max_tokens,
                **req.sampling,
            },
            timeout=_GENERATE_TIMEOUT_S,
        )
    except (HTTPError, URLError, OSError, json.JSONDecodeError) as e:
        logger.warning("Generation failed for %s: %s", req.request_id, e)
        return _row(req, "error", prompt_tokens=prompt_tokens)

    choice = (resp.get("choices") or [{}])[0]
    usage = resp.get("usage") or {}
    return _row(
        req,
        "ok",
        prompt_tokens=prompt_tokens or usage.get("prompt_tokens"),
        granted_max_tokens=granted,
        completion_tokens=usage.get("completion_tokens"),
        finish_reason=choice.get("finish_reason"),
        response_text=(choice.get("message") or {}).get("content"),
        metrics_json=json.dumps(resp.get("metrics") or {}),
    )


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


def run_requests(
    root_url: str,
    model: str,
    batches: Iterable[Sequence[Request]],
    table_dir: Path,
    *,
    max_concurrency: int,
    fit_test: bool = True,
    min_generation_room: int = 0,
    should_stop: Callable[[list[dict]], bool] | None = None,
) -> Path:
    """Send *batches* of requests concurrently and record every result.

    Batches are processed in order; within a batch requests run concurrently.
    After each batch the (list of) rows is written as a Parquet part file and,
    if *should_stop* returns True for that batch's rows, the run halts (used by
    ordered benchmarks to stop once a whole batch is oversized). Returns the
    table directory, readable via :func:`load_table`.
    """
    table_dir.mkdir(parents=True, exist_ok=True)
    # Start from a clean table: a reuse of this directory by a shorter run would
    # otherwise leave stale higher-index part files that load_table() reads back,
    # silently mixing the previous run's rows into this one's reports.
    for stale in table_dir.glob("part-*.parquet"):
        stale.unlink()
    n_ok = n_oversized = n_error = 0

    with ThreadPoolExecutor(max_workers=max_concurrency) as pool:
        for part_idx, batch in enumerate(batches):
            if not batch:
                continue
            futures = [
                pool.submit(
                    _execute,
                    root_url,
                    model,
                    req,
                    fit_test=fit_test,
                    min_room=min_generation_room,
                )
                for req in batch
            ]
            rows = [f.result() for f in as_completed(futures)]

            pq.write_table(
                pa.Table.from_pylist(rows, schema=TABLE_SCHEMA),
                table_dir / f"part-{part_idx:05d}.parquet",
                compression="zstd",
            )
            n_ok += sum(r["status"] == "ok" for r in rows)
            n_oversized += sum(r["status"] == "oversized" for r in rows)
            n_error += sum(r["status"] == "error" for r in rows)
            logger.info(
                "Batch %d: %d ok, %d oversized, %d error (cumulative %d/%d/%d)",
                part_idx,
                sum(r["status"] == "ok" for r in rows),
                sum(r["status"] == "oversized" for r in rows),
                sum(r["status"] == "error" for r in rows),
                n_ok,
                n_oversized,
                n_error,
            )
            if should_stop is not None and should_stop(rows):
                logger.info(
                    "Stop predicate satisfied after batch %d; halting.", part_idx
                )
                break

    logger.info(
        "Run complete: %d ok, %d oversized, %d error. Table: %s",
        n_ok,
        n_oversized,
        n_error,
        table_dir,
    )
    return table_dir


def load_table(table_dir: Path) -> pa.Table:
    """Read a recorded table (a directory of Parquet part files) back in."""
    return pq.read_table(table_dir, schema=TABLE_SCHEMA)
