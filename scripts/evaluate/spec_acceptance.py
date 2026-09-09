"""Generic per-request speculative-decoding acceptance harness.

Dataset-agnostic machinery for measuring how spec-decode acceptance holds up as
context length grows. A caller supplies an iterable of "items" plus small
adapter callables (``item -> chat messages``, ``item -> stable key``,
``item -> length proxy``); everything else -- deterministic stratified
selection, rendering, generation, raw-result persistence, binning and
reporting -- lives here so future benchmarks can reuse it.

For each selected item the harness first renders it through the target server's
own ``/v1/chat/completions/render`` endpoint. That applies the chat template and
tokenizes exactly as the server would, giving an exact prompt length and
naturally rejecting anything that doesn't fit the model's context window -- no
separate approximate tokenizer or pre-run bucketing needed. Everything that
fits is then generated with ``--per-request-spec-decode-metrics detailed``
required, so each response carries its own acceptance stats plus per-verify-step
arrays. Every raw per-request result is streamed to ``raw_requests.jsonl`` as it
completes -- that's the source of truth; the aggregated CSVs are just one
slicing of it, rebuildable after the fact with different bin edges via
``load_results`` + ``write_report`` (see ``rebin_mrcr.py``) without re-running
inference.

Results are sliced into context-length buckets two ways:

* by the prompt length at the start of the request (``bin_by_start_length``)
* by the running token position of each verify step (``bin_by_position``) --
  prompt length plus however much has been generated so far.

Deterministic superset selection
---------------------------------
``select_stratified`` chooses which items to run as a pure function of the
items, the caller's flags, and (through the render fit-test) the server's
``max_model_len`` -- never of run order or wall-clock. Items are placed into
fixed absolute length bins by a caller-supplied length proxy and, within each
bin, ranked by a stable content hash; the lowest-ranked ``samples_per_bin`` per
bin form the candidate set. That set is independent of ``max_model_len``, so the
only context-dependent step is the render fit-test, which is monotonic: raising
``max_model_len`` can only let *more* candidates through. The kept set at a
smaller context is therefore always a subset of the kept set at a larger one.
"""

from __future__ import annotations

import hashlib
import json
import logging
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, TypeVar
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from perf_utils import CsvWriter

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

logger = logging.getLogger("evaluate")

T = TypeVar("T")

# Default context-length bucket edges (tokens) for the by-request-start-length
# report. These are OpenAI MRCR's native power-of-2 bins; they only slice
# results for reporting, they don't gate which requests get sent.
DEFAULT_CONTEXT_BIN_EDGES = (
    0,
    4096,
    8192,
    16384,
    32768,
    65536,
    131072,
    262144,
    524288,
    1048576,
)

# Default bin width (tokens) for the by-token-position report -- much finer than
# the context edges since acceptance can shift noticeably within one request as
# generation proceeds.
DEFAULT_POSITION_BIN_SIZE = 256

# Default requested generation length; the render endpoint clips this per-sample
# to whatever actually fits under the server's max_model_len.
DEFAULT_MAX_NEW_TOKENS = 512
# Skip samples left with less than this much generation room -- too few verify
# steps to say anything about acceptance.
MIN_GENERATION_ROOM = 32

_RENDER_TIMEOUT_S = 180
_GENERATE_TIMEOUT_S = 1200

RAW_LOG_FILENAME = "raw_requests.jsonl"

_Failure = Literal["oversized", "error"]


@dataclass
class AcceptanceResult:
    prompt_tokens: int
    metrics: dict


# ---------------------------------------------------------------------------
# Raw-result persistence
# ---------------------------------------------------------------------------


def append_result(f, result: AcceptanceResult) -> None:
    """Append *result* to an open ``raw_requests.jsonl`` file, flushing immediately.

    Flushing per line means a killed or crashed run still leaves every completed
    request's data on disk.
    """
    f.write(
        json.dumps({"prompt_tokens": result.prompt_tokens, "metrics": result.metrics})
    )
    f.write("\n")
    f.flush()


def load_results(path: Path) -> list[AcceptanceResult]:
    """Reload results previously written with :func:`append_result`."""
    results = []
    with path.open() as f:
        for raw_line in f:
            stripped = raw_line.strip()
            if not stripped:
                continue
            obj = json.loads(stripped)
            results.append(AcceptanceResult(obj["prompt_tokens"], obj["metrics"]))
    return results


# ---------------------------------------------------------------------------
# HTTP: render + generate
# ---------------------------------------------------------------------------


def _post(root_url: str, path: str, body: dict, timeout: int) -> dict:
    data = json.dumps(body).encode()
    req = Request(  # noqa: S310
        f"{root_url}{path}",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urlopen(req, timeout=timeout) as resp:  # noqa: S310
        return json.loads(resp.read())


def render(
    root_url: str,
    model: str,
    messages: list[dict],
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
) -> tuple[int, int] | _Failure:
    """Render *messages*, returning ``(prompt_tokens, granted_max_tokens)``.

    Doubles as the context-length fit-test: a render failure (or too little
    generation room left after clipping) means the prompt doesn't fit the model,
    so the caller should drop the sample rather than send it for real.
    """
    try:
        rendered = _post(
            root_url,
            "/v1/chat/completions/render",
            {"model": model, "messages": messages, "max_tokens": max_new_tokens},
            timeout=_RENDER_TIMEOUT_S,
        )
        prompt_tokens = len(rendered["token_ids"])
        granted = rendered["sampling_params"]["max_tokens"]
    except HTTPError:
        return "oversized"
    except (URLError, OSError, json.JSONDecodeError, KeyError, TypeError) as e:
        logger.warning("Render failed, skipping sample: %s", e)
        return "error"

    if granted < MIN_GENERATION_ROOM:
        return "oversized"
    return prompt_tokens, granted


def render_and_generate(
    root_url: str,
    model: str,
    messages: list[dict],
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
) -> AcceptanceResult | _Failure:
    """Render *messages* through the target server, then (if it fits) generate."""
    rendered = render(root_url, model, messages, max_new_tokens)
    if isinstance(rendered, str):
        return rendered
    prompt_tokens, granted = rendered

    try:
        response = _post(
            root_url,
            "/v1/chat/completions",
            {"model": model, "messages": messages, "max_tokens": granted},
            timeout=_GENERATE_TIMEOUT_S,
        )
        spec = (response.get("metrics") or {}).get("speculative_decoding")
    except (HTTPError, URLError, OSError, json.JSONDecodeError) as e:
        logger.warning("Generation request failed, skipping sample: %s", e)
        return "error"

    if not spec:
        return "error"
    return AcceptanceResult(prompt_tokens, spec)


def _has_detailed_metrics(result: AcceptanceResult) -> bool:
    return bool(result.metrics.get("per_step_accepted"))


def _fail_missing_detailed_metrics() -> None:
    logger.error(
        "Server did not return detailed per-request spec-decode metrics for a "
        "test request. Relaunch it with --per-request-spec-decode-metrics "
        "detailed and a --speculative-config."
    )
    sys.exit(1)


# ---------------------------------------------------------------------------
# Deterministic stratified selection
# ---------------------------------------------------------------------------


def stable_hash(value: str) -> str:
    """A stable, process-independent hash usable as a deterministic sort key."""
    return hashlib.sha256(value.encode()).hexdigest()


def _bin_index(length: float, bin_edges: tuple[int, ...]) -> int:
    """Return the index of the first edge >= *length* (last index for overflow)."""
    for i, edge in enumerate(bin_edges):
        if length <= edge:
            return i
    return len(bin_edges)


def select_stratified(
    items: list[T],
    *,
    bin_edges: tuple[int, ...],
    samples_per_bin: int,
    key_fn: Callable[[T], str],
    length_fn: Callable[[T], float],
) -> list[list[T]]:
    """Deterministically pick up to *samples_per_bin* items per length bin.

    Items are placed into fixed absolute bins by ``length_fn`` and, within each
    bin, ranked by ``key_fn`` (a stable content hash); the lowest-ranked
    ``samples_per_bin`` per bin are kept. The result is grouped by bin and
    ordered by ascending bin, so callers can render smallest-first and stop once
    a whole bin no longer fits. The selection depends only on the items and the
    arguments here -- never on ``max_model_len`` -- which is what makes a
    larger-context run a superset of a smaller one.
    """
    bins: dict[int, list[T]] = defaultdict(list)
    for item in items:
        bins[_bin_index(length_fn(item), bin_edges)].append(item)
    grouped = []
    for b in sorted(bins):
        ranked = sorted(bins[b], key=key_fn)
        grouped.append(ranked[:samples_per_bin])
    return grouped


# ---------------------------------------------------------------------------
# Binning / reporting
# ---------------------------------------------------------------------------


def _bucket_label(n_tokens: int, edges: tuple[int, ...]) -> str:
    for lo, hi in zip(edges, edges[1:], strict=False):
        if n_tokens <= hi:
            return f"{lo}-{hi}"
    return f"{edges[-1]}+"


def _position_bucket_label(n_tokens: int, bin_size: int) -> str:
    lo = (n_tokens // bin_size) * bin_size
    return f"{lo}-{lo + bin_size}"


def _bucket_sort_key(item: tuple[str, dict]) -> int:
    return int(item[0].split("-")[0].rstrip("+"))


def _finish_bucket(bucket: dict) -> dict:
    bucket["draft_acceptance_rate"] = (
        bucket["num_accepted_draft_tokens"] / bucket["num_draft_tokens"]
        if bucket["num_draft_tokens"]
        else 0.0
    )
    bucket["mean_acceptance_length"] = (
        1 + bucket["num_accepted_draft_tokens"] / bucket["num_spec_steps"]
        if bucket["num_spec_steps"]
        else 0.0
    )
    return bucket


def bin_by_start_length(
    results: list[AcceptanceResult],
    edges: tuple[int, ...] = DEFAULT_CONTEXT_BIN_EDGES,
) -> list[dict]:
    """Aggregate whole-request acceptance stats by prompt length at request start."""
    buckets: dict[str, dict] = {}
    for r in results:
        label = _bucket_label(r.prompt_tokens, edges)
        bucket = buckets.setdefault(
            label,
            {
                "bucket": label,
                "num_requests": 0,
                "num_spec_steps": 0,
                "num_draft_tokens": 0,
                "num_accepted_draft_tokens": 0,
            },
        )
        bucket["num_requests"] += 1
        bucket["num_spec_steps"] += r.metrics["num_spec_steps"]
        bucket["num_draft_tokens"] += r.metrics["num_draft_tokens"]
        bucket["num_accepted_draft_tokens"] += r.metrics["num_accepted_draft_tokens"]
    return [_finish_bucket(b) for _, b in sorted(buckets.items(), key=_bucket_sort_key)]


def bin_by_position(
    results: list[AcceptanceResult],
    bin_size: int = DEFAULT_POSITION_BIN_SIZE,
) -> list[dict]:
    """Aggregate per-verify-step acceptance stats by the running token position.

    Each step's position is the prompt length plus everything committed by
    earlier steps in that same request (accepted draft tokens plus the
    always-accepted bonus token).
    """
    buckets: dict[str, dict] = {}
    for r in results:
        accepted = r.metrics.get("per_step_accepted")
        drafted = r.metrics.get("per_step_drafted")
        if not accepted or not drafted:
            continue
        pos = r.prompt_tokens
        for a, d in zip(accepted, drafted, strict=True):
            label = _position_bucket_label(pos, bin_size)
            bucket = buckets.setdefault(
                label,
                {
                    "bucket": label,
                    "num_spec_steps": 0,
                    "num_draft_tokens": 0,
                    "num_accepted_draft_tokens": 0,
                },
            )
            bucket["num_spec_steps"] += 1
            bucket["num_draft_tokens"] += d
            bucket["num_accepted_draft_tokens"] += a
            pos += a + 1
    return [_finish_bucket(b) for _, b in sorted(buckets.items(), key=_bucket_sort_key)]


def _print_bucket_table(title: str, rows: list[dict]) -> None:
    print(f"\n=== {title} ===\n")
    has_requests = "num_requests" in rows[0]
    header = f"  {'Bucket':<18}"
    if has_requests:
        header += f" {'Requests':>9}"
    header += f" {'Steps':>8} {'Accept Rate':>12} {'Mean Len':>9}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for row in rows:
        line = f"  {row['bucket']:<18}"
        if has_requests:
            line += f" {row['num_requests']:>9}"
        line += (
            f" {row['num_spec_steps']:>8} {row['draft_acceptance_rate']:>12.4f}"
            f" {row['mean_acceptance_length']:>9.4f}"
        )
        print(line)
    print()


def write_report(
    output_dir: Path,
    results: list[AcceptanceResult],
    *,
    context_bin_edges: tuple[int, ...] = DEFAULT_CONTEXT_BIN_EDGES,
    position_bin_size: int = DEFAULT_POSITION_BIN_SIZE,
) -> None:
    by_start = bin_by_start_length(results, context_bin_edges)
    _print_bucket_table("Acceptance by context length at request start", by_start)
    CsvWriter(
        output_dir / "acceptance_by_start_length.csv",
        [
            "bucket",
            "num_requests",
            "num_spec_steps",
            "num_draft_tokens",
            "num_accepted_draft_tokens",
            "draft_acceptance_rate",
            "mean_acceptance_length",
        ],
    ).append_rows(by_start)

    by_position = bin_by_position(results, position_bin_size)
    if not by_position:
        logger.warning(
            "No per-step data available; skipping acceptance-by-position report"
        )
        return
    _print_bucket_table("Acceptance by context length at token position", by_position)
    CsvWriter(
        output_dir / "acceptance_by_position.csv",
        [
            "bucket",
            "num_spec_steps",
            "num_draft_tokens",
            "num_accepted_draft_tokens",
            "draft_acceptance_rate",
            "mean_acceptance_length",
        ],
    ).append_rows(by_position)


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


def run_acceptance(
    root_url: str,
    model: str,
    candidate_bins: list[list[T]],
    to_messages: Callable[[T], list[dict]],
    output_dir: Path,
    *,
    max_concurrency: int,
    context_bin_edges: tuple[int, ...] = DEFAULT_CONTEXT_BIN_EDGES,
    position_bin_size: int = DEFAULT_POSITION_BIN_SIZE,
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
) -> None:
    """Render + generate every candidate, stream raw results, and write reports.

    *candidate_bins* is the ascending-by-length grouping returned by
    :func:`select_stratified`. Each bin is rendered/generated as a concurrent
    batch; once a whole bin renders oversized the run stops, since all larger
    bins are strictly bigger and cannot fit. The stop decision keys off actual
    render results, never a length estimate, so a fitting candidate is never
    dropped.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_log_path = output_dir / RAW_LOG_FILENAME
    results: list[AcceptanceResult] = []
    n_oversized = 0
    n_failed = 0
    checked_detailed = False
    n_candidates = sum(len(b) for b in candidate_bins)

    with (
        raw_log_path.open("w") as raw_log,
        ThreadPoolExecutor(max_workers=max_concurrency) as pool,
    ):
        for bin_items in candidate_bins:
            if not bin_items:
                continue
            futures = [
                pool.submit(
                    render_and_generate,
                    root_url,
                    model,
                    to_messages(item),
                    max_new_tokens,
                )
                for item in bin_items
            ]
            bin_all_oversized = True
            for future in as_completed(futures):
                result = future.result()
                if isinstance(result, AcceptanceResult):
                    bin_all_oversized = False
                    if not checked_detailed:
                        if not _has_detailed_metrics(result):
                            _fail_missing_detailed_metrics()
                        checked_detailed = True
                    results.append(result)
                    append_result(raw_log, result)
                elif result == "oversized":
                    n_oversized += 1
                else:
                    # A transport error leaves this sample's fit unknown, so it
                    # must not trigger the oversized early-stop.
                    bin_all_oversized = False
                    n_failed += 1
            logger.info(
                "Progress: %d/%d candidates done (%d kept, %d oversized, %d fail)",
                len(results) + n_oversized + n_failed,
                n_candidates,
                len(results),
                n_oversized,
                n_failed,
            )
            if bin_all_oversized:
                logger.info(
                    "A full length bin rendered oversized; stopping "
                    "(all larger bins are strictly bigger)."
                )
                break

    logger.info(
        "Run complete: %d/%d candidates kept (%d exceeded max context, %d failed). "
        "Raw per-request results: %s",
        len(results),
        n_candidates,
        n_oversized,
        n_failed,
        raw_log_path,
    )
    if not results:
        logger.error("No usable results collected")
        sys.exit(1)

    write_report(
        output_dir,
        results,
        context_bin_edges=context_bin_edges,
        position_bin_size=position_bin_size,
    )
