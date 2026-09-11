"""Agentic-coding-trajectory replay: regenerate each recorded model response.

A thin adapter, mirroring ``mrcr.py``. It turns ThoughtWorks' pre-recorded
agentic coding sessions into a set of chat requests; the generic runner
(``request_runner``) sends them and records everything, and the analysis layer
(``acceptance_report``) turns the recording into acceptance reports. Nothing
about inference, persistence, or binning lives here.

The replay
----------
A session is just a message list: a system prompt, a user task, then an
alternation of *model responses* and whatever comes back between them (tool
outputs as ``tool`` or ``user`` messages, follow-up ``user`` turns, ...). The
model responses are exactly the ``assistant`` messages -- call them X1, X2, ...
For every X_i we replay the ground-truth prefix that precedes it and regenerate:

    request i:  messages[:idx_i]  ->  regenerate X_i

so a session ``S U X1 T1 X2 T2`` yields ``[S,U] -> X1``, ``[S,U,X1,T1] -> X2``.
This is framework-agnostic: it never inspects *how* a framework encodes tool
use, only which messages are the model's own. Correctness is not graded (like
MRCR); the sessions are a source of realistic long, multi-turn agentic contexts
to study how speculative-decoding acceptance holds up turn by turn.

KV-cache locality
-----------------
Within one session every replay prefix is *nested*: ``prefix(X1)`` is a prefix
of ``prefix(X2)`` and so on. All of a session's replays go in the *same* batch
and are sent concurrently, so the server's automatic prefix cache computes each
shared prefix block once and serves it to the other responses in the batch rather
than recomputing it. Whole sessions are packed into batches of about
``max_concurrency`` requests -- never split across a batch boundary -- so several
sessions also run in parallel and keep the GPU busy (see ``_pack_waves``). This
relies only on *within-batch* sharing, not on the cache surviving between batches,
which keeps the strategy simple and independent of eviction timing.

Note -- for a hybrid Mamba/attention target under MTP (e.g. Qwen3.5-4B) the Mamba
state is only prefix-cached when checkpoints are retained densely. vLLM 0.29
makes that the default for hybrid + eagle/MTP models; on older builds pass
``--prefix-cache-retention-interval <block_size>`` (a positive multiple of the
scheduler block size) to enable it. Without it the Mamba layers recompute the
shared prefix even though the attention blocks hit, so the sharing above does not
pay off (verified empirically). The packing is correct and harmless either way.

Server requirements: ``--per-request-spec-decode-metrics detailed`` (each
response carries its own ``speculative_decoding`` block) and the render endpoint
(``VLLM_ENABLE_SCALE_OUT_ENDPOINTS=1``) for the exact-length + fit test.
"""

from __future__ import annotations

import json
import logging
import sys
from typing import TYPE_CHECKING

import pandas as pd
from acceptance_report import (
    DEFAULT_CONTEXT_BIN_EDGES,
    DEFAULT_POSITION_BIN_SIZE,
    load_spec_records,
    write_report,
)
from huggingface_hub import hf_hub_download
from request_runner import (
    TABLE_DIRNAME,
    Request,
    run_requests,
    stable_hash,
)

if TYPE_CHECKING:
    from pathlib import Path

logger = logging.getLogger("evaluate")

DATASET_REPO = "thoughtworks/agentic-coding-trajectories"
SESSIONS_FILE = "sessions.parquet"

# Default number of whole trajectories to sample. Unlike MRCR (one request per
# context length, so length must be stratified), a trajectory self-sweeps context
# length -- every response is replayed from a longer prefix than the last -- so a
# flat sample already spans the range and no per-length binning is needed.
DEFAULT_NUM_SESSIONS = 50

# Requested generation budget per replayed response; the render endpoint clips
# this to whatever context room is left after the prefix.
DEFAULT_MAX_NEW_TOKENS = 1024
# Drop replays left with less than this much generation room -- too few steps to
# measure meaningfully.
MIN_GENERATION_ROOM = 32


def _load_sessions() -> list[dict]:
    path = hf_hub_download(DATASET_REPO, SESSIONS_FILE, repo_type="dataset")
    return pd.read_parquet(path).to_dict("records")


def _to_openai_messages(raw: list[dict]) -> list[dict]:
    """Rebuild a session's stored messages into OpenAI chat format.

    Tool calls (``tool_calls_json``) and tool-result ids (``tool_call_id``) are
    already stored in OpenAI shape, so they pass straight through -- the prefix
    stays a faithful reconstruction of the real trajectory for every framework.
    """
    out: list[dict] = []
    for m in raw:
        msg: dict = {"role": m["role"], "content": m.get("content") or ""}
        if m.get("tool_calls_json"):
            msg["tool_calls"] = json.loads(m["tool_calls_json"])
        if m.get("tool_call_id"):
            msg["tool_call_id"] = m["tool_call_id"]
        out.append(msg)
    return out


def _subsample_indices(indices: list[int], cap: int) -> list[int]:
    """Evenly thin *indices* to at most *cap*, keeping order and both ends.

    Prefixes remain nested whichever responses are kept, so thinning never hurts
    cache locality; spacing them out keeps the run's prefix-length coverage.
    """
    n = len(indices)
    if cap <= 0 or n <= cap:
        return indices
    step = (n - 1) / (cap - 1) if cap > 1 else 0
    picked = sorted({round(i * step) for i in range(cap)})
    return [indices[j] for j in picked]


def _pack_waves(
    per_session: list[list[Request]], max_concurrency: int
) -> list[list[Request]]:
    """Greedily group whole sessions into waves of about *max_concurrency*.

    A session is never split, so its nested prefixes stay in one wave; sessions
    are added to the current wave until the next one would push it past
    *max_concurrency* requests, then a new wave starts. A single session larger
    than *max_concurrency* becomes its own wave.
    """
    waves: list[list[Request]] = []
    current: list[Request] = []
    for session in per_session:
        if current and len(current) + len(session) > max_concurrency:
            waves.append(current)
            current = []
        current.extend(session)
    if current:
        waves.append(current)
    return waves


def _session_replays(
    sess_idx: int,
    session: dict,
    *,
    max_responses: int,
    max_new_tokens: int,
) -> list[Request]:
    """Build the ordered replay requests for one session (one per response)."""
    messages = _to_openai_messages(json.loads(session["messages_json"]))
    # Model responses are the assistant messages; replay each from its prefix.
    # idx 0 can never be a response (nothing precedes it), so it's naturally
    # excluded by the ``idx > 0`` slice being non-empty.
    resp_indices = [
        i for i, m in enumerate(messages) if m["role"] == "assistant" and i > 0
    ]
    resp_indices = _subsample_indices(resp_indices, max_responses)

    replays: list[Request] = []
    for resp_i, idx in enumerate(resp_indices):
        prefix = messages[:idx]
        replays.append(
            Request(
                messages=prefix,
                max_tokens=max_new_tokens,
                request_id=f"agentic-{sess_idx}-r{resp_i}",
                metadata={
                    "session_id": session["session_id"],
                    "source_dataset": session["source_dataset"],
                    "agent_framework": session["agent_framework"],
                    "response_index": resp_i,
                    "msg_index": idx,
                    "prefix_messages": len(prefix),
                    "gt_response_chars": len(messages[idx].get("content") or ""),
                    "session_total_tokens": int(session["total_tokens"]),
                },
            )
        )
    return replays


def run_agentic(
    target: str,
    model_info: dict | None,
    output_dir: Path,
    max_concurrency: int,
    *,
    num_sessions: int,
    max_responses_per_session: int,
    min_session_tokens: int = 0,
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
    selection_seed: int = 0,
    position_bin_size: int = DEFAULT_POSITION_BIN_SIZE,
) -> None:
    """Select sessions, replay every model response, and report acceptance."""
    if model_info is None:
        logger.error("Could not determine served model info from %s/models", target)
        sys.exit(1)
    model_name = model_info["id"]
    logger.info(
        "Server reports model=%s max_model_len=%s",
        model_name,
        model_info.get("max_model_len"),
    )
    root_url = target.rstrip("/").removesuffix("/v1")

    logger.info("Loading agentic-coding trajectories...")
    sessions = _load_sessions()
    if not sessions:
        logger.error("Dataset %s returned no sessions", DATASET_REPO)
        sys.exit(1)
    logger.info("Loaded %d sessions", len(sessions))

    # Optionally restrict to long trajectories (using the dataset's own
    # total_tokens column) so the run exercises high context lengths -- most
    # sessions are short, so without this a flat sample rarely reaches the tail.
    eligible = [
        ir
        for ir in enumerate(sessions)
        if int(ir[1]["total_tokens"]) >= min_session_tokens
    ]
    if not eligible:
        logger.error(
            "No sessions with total_tokens >= %d (max available is %d)",
            min_session_tokens,
            max(int(s["total_tokens"]) for s in sessions),
        )
        sys.exit(1)

    # A trajectory already spans many context lengths (each response is replayed
    # from a longer prefix than the last), so -- unlike MRCR's single-length
    # requests -- no stratification by length is needed: the range comes for free.
    # Take a deterministic flat sample of whole sessions, ranked by a stable
    # content hash (pseudo-random but reproducible). Selection is
    # context-independent, and each response is fit tested individually by the
    # runner, so a larger max_model_len keeps a superset of responses (a
    # trajectory with prefixes A<B<C where B<len<C runs A and B and drops only C).
    ranked = sorted(
        eligible,
        key=lambda ir: stable_hash(f"{selection_seed}:{ir[1]['session_id']}"),
    )
    selected = ranked[:num_sessions]

    per_session = [
        _session_replays(
            sess_idx,
            sess,
            max_responses=max_responses_per_session,
            max_new_tokens=max_new_tokens,
        )
        for sess_idx, sess in selected
    ]
    per_session = [b for b in per_session if b]
    # Pack whole sessions into waves of ~max_concurrency requests: a session is
    # never split across a batch boundary, so its nested prefixes stay cached
    # together (locality, see module docstring), while several sessions share a
    # wave so the server runs them in parallel and the GPU stays busy.
    batches = _pack_waves(per_session, max_concurrency)
    n_replays = sum(len(b) for b in batches)
    if not n_replays:
        logger.error("No model responses to replay in the selected sessions")
        sys.exit(1)
    logger.info(
        "Selected %d of %d eligible sessions (>=%d tokens, %d total); replaying "
        "%d responses (<=%d per session) in %d waves",
        len(per_session),
        len(eligible),
        min_session_tokens,
        len(sessions),
        n_replays,
        max_responses_per_session,
        len(batches),
    )

    table_dir = run_requests(
        root_url,
        model_name,
        batches,
        output_dir / TABLE_DIRNAME,
        max_concurrency=max_concurrency,
        fit_test=True,
        min_generation_room=MIN_GENERATION_ROOM,
    )

    records_out = load_spec_records(table_dir)
    if not records_out:
        logger.error("No usable spec-decode results were recorded")
        sys.exit(1)
    write_report(
        output_dir,
        records_out,
        context_bin_edges=DEFAULT_CONTEXT_BIN_EDGES,
        position_bin_size=position_bin_size,
    )
    logger.info(
        "Render the acceptance figure with:\n"
        "  python plot.py acceptance --table %s --output %s",
        table_dir,
        output_dir / "acceptance.png",
    )
