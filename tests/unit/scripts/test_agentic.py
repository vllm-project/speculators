"""Unit tests for the agentic-trajectory replay adapter (CPU-only, no server).

Covers the pure request-building logic: message reconstruction, response
selection/thinning, and the within-batch wave packing that keeps each session's
nested prefixes together.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# agentic.py lives in scripts/evaluate/ and imports its siblings as top-level
# modules, so that directory must be on the path (scripts/ alone is not enough).
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts" / "evaluate"))

from agentic import (  # type: ignore[import-not-found]
    _pack_waves,
    _session_replays,
    _subsample_indices,
    _to_openai_messages,
)

# ---------------------------------------------------------------------------
# _to_openai_messages
# ---------------------------------------------------------------------------


def test_to_openai_messages_passes_through_tool_shape() -> None:
    tool_calls = [{"id": "c1", "type": "function", "function": {"name": "ls"}}]
    raw = [
        {"role": "system", "content": "sys"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls_json": json.dumps(tool_calls),
        },
        {"role": "tool", "content": "out", "tool_call_id": "c1"},
    ]
    out = _to_openai_messages(raw)
    assert out[0] == {"role": "system", "content": "sys"}
    # None content becomes "" and the tool_calls come back as parsed OpenAI shape.
    assert out[1] == {"role": "assistant", "content": "", "tool_calls": tool_calls}
    assert out[2] == {"role": "tool", "content": "out", "tool_call_id": "c1"}


# ---------------------------------------------------------------------------
# _subsample_indices
# ---------------------------------------------------------------------------


def test_subsample_keeps_all_when_under_cap() -> None:
    assert _subsample_indices([2, 5, 9], 8) == [2, 5, 9]
    assert _subsample_indices([2, 5, 9], 3) == [2, 5, 9]


def test_subsample_keeps_both_ends_and_order() -> None:
    picked = _subsample_indices(list(range(0, 20, 2)), 4)  # 10 indices -> 4
    assert len(picked) == 4
    assert picked[0] == 0  # first end retained
    assert picked[-1] == 18  # last end retained
    assert picked == sorted(picked)  # order (and thus nesting) preserved
    assert set(picked).issubset(set(range(0, 20, 2)))


def test_subsample_cap_one_is_deterministic() -> None:
    assert _subsample_indices([3, 6, 9, 12], 1) == [3]


# ---------------------------------------------------------------------------
# _pack_waves
# ---------------------------------------------------------------------------


def _sessions(sizes: list[int]) -> list[list[int]]:
    """Stand-in sessions: a list of `size` placeholder requests each."""
    return [[i] * size for i, size in enumerate(sizes)]


def test_pack_waves_never_splits_a_session() -> None:
    per_session = _sessions([3, 3, 3])
    waves = _pack_waves(per_session, max_concurrency=6)
    # Two 3-req sessions fill a wave of 6; the third starts a new wave.
    assert [len(w) for w in waves] == [6, 3]
    # No session is split: every wave is a concatenation of whole sessions.
    flat = [r for w in waves for r in w]
    assert flat == [r for s in per_session for r in s]


def test_pack_waves_oversized_session_is_its_own_wave() -> None:
    per_session = _sessions([10, 2])
    waves = _pack_waves(per_session, max_concurrency=5)
    assert [len(w) for w in waves] == [10, 2]


def test_pack_waves_empty() -> None:
    assert _pack_waves([], max_concurrency=8) == []


# ---------------------------------------------------------------------------
# _session_replays
# ---------------------------------------------------------------------------


def _session(messages: list[dict], **over: object) -> dict:
    base = {
        "messages_json": json.dumps(messages),
        "session_id": "sid",
        "source_dataset": "ds",
        "agent_framework": "fw",
        "total_tokens": 1234,
    }
    base.update(over)
    return base


def test_session_replays_one_request_per_assistant_response() -> None:
    messages = [
        {"role": "system", "content": "s"},
        {"role": "user", "content": "u0"},
        {"role": "assistant", "content": "a0"},  # idx 2 -> response r0
        {"role": "tool", "content": "t0"},
        {"role": "assistant", "content": "a1"},  # idx 4 -> response r1
        {"role": "user", "content": "u1"},
        {"role": "assistant", "content": "a2"},  # idx 6 -> response r2
    ]
    reps = _session_replays(7, _session(messages), max_responses=8, max_new_tokens=64)
    assert [r.request_id for r in reps] == [
        "agentic-7-r0",
        "agentic-7-r1",
        "agentic-7-r2",
    ]
    # Each replay's prompt is exactly the ground-truth prefix before its response.
    assert [len(r.messages) for r in reps] == [2, 4, 6]
    assert reps[0].messages == messages[:2]
    r0 = reps[0]
    assert r0.max_tokens == 64
    assert r0.metadata["response_index"] == 0
    assert r0.metadata["msg_index"] == 2
    assert r0.metadata["session_id"] == "sid"
    assert r0.metadata["agent_framework"] == "fw"
    assert r0.metadata["session_total_tokens"] == 1234
    assert r0.metadata["gt_response_chars"] == len("a0")


def test_session_replays_ignores_leading_assistant_and_thins() -> None:
    # A leading assistant (idx 0) can never be a response; and 5 responses
    # thinned to 2 keeps the first and last.
    messages = [{"role": "assistant", "content": "lead"}]
    for k in range(5):
        messages.append({"role": "user", "content": f"u{k}"})
        messages.append({"role": "assistant", "content": f"a{k}"})
    reps = _session_replays(0, _session(messages), max_responses=2, max_new_tokens=8)
    assert len(reps) == 2
    # First kept response is the earliest non-leading assistant (idx 2);
    # last kept is the final assistant (idx 10).
    assert reps[0].metadata["msg_index"] == 2
    assert reps[-1].metadata["msg_index"] == 10
