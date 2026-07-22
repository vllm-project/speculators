"""Unit tests for branches a live render endpoint cannot reach, and the client.

The happy path -- fan-out and boundary derivation against a real chat template --
is covered against a live vLLM server in ``tests/e2e/smoke/test_render_boundary``.
What is left here is only what a real server cannot produce on demand: the
scaffold fallback (needs a template that pre-fills ``<think>``), the unstable
guard (needs a template that rewrites history), and the client's error paths.
"""

import time

import pytest
from datasets import Dataset as HFDataset

from speculators.data_generation import preprocessing, render_client
from speculators.data_generation.preprocessing import build_eagle3_dataset
from speculators.data_generation.vllm_client import InvalidResponseError


def _conv(n: int) -> list[dict]:
    """A conversation of ``n`` turns alternating user/assistant from user."""
    roles = ["user", "assistant"] * ((n + 1) // 2)
    return [{"role": roles[i], "content": f"m{i}"} for i in range(n)]


def _patch_encode(monkeypatch, renders: dict[tuple[int, bool], list[int]]):
    """Stub ``_encode_render`` to return crafted ids keyed by (prefix_len, gen)."""

    def fake(conv_prefix, render_endpoint, *, add_generation_prompt, tools=None):
        return renders[(len(conv_prefix), add_generation_prompt)]

    monkeypatch.setattr(preprocessing, "_encode_render", fake)


# --------------------------------------------------------------------------- #
# _render_boundary_rows -- branches no real template reaches                    #
# --------------------------------------------------------------------------- #
def test_scaffold_lcp_fallback(monkeypatch):
    # Load-bearing, not hypothetical: DeepSeek-R1 distills pre-fill `<think>\n`
    # in the generation prompt, and Qwen3.5 pre-fills an empty `<think></think>`
    # that recorded reasoning then contradicts. Both break the prefix and land
    # here. Qwen3-0.6B (the e2e model) does not, so this stays a unit test.
    # The generation prompt ends in a scaffold token the full render replaces;
    # boundary falls back to the common prefix, valid because history agrees.
    _patch_encode(
        monkeypatch,
        {
            (1, True): [1, 2, 3, 77],  # prompt with scaffold 77
            (1, False): [1, 2, 3],  # history render
            (2, False): [1, 2, 3, 4, 5],  # full: diverges from prompt at idx 3
        },
    )
    rows = preprocessing._render_boundary_rows(_conv(2), "http://x", 100)
    assert len(rows) == 1
    assert rows[0]["loss_mask"] == [0, 0, 0, 1, 1]


def test_boundary_unstable_raises(monkeypatch):
    # Renders diverge inside history (not just the generation-prompt tail).
    _patch_encode(
        monkeypatch,
        {
            (1, True): [1, 2, 3],
            (1, False): [1, 9, 9],  # history disagrees with the full render
            (2, False): [1, 5, 6, 7],
        },
    )
    with pytest.raises(preprocessing.BoundaryUnstableError):
        preprocessing._render_boundary_rows(_conv(2), "http://x", 100)


def test_over_length_turn_does_not_drop_later_turns(monkeypatch):
    # Context is not monotonic in the turn index. Qwen3 strips `<think>` from
    # history once a later user turn arrives, so turn 3 can be over the window
    # while turn 5 -- rendered after the strip -- is 4 tokens. Stopping at the
    # first over-length turn silently discarded every trainable turn after it.
    _patch_encode(
        monkeypatch,
        {
            (1, True): [1, 2],  # turn 1 context: fits
            (2, False): [1, 2, 8, 9],
            (3, True): [1] * 12,  # turn 3 context: over max_length=10
            (5, True): [1, 2, 3, 4],  # turn 5: reasoning stripped, fits again
            (6, False): [1, 2, 3, 4, 7, 7],
        },
    )
    rows = preprocessing._render_boundary_rows(_conv(6), "http://x", 10)
    assert len(rows) == 2  # turns 1 and 5; only turn 3 is skipped
    assert rows[0]["loss_mask"] == [0, 0, 1, 1]
    assert rows[1]["loss_mask"] == [0, 0, 0, 0, 1, 1]


def test_over_length_first_turn_yields_no_rows(monkeypatch):
    # The first assistant turn's context holds no assistant message, so nothing
    # can be stripped from it -- it is the smallest the conversation ever gets.
    # Every later turn overflows too, and the conversation yields nothing.
    _patch_encode(
        monkeypatch,
        {
            (1, True): [1] * 12,
            (3, True): [1] * 15,
        },
    )
    assert preprocessing._render_boundary_rows(_conv(4), "http://x", 10) == []


# --------------------------------------------------------------------------- #
# _append_row -- clip / filter / keep                                          #
# --------------------------------------------------------------------------- #
def test_append_row_statuses():
    results: dict[str, list] = {"input_ids": [], "loss_mask": [], "seq_len": []}
    assert (
        preprocessing._append_row(results, [1, 2, 3], [0, 0, 0], 10, None)
        == "unsupervised"
    )
    assert preprocessing._append_row(results, [1, 2, 3], [0, 1, 1], 10, 3) == "filtered"
    assert preprocessing._append_row(results, [1, 2, 3], [0, 1, 1], 10, 1) == "kept"
    assert len(results["input_ids"]) == 1
    assert results["seq_len"] == [3]


# --------------------------------------------------------------------------- #
# render_client                                                                #
# --------------------------------------------------------------------------- #
class _Resp:
    def __init__(self, status_code, payload=None, text=""):
        self.status_code = status_code
        self._payload = payload
        self.text = text

    def json(self):
        return self._payload


def test_render_conversation_missing_token_ids_raises(monkeypatch):
    monkeypatch.setattr(render_client.httpx, "post", lambda *a, **k: _Resp(200, {}))
    with pytest.raises(render_client.RenderError):
        render_client.render_conversation(
            "http://x", [], add_generation_prompt=False, max_retries=0
        )


def test_render_conversation_client_error_not_retried(monkeypatch):
    calls = []

    def post(*a, **k):
        calls.append(1)
        return _Resp(400, {}, "bad request")

    monkeypatch.setattr(render_client.httpx, "post", post)
    with pytest.raises(InvalidResponseError):
        render_client.render_conversation("http://x", [], add_generation_prompt=False)
    assert len(calls) == 1  # 4xx is deterministic: no retry


@pytest.mark.parametrize("status", [408, 429])
def test_render_conversation_transient_status_is_retried(monkeypatch, status):
    calls = []

    def post(*a, **k):
        calls.append(1)
        return _Resp(status, {}, "slow down")

    monkeypatch.setattr(render_client.httpx, "post", post)
    monkeypatch.setattr(time, "sleep", lambda _: None)  # skip the backoff
    with pytest.raises(render_client.RenderError):
        render_client.render_conversation(
            "http://x", [], add_generation_prompt=False, max_retries=2
        )
    assert len(calls) == 3  # initial attempt + 2 retries


# --------------------------------------------------------------------------- #
# build_eagle3_dataset -- contracts that need no render at all                  #
# --------------------------------------------------------------------------- #
def test_build_eagle3_dataset_requires_render_endpoint():
    data = {
        "conversations": [
            [
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": "yo"},
            ]
        ]
    }
    with pytest.raises(ValueError, match="render_endpoint is required"):
        build_eagle3_dataset(HFDataset.from_dict(data), None, num_proc=1)


def test_pretokenized_dataset_skips_render():
    # The load-bearing contract: on-policy pre-tokenized rows build without a
    # render endpoint. Passthrough content (ids/mask) is covered by the regen
    # tests in test_response_regeneration.py.
    data = {"input_ids": [[1, 2, 3, 4]], "loss_mask": [[0, 0, 1, 1]]}
    ds = build_eagle3_dataset(HFDataset.from_dict(data), None, num_proc=1)
    assert len(ds) == 1
