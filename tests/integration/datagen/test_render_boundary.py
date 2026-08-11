"""Unit tests for branches a live render endpoint cannot reach, and the client.

The happy path -- fan-out and boundary derivation against a real chat template --
is covered against a live vLLM server in ``tests/e2e/smoke/test_render_boundary``.
What is left here is only what a real server cannot produce on demand: the
scaffold fallback (needs a template that pre-fills ``<think>``), the unstable
guard (needs a template that rewrites history), and the client's error paths.
"""

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import cast

import pytest
from datasets import Dataset as HFDataset
from datasets import load_dataset
from transformers import ProcessorMixin

from speculators.data_generation import preprocessing, render_client
from speculators.data_generation.preprocessing import (
    ProcessorLike,
    build_speculator_training_dataset,
)
from speculators.data_generation.vllm_client import InvalidResponseError

# Neither build path below reads the processor: the missing-endpoint guard
# raises before it is used, and speculator-format rows skip rendering entirely.
NO_PROCESSOR = cast("ProcessorLike", None)


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
    # Qwen3 strips `<think>` from history once a later user turn arrives, so
    # turn 3 can exceed the window while turn 5 fits again.
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
    # No assistant message in the first turn's context, so nothing can be
    # stripped: it is the smallest the conversation ever gets.
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
# build_speculator_training_dataset -- contracts that need no render at all   #
# --------------------------------------------------------------------------- #
def test_build_speculator_training_dataset_requires_render_endpoint():
    data = {
        "conversations": [
            [
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": "yo"},
            ]
        ]
    }
    with pytest.raises(ValueError, match="render_endpoint is required"):
        build_speculator_training_dataset(
            HFDataset.from_dict(data), NO_PROCESSOR, num_proc=1
        )


def test_pretokenized_dataset_skips_render():
    # The load-bearing contract: speculator-format rows build without a render
    # endpoint. Passthrough content (ids/mask) is covered by the regeneration
    # tests in test_response_regeneration.py.
    data = {"input_ids": [[1, 2, 3, 4]], "loss_mask": [[0, 0, 1, 1]]}
    ds = build_speculator_training_dataset(
        HFDataset.from_dict(data), NO_PROCESSOR, num_proc=1
    )
    assert len(ds) == 1


# --------------------------------------------------------------------------- #
# multiproc map -- Arrow schema alignment of the stored messages column        #
# --------------------------------------------------------------------------- #
class _FakeMMProcessor(ProcessorMixin):
    """Multimodal stand-in: the builder only isinstance-checks the processor
    to decide whether the ``messages`` column is kept."""

    def __init__(self):
        pass


def _stub_token_count(messages: list[dict], add_generation_prompt: bool) -> int:
    """Deterministic stand-in tokenization, monotonic in content length so a
    full render always extends its generation-prompt render."""
    n = sum(2 + len(str(m.get("content", ""))) // 10 for m in messages)
    return n + bool(add_generation_prompt)


@pytest.fixture
def render_stub():
    """A local render endpoint reachable from ``map`` worker processes, where
    a monkeypatch would not survive the spawn."""

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            n = _stub_token_count(body["messages"], body["add_generation_prompt"])
            payload = json.dumps({"token_ids": list(range(n))}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, *args):
            """Keep test output clean."""

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_port}"
    server.shutdown()
    thread.join()


@pytest.mark.sanity
def test_multiproc_heterogeneous_conversations(render_stub, tmp_path):
    """Shards with different conversation shapes must still concatenate.

    With ``num_proc > 1`` each worker shard used to infer its own Arrow schema
    for the ``messages`` column. Rows with plain-string content landing in one
    shard and typed-part-list content (or tool calls) in another made the
    schemas disagree, and ``map`` crashed with "The features can't be
    aligned". Storing ``messages`` as a JSON string keeps the schema
    deterministic regardless of sharding.
    """
    img_path = str(tmp_path / "blank.png")  # never opened: the stub renders

    text_conv = [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi!"},
    ]
    mm_conv = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Describe this image."},
                {"type": "image", "path": img_path},
            ],
        },
        {"role": "assistant", "content": [{"type": "text", "text": "Blank."}]},
    ]
    # Load through the json builder like prepare_data.py does: from_dict would
    # reject the heterogeneous conversations column outright.
    data_file = tmp_path / "convs.jsonl"
    with data_file.open("w") as f:
        for conv in (text_conv, text_conv, mm_conv, mm_conv):
            f.write(json.dumps({"conversations": conv}) + "\n")
    dataset = load_dataset("json", data_files=str(data_file), split="train")

    result = build_speculator_training_dataset(
        dataset,
        _FakeMMProcessor(),
        max_length=2048,
        num_proc=2,
        render_endpoint=render_stub,
    )

    assert len(result) == 4
    decoded = [json.loads(m) for m in result["messages"]]
    assert decoded[0][-1] == {"role": "assistant", "content": "Hi!"}
    assert decoded[2][0]["content"][1] == {
        "type": "image_url",
        "image_url": {"url": f"file://{img_path}"},
    }
