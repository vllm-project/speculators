"""Real, dependency-light tests for the response-regeneration script.

No network and no mocked HTTP: the script's seams are exercised directly against
the real downstream ``_preprocess_batch``, and ``worker`` is driven end to end
over a fake endpoint.

The script is not a package, so it is imported by path.
"""

import argparse
import asyncio
import copy
import importlib.util
import json
import time
from pathlib import Path
from typing import Any

import pytest

from speculators.data_generation import vllm_client
from speculators.data_generation.configs import DATASET_CONFIGS, DatasetConfig
from speculators.data_generation.preprocessing import _preprocess_batch
from speculators.data_generation.vllm_client import InvalidResponseError


@pytest.fixture(autouse=True)
def _no_retry_backoff(monkeypatch):
    # with_retries sleeps RETRY_BACKOFF_BASE ** attempt between retries; zero it
    # so the retry tests don't actually sleep.
    monkeypatch.setattr(vllm_client, "RETRY_BACKOFF_BASE", 0)


_SCRIPT_PATH = (
    Path(__file__).resolve().parents[3]
    / "scripts"
    / "response_regeneration"
    / "script.py"
)


def _load_regen_module():
    spec = importlib.util.spec_from_file_location("response_regen_script", _SCRIPT_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


regen = _load_regen_module()


# ---------------------------------------------------------------------------
# 1. extract_conversation over the real dataset shapes
# ---------------------------------------------------------------------------

# HuggingFaceH4/ultrachat_200k: full conversation in `messages` (role/content).
_ULTRACHAT_ROW = {
    "prompt": "Tell me about photosynthesis.",
    "messages": [
        {"role": "user", "content": "Tell me about photosynthesis."},
        {"role": "assistant", "content": "<original answer to drop>"},
        {"role": "user", "content": "Now summarize it in one line."},
        {"role": "assistant", "content": "<original answer to drop>"},
    ],
}

# A conversation that carries a system prompt (sharegpt / open-perfectblend style).
_SYSTEM_PROMPTED_ROW = {
    "conversations": [
        {"from": "system", "value": "You are a terse assistant."},
        {"from": "human", "value": "Hi"},
        {"from": "gpt", "value": "<original answer to drop>"},
    ],
}

# Magpie-Pro carries BOTH a scalar `instruction` and a `conversations` list.
_MAGPIE_ROW = {
    "instruction": "Solve 2+2.",
    "response": "4",
    "conversations": [
        {"from": "human", "value": "Solve 2+2."},
        {"from": "gpt", "value": "4"},
    ],
}

# mlabonne/open-perfectblend: `conversations` (from/value), genuinely multi-turn.
_OPEN_PERFECTBLEND_ROW = {
    "source": "blend",
    "conversations": [
        {"from": "human", "value": "h1"},
        {"from": "gpt", "value": "g1"},
        {"from": "human", "value": "h2"},
        {"from": "gpt", "value": "g2"},
    ],
}

# openai/gsm8k: no conversation field; only a scalar prompt column.
_GSM8K_ROW = {"question": "What is 6*7?", "answer": "42"}


_EXTRACT_CASES = [
    pytest.param(
        _ULTRACHAT_ROW,
        "prompt",
        [
            {"role": "user", "content": "Tell me about photosynthesis."},
            {"role": "user", "content": "Now summarize it in one line."},
        ],
        id="ultrachat_messages_multiturn_drops_assistant",
    ),
    pytest.param(
        _SYSTEM_PROMPTED_ROW,
        "prompt",
        [
            {"role": "system", "content": "You are a terse assistant."},
            {"role": "user", "content": "Hi"},
        ],
        id="system_prompt_preserved",
    ),
    pytest.param(
        _MAGPIE_ROW,
        "instruction",
        [{"role": "user", "content": "Solve 2+2."}],
        id="magpie_uses_conversations_not_instruction",
    ),
    pytest.param(
        _OPEN_PERFECTBLEND_ROW,
        "missing_field",
        [
            {"role": "user", "content": "h1"},
            {"role": "user", "content": "h2"},
        ],
        id="open_perfectblend_from_value_multiturn",
    ),
    pytest.param(
        _GSM8K_ROW,
        "question",
        [{"role": "user", "content": "What is 6*7?"}],
        id="gsm8k_single_prompt_fallback",
    ),
    pytest.param(
        {"messages": [], "prompt": "fallback prompt"},
        "prompt",
        [{"role": "user", "content": "fallback prompt"}],
        id="empty_messages_falls_back_to_prompt",
    ),
    pytest.param(
        {"messages": ["not-a-dict", {"role": "user", "content": "ok"}]},
        "prompt",
        [{"role": "user", "content": "ok"}],
        id="non_dict_turn_element_skipped",
    ),
    pytest.param(
        {"conversations": [{"role": "user", "content": "c1"}]},
        "prompt",
        [{"role": "user", "content": "c1"}],
        id="conversations_role_content_schema",
    ),
    pytest.param(
        {"messages": [{"role": "system", "content": "S"}], "prompt": "P"},
        "prompt",
        [{"role": "user", "content": "P"}],
        id="system_only_falls_back_to_prompt",
    ),
    pytest.param(
        {
            "messages": [
                {"role": "user", "content": ""},
                {"role": "user", "content": "u"},
            ]
        },
        "prompt",
        [{"role": "user", "content": "u"}],
        id="empty_content_turn_skipped",
    ),
]


@pytest.mark.parametrize(("row", "prompt_field", "expected"), _EXTRACT_CASES)
def test_extract_conversation_turns(row, prompt_field, expected):
    assert regen.extract_conversation(row, prompt_field)[0] == expected


def test_extract_conversation_no_usable_input_returns_empty():
    # No conversation field and no prompt_field value -> nothing to regenerate.
    assert regen.extract_conversation({"answer": "orphan"}, "question")[0] == []


# ---------------------------------------------------------------------------
# 2. The generation boundary is the loss mask; pre-tokenized rows pass through.
# ---------------------------------------------------------------------------


def test_build_boundary_sample_is_the_mask():
    input_ids, loss_mask = regen.build_boundary_sample([10, 11, 12, 13], [20, 21, 22])
    assert input_ids == [10, 11, 12, 13, 20, 21, 22]
    assert loss_mask == [0, 0, 0, 0, 1, 1, 1]


def test_pretokenized_rows_pass_through_preprocessing():
    # A regen row reaches training already masked: no processor, no re-masking,
    # and the review-only `conversations` field is dropped.
    input_ids, loss_mask = regen.build_boundary_sample([10, 11, 12], [20, 21])
    out = _preprocess_batch(
        {
            "input_ids": [input_ids],
            "loss_mask": [loss_mask],
            "conversations": [[{"role": "user", "content": "2+2?"}]],
        },
        processor=None,  # type: ignore[arg-type]  # passthrough never touches it
        max_length=2048,
        assistant_pattern=None,
    )
    assert out["input_ids"][0].tolist() == input_ids
    assert out["loss_mask"][0].tolist() == loss_mask
    assert "conversations" not in out


def test_pretokenized_passthrough_truncates_and_filters():
    # Truncation can cut the completion span away (all-zero mask); such a row must
    # be dropped by minimum_valid_tokens, like the tokenized path.
    kept = regen.build_boundary_sample([1, 2], [3, 4])  # fits max_length=4
    cut = regen.build_boundary_sample([1, 2, 3, 4], [5, 6])  # completion truncated off
    out = _preprocess_batch(
        {"input_ids": [kept[0], cut[0]], "loss_mask": [kept[1], cut[1]]},
        processor=None,  # type: ignore[arg-type]  # passthrough never touches it
        max_length=4,
        assistant_pattern=None,
        minimum_valid_tokens=1,
    )
    assert [t.tolist() for t in out["input_ids"]] == [[1, 2, 3, 4]]
    assert [m.tolist() for m in out["loss_mask"]] == [[0, 0, 1, 1]]


def test_pretokenized_passthrough_rejects_length_mismatch():
    # The passthrough accepts rows from any dataset carrying both columns. A row
    # whose mask is shorter than its ids must fail loudly here: the collator packs
    # each key independently, so it would otherwise shift the mask silently.
    with pytest.raises(ValueError, match="shape mismatch"):
        _preprocess_batch(
            {"input_ids": [[1, 2, 3, 4, 5]], "loss_mask": [[0, 0, 1]]},
            processor=None,  # type: ignore[arg-type]  # passthrough never touches it
            max_length=2048,
            assistant_pattern=None,
        )


# ---------------------------------------------------------------------------
# 3. Stable resume identity.
#
# The prior resume keyed rows on ``uuid or idx`` (idx = streaming enumeration
# index), which is unstable across --limit/--language-filter/order changes and
# never matched the emitted output. A conversation now fans out to one row per
# target generation, so the row ``id`` is generation-suffixed and cannot itself
# be the resume key; each row carries the conversation's ``primary_id`` for that.
# ---------------------------------------------------------------------------


def test_primary_identifier_prefers_explicit_id_over_uuid():
    assert regen._primary_identifier({"id": "abc", "uuid": "zzz"}) == "abc"


def test_primary_identifier_uuid_when_no_id():
    assert regen._primary_identifier({"uuid": "u1"}) == "u1"


def test_primary_identifier_ignores_empty_values():
    # An empty-string id is not "present"; resolution falls through to uuid.
    assert regen._primary_identifier({"id": "", "uuid": "u"}) == "u"


def test_primary_identifier_falls_back_to_content_hash():
    # No explicit id/uuid -> deterministic content hash. Nested metadata ids are
    # intentionally not consulted (the inputs that need this have no id at all).
    row = {"question": "What is 6*7?", "answer": "42"}
    reordered = {"answer": "42", "question": "What is 6*7?"}
    pid = regen._primary_identifier(row)

    assert pid.startswith("hash_")
    # Same content, different key order -> same key (JSON is sorted).
    assert regen._primary_identifier(reordered) == pid
    # Different content -> different key.
    assert regen._primary_identifier({"question": "other"}) != pid
    # A nested metadata id is not used as a source.
    assert regen._primary_identifier({"metadata": {"sample_id": 7}}).startswith("hash_")


def test_load_seen_missing_file_returns_empty(tmp_path):
    assert regen.load_seen(str(tmp_path / "nope.jsonl")) == set()


def test_load_seen_reads_id_and_skips_malformed_lines(tmp_path):
    out = tmp_path / "out.jsonl"
    out.write_text(
        "not json\n" + json.dumps({"id": "P"}) + "\n",
        encoding="utf-8",
    )
    assert regen.load_seen(str(out)) == {"P"}


def test_load_seen_ignores_rows_without_id(tmp_path):
    # A record without a top-level id contributes no resume key.
    out = tmp_path / "out.jsonl"
    out.write_text(
        json.dumps({"conversations": [], "metadata": {"idx": 3}}) + "\n",
        encoding="utf-8",
    )
    assert regen.load_seen(str(out)) == set()


def test_resume_roundtrip_hash_only_row(tmp_path):
    # A row with no explicit id resolves to a content hash; the written output
    # stores that hash as the top-level id, and load_seen must recover it so a
    # re-run skips the row. This is the exact case the old resume missed.
    row = {"question": "What is 6*7?", "answer": "42"}
    primary_id = regen._primary_identifier(row)

    out = tmp_path / "out.jsonl"
    out.write_text(
        json.dumps(
            {
                "id": primary_id,
                "conversations": [],
                "metadata": {"idx": 0},
            }
        )
        + "\n",
        encoding="utf-8",
    )

    assert primary_id in regen.load_seen(str(out))


# ---------------------------------------------------------------------------
# 4. Retry / backoff around a single request.
# ---------------------------------------------------------------------------


class _FakeResponse:
    def __init__(self, *, ok, status, text, payload):
        self.ok = ok
        self.status = status
        self._text = text
        self._payload = payload

    async def text(self):
        return self._text

    async def json(self):
        return self._payload

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False


class _FakeSession:
    """Minimal aiohttp.ClientSession stand-in: hands out queued responses."""

    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = 0

    def post(self, endpoint, json):
        self.calls += 1
        return self._responses.pop(0)


def test_post_chat_retries_transient_failure_then_succeeds():
    ok_payload = {"choices": [{"message": {"content": "hi"}}]}
    session = _FakeSession(
        [
            _FakeResponse(ok=False, status=503, text="busy", payload={}),
            _FakeResponse(ok=False, status=503, text="busy", payload={}),
            _FakeResponse(ok=True, status=200, text="", payload=ok_payload),
        ]
    )

    async def scenario():
        return await regen._post_chat(
            session,
            "http://x/v1/chat/completions",
            {"model": "m"},
            max_retries=3,
        )

    assert asyncio.run(scenario()) == ok_payload
    assert session.calls == 3


def test_post_chat_raises_after_exhausting_retries():
    session = _FakeSession(
        [_FakeResponse(ok=False, status=500, text="err", payload={}) for _ in range(3)]
    )

    async def scenario():
        with pytest.raises(RuntimeError, match="HTTP 500"):
            await regen._post_chat(
                session,
                "http://x/v1/chat/completions",
                {"model": "m"},
                max_retries=2,
            )

    asyncio.run(scenario())
    assert session.calls == 3


def test_post_chat_fails_fast_on_permanent_status():
    # A permanent (non-transient) status raises InvalidResponseError, which
    # with_retries never retries: one attempt only.
    session = _FakeSession(
        [_FakeResponse(ok=False, status=404, text="nope", payload={})]
    )

    async def scenario():
        with pytest.raises(InvalidResponseError, match="HTTP 404"):
            await regen._post_chat(
                session,
                "http://x/v1/chat/completions",
                {"model": "m"},
                max_retries=3,
            )

    asyncio.run(scenario())
    assert session.calls == 1


# ---------------------------------------------------------------------------
# 5. worker() end to end over a fake endpoint.
# ---------------------------------------------------------------------------


class _Args:
    model = "m"
    max_tokens = 16
    max_retries = 0
    sampling_params: dict[str, Any] = {}


class _NullProgress:
    def set_postfix(self, ordered_dict=None, **kwargs): ...
    def update(self, n): ...


_TWO_TURN_ITEM = {
    "idx": 41,
    "primary_id": "conv-abc",
    "turns": [
        {"role": "user", "content": "2+2?"},
        {"role": "user", "content": "3+3?"},
    ],
}


def _ok(prompt_token_ids, completion_token_ids, text):
    payload = {
        "choices": [
            {
                "message": {"content": text},
                "token_ids": completion_token_ids,
                "finish_reason": "stop",
            }
        ],
        "prompt_token_ids": prompt_token_ids,
    }
    return _FakeResponse(ok=True, status=200, text="", payload=payload)


def _run_worker(responses, tmp_path, stem):
    out_path, err_path = tmp_path / f"{stem}.jsonl", tmp_path / f"{stem}.errors.jsonl"

    async def scenario(out_fh, err_fh):
        queue: asyncio.Queue = asyncio.Queue()
        await queue.put(_TWO_TURN_ITEM)
        await queue.put(None)
        stats = {
            "ok": 0,
            "errors": 0,
            "truncated": 0,
            "requests": 0,
            "completion_tokens": 0,
            "total_request_s": 0.0,
            "start_time": time.perf_counter(),
        }
        await regen.worker(
            _FakeSession(responses),
            queue,
            _Args(),
            out_fh,
            err_fh,
            "http://x/v1/chat/completions",
            _NullProgress(),
            stats,
        )
        return stats

    with (
        out_path.open("w", encoding="utf-8") as out_fh,
        err_path.open("w", encoding="utf-8") as err_fh,
    ):
        stats = asyncio.run(scenario(out_fh, err_fh))
    return stats, out_path, err_path


def test_worker_row_identity_and_all_or_nothing_writes(tmp_path):
    # Two assistant turns -> two rows. `primary_id` is the queue item's stable id
    # (never the streaming `idx`); `id` is that id plus a generation suffix; and
    # resume keys on the former, so a re-run of this conversation is skipped.
    stats, out_path, _ = _run_worker(
        [_ok([1, 2], [3, 4], "four"), _ok([1, 2, 3, 4, 5], [6], "six")],
        tmp_path,
        "ok",
    )
    assert stats["ok"] == 1
    assert stats["errors"] == 0
    assert stats["truncated"] == 0
    assert stats["requests"] > 0
    assert stats["total_request_s"] > 0.0
    rows = [json.loads(line) for line in out_path.read_text().splitlines()]
    assert [r["id"] for r in rows] == ["conv-abc_gen0", "conv-abc_gen1"]
    assert {r["primary_id"] for r in rows} == {"conv-abc"}
    # The boundary is the mask: prompt 0s then completion 1s.
    assert rows[0]["input_ids"] == [1, 2, 3, 4]
    assert rows[0]["loss_mask"] == [0, 0, 1, 1]
    assert regen.load_seen(str(out_path)) == {"conv-abc"}

    # Turn 2 fails: turn 1's sample is discarded rather than half-written, which
    # is what lets load_seen treat one row as a finished conversation.
    stats, out_path, err_path = _run_worker(
        [
            _ok([1, 2], [3, 4], "four"),
            _FakeResponse(ok=False, status=404, text="nope", payload={}),
        ],
        tmp_path,
        "fail",
    )
    assert stats["ok"] == 0
    assert stats["errors"] == 1
    assert stats["truncated"] == 0
    assert out_path.read_text() == ""
    assert regen.load_seen(str(out_path)) == set()
    error = json.loads(err_path.read_text())
    assert error["id"] == "conv-abc"
    # The failed conversation still reports the row it had completed.
    assert error["metadata"]["generations_completed"] == 1


# ---------------------------------------------------------------------------
# 3. Tool-call regeneration: tools/results are carried in, tool-call tokens are
#    regenerated on-policy, and cached results are spliced back positionally.
# ---------------------------------------------------------------------------


def _tool_call(call_id="call_1", name="get_weather", arguments='{"city": "Tokyo"}'):
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": name, "arguments": arguments},
    }


def _response(
    *, prompt_token_ids, token_ids, content=None, tool_calls=None, finish="stop"
):
    """A vLLM chat-completion response with ``return_token_ids`` populated."""
    return {
        "choices": [
            {
                "message": {"content": content, "tool_calls": tool_calls},
                "finish_reason": finish,
                "token_ids": token_ids,
            }
        ],
        "prompt_token_ids": prompt_token_ids,
        "usage": {"completion_tokens": len(token_ids)},
    }


def _fake_post(responses):
    """A post_fn returning canned responses in order and recording sent payloads."""
    sent = []

    async def post(payload):
        sent.append(copy.deepcopy(payload))
        return responses[len(sent) - 1]

    return post, sent


def _regen(
    item, responses, *, model="m", max_tokens=64, endpoint="ep", sampling_params=None
):
    post, sent = _fake_post(responses)
    samples: list = []
    truncated = asyncio.run(
        regen.regenerate_conversation(
            post,
            item,
            model=model,
            max_tokens=max_tokens,
            endpoint=endpoint,
            sampling_params=sampling_params or {},
            samples=samples,
        )
    )
    return samples, truncated, sent


# --- ingestion: tools + tool results carried out of the raw row ---


def test_extract_tools_passthrough_and_json_string():
    tools = [{"type": "function", "function": {"name": "f"}}]
    # A list passes through; a JSON-string column (the Hermes shape) is decoded.
    assert regen.extract_tools({"tools": tools}) == tools
    assert regen.extract_tools({"tools": json.dumps(tools)}) == tools
    # Absent / empty -> None (tool-free datasets unchanged).
    assert regen.extract_tools({"prompt": "hi"}) is None
    assert regen.extract_tools({"tools": []}) is None


def test_extract_tools_raises_when_declared_but_unusable():
    # A row that advertises a tools field we cannot read as a list must fail
    # loud, not silently regenerate tool-free.
    with pytest.raises(ValueError):
        regen.extract_tools({"tools": {"name": "f"}})  # present but not a list
    with pytest.raises(ValueError):
        regen.extract_tools({"tools": "not json"})  # present but not a list
    # Absent or explicitly empty stays tool-free.
    assert regen.extract_tools({}) is None
    assert regen.extract_tools({"tools": []}) is None
    assert regen.extract_tools({"tools": ""}) is None


def test_extract_conversation_collects_ordered_results():
    # role=tool results without a <tool_response> wrapper -> content, no names.
    row = {
        "messages": [
            {"role": "user", "content": "q"},
            {"role": "assistant", "tool_calls": [_tool_call()]},
            {"role": "tool", "content": "r1"},
            {"role": "user", "content": "q2"},
            {"role": "tool", "content": "r2"},
        ]
    }
    assert regen.extract_conversation(row, None)[1] == [("r1", []), ("r2", [])]
    # from/value schema (the Hermes shape), non-dict elements skipped.
    conv = {
        "conversations": [
            {"from": "human", "value": "q"},
            "x",
            {"from": "tool", "value": "r"},
        ]
    }
    assert regen.extract_conversation(conv, None)[1] == [("r", [])]
    # Tool-free row -> no results.
    only_user = {"messages": [{"role": "user", "content": "q"}]}
    assert regen.extract_conversation(only_user, None)[1] == []


def test_extract_conversation_pairs_hermes_results_with_tool_names():
    # Hermes shape: the tool turn's <tool_response> embeds the answering tool's
    # name, which the result carries for the splice name-match guard.
    row = {
        "conversations": [
            {"from": "system", "value": "sys"},
            {"from": "human", "value": "weather?"},
            {"from": "gpt", "value": '<tool_call>{"name": "get_weather"}</tool_call>'},
            {
                "from": "tool",
                "value": '<tool_response>\n{"name": "get_weather", "content": "15C"}\n'
                "</tool_response>",
            },
            {"from": "gpt", "value": "It is 15C."},
        ]
    }
    turns, results = regen.extract_conversation(row, None)
    assert turns == [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "weather?"},
    ]
    assert len(results) == 1
    content, names = results[0]
    assert names == ["get_weather"]
    assert "<tool_response>" in content


# --- per-response validation guards (the empty-content-is-a-tool-call bug fix) ---


def test_sample_from_response_rejects_empty_and_missing_token_ids():
    # Neither content nor tool_calls -> empty generation.
    with pytest.raises(ValueError, match="empty assistant generation"):
        regen._sample_from_response(
            _response(prompt_token_ids=[1], token_ids=[2], content=None),
            prefix=[],
            conv_id="c",
            sample_index=0,
            idx=0,
            endpoint="ep",
            sampling_params={},
        )
    # Content present but the endpoint returned no token ids.
    bad = {
        "choices": [{"message": {"content": "hi"}, "token_ids": []}],
        "prompt_token_ids": [],
    }
    with pytest.raises(ValueError, match="return_token_ids"):
        regen._sample_from_response(
            bad,
            prefix=[],
            conv_id="c",
            sample_index=0,
            idx=0,
            endpoint="ep",
            sampling_params={},
        )


# --- the tool-call loop: splice, truncate, and the unchanged plain path ---


def test_sampling_params_reach_the_request_and_metadata():
    item = {"idx": 0, "primary_id": "u", "turns": [{"role": "user", "content": "hi"}]}
    responses = [_response(prompt_token_ids=[1, 2], token_ids=[3], content="hello")]
    # `max_tokens` is ours to own: a user-supplied value must not win.
    params = {"temperature": 0.6, "top_p": 0.95, "max_tokens": 1}
    samples, _, sent = _regen(item, responses, max_tokens=64, sampling_params=params)

    assert sent[0]["temperature"] == 0.6
    assert sent[0]["top_p"] == 0.95
    assert sent[0]["max_tokens"] == 64
    # Recorded for reproducibility of the generated row.
    assert samples[0]["metadata"]["sampling_params"] == params


def test_regenerate_plain_conversation_is_unchanged_and_sends_no_tools():
    item = {"idx": 0, "primary_id": "u", "turns": [{"role": "user", "content": "hi"}]}
    responses = [_response(prompt_token_ids=[1, 2], token_ids=[3], content="hello")]
    samples, truncated, sent = _regen(item, responses)

    assert not truncated
    assert len(samples) == 1
    assert samples[0]["metadata"]["is_tool_call"] is False
    # Tool-free path must not advertise tools to the endpoint.
    assert "tools" not in sent[0]


def test_regenerate_splices_cached_result_after_regenerated_call():
    item = {
        "idx": 1,
        "primary_id": "u1",
        "turns": [{"role": "user", "content": "weather?"}],
        "tools": [{"type": "function", "function": {"name": "get_weather"}}],
        "tool_results": [("15C", ["get_weather"])],
    }
    responses = [
        _response(  # target regenerates a tool call (empty content)
            prompt_token_ids=[1, 2],
            token_ids=[3, 4],
            tool_calls=[_tool_call(call_id="call_1")],
            finish="tool_calls",
        ),
        _response(  # target's final answer, conditioned on prompt+call+result
            prompt_token_ids=[1, 2, 3, 4, 5, 6],
            token_ids=[7, 8],
            content="It is 15C.",
        ),
    ]
    samples, truncated, sent = _regen(item, responses)

    assert not truncated
    assert len(samples) == 2
    # Row 0 is the on-policy tool call: its generated tokens get loss_mask 1.
    assert samples[0]["loss_mask"] == [0, 0, 1, 1]
    assert samples[0]["metadata"]["is_tool_call"] is True
    # The second request replays the regenerated call and the spliced cached result.
    assert sent[0]["tool_choice"] == "auto"
    assert sent[1]["messages"][-2]["tool_calls"] == [_tool_call(call_id="call_1")]
    assert sent[1]["messages"][-1] == {
        "role": "tool",
        "content": "15C",
        "tool_call_id": "call_1",
    }


@pytest.mark.parametrize(
    ("tool_calls", "tool_results"),
    [
        ([_tool_call()], []),  # no cached result left to splice
        # parallel call: no 1:1 pairing (result shape is (content, names))
        ([_tool_call("a"), _tool_call("b")], [("r", [])]),
    ],
    ids=["no_cached_result", "parallel_calls"],
)
def test_regenerate_truncates_but_keeps_committed_call_row(tool_calls, tool_results):
    item = {
        "idx": 2,
        "primary_id": "u",
        "turns": [{"role": "user", "content": "q"}],
        "tools": [{"type": "function", "function": {"name": "f"}}],
        "tool_results": tool_results,
    }
    responses = [
        _response(
            prompt_token_ids=[1],
            token_ids=[2],
            tool_calls=tool_calls,
            finish="tool_calls",
        )
    ]
    samples, truncated, sent = _regen(item, responses)

    assert truncated
    assert len(samples) == 1  # committed tool-call row kept; continuation stopped
    assert len(sent) == 1  # no follow-up request once we cannot continue coherently


def test_regenerate_truncates_on_tool_name_mismatch():
    # The cached result answers `get_weather`; the target regenerates a call for
    # a different tool, so it cannot be spliced coherently -> truncate.
    item = {
        "idx": 3,
        "primary_id": "u",
        "turns": [{"role": "user", "content": "q"}],
        "tools": [{"type": "function", "function": {"name": "get_time"}}],
        "tool_results": [("<tool_response>15C</tool_response>", ["get_weather"])],
    }
    responses = [
        _response(
            prompt_token_ids=[1],
            token_ids=[2],
            tool_calls=[_tool_call(call_id="c", name="get_time")],
            finish="tool_calls",
        )
    ]
    samples, truncated, sent = _regen(item, responses)

    assert truncated
    assert len(samples) == 1  # the committed call row is kept
    assert len(sent) == 1  # no splice, no follow-up request


# ---------------------------------------------------------------------------
# 6. Every shared-registry preset works on-policy (off-policy parity).
# ---------------------------------------------------------------------------


def test_prepare_row_normalizes_like_off_policy():
    # nemotron rows only become extractable through the preset's normalize_fn.
    row = {
        "input": [{"role": "user", "content": "Hi"}],
        "output": "<original answer to drop>",
    }
    _, turns, _ = regen.prepare_row(row, DATASET_CONFIGS["nemotron"])
    assert turns == [{"role": "user", "content": "Hi"}]


def test_prepare_row_applies_filter_fn():
    config = DatasetConfig(
        name="t",
        hf_path="t",
        split="train",
        filter_fn=lambda row: row["keep"],
    )
    row = {"keep": False, "conversations": [{"role": "user", "content": "Hi"}]}
    assert regen.prepare_row(row, config) is None
    _, turns, _ = regen.prepare_row(row | {"keep": True}, config)
    assert turns == [{"role": "user", "content": "Hi"}]


def test_prepare_row_merges_normalize_output_over_raw_row():
    # HF map merges columns: normalize output must not clobber the raw fallback.
    config = DatasetConfig(
        name="t",
        hf_path="t",
        split="train",
        normalize_fn=lambda row: {"conversations": []},
        prompt_field="prompt",
    )
    _, turns, _ = regen.prepare_row({"prompt": "Hi"}, config)
    assert turns == [{"role": "user", "content": "Hi"}]


def test_dataset_choice_rejects_multimodal_with_a_reason():
    with pytest.raises(argparse.ArgumentTypeError, match="does not support images"):
        regen._dataset_choice("sharegpt4v_coco")
    assert regen._dataset_choice("ultrachat") == "ultrachat"


def test_tools_and_results_are_read_from_the_normalized_row():
    # Under a normalize_fn preset the conversation only appears in `messages`
    # after normalization; reading tools off the raw row regenerates tool-free.
    config = DatasetConfig(
        name="toolcalls",
        hf_path="t",
        split="train",
        normalize_fn=lambda row: {"messages": row["input"]},
    )
    row = {
        "input": [
            {"role": "user", "content": "weather in Tokyo?"},
            {"role": "tool", "content": "sunny"},
        ],
        "tools": [{"type": "function", "function": {"name": "get_weather"}}],
    }

    normalized, _, tool_results = regen.prepare_row(row, config)

    assert regen.extract_tools(normalized) == [
        {"type": "function", "function": {"name": "get_weather"}}
    ]
    assert tool_results == [("sunny", [])]
    # the raw row hides the conversation behind `input`: results would be lost
    assert regen.extract_conversation(row, None)[1] == []
