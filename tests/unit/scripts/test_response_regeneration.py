"""Real, dependency-light tests for the response-regeneration script.

Two seams are covered with no network and no mocked HTTP:

1. ``extract_turns`` over the *actual* schemas of the supported datasets
   (ultrachat ``messages`` role/content, magpie/open-perfectblend
   ``conversations`` from/value, gsm8k single prompt), plus the robustness
   fixes (empty conversation lists, non-dict elements, mixed schema).

2. The conversation format the worker emits, consumed by the *real* downstream
   ``_normalize_conversation``. This pins the train/infer contract the PR exists
   to protect: the system turn and per-turn reasoning survive into training, and
   the regenerated human/gpt turns map to user/assistant targets.

The script is not a package, so it is imported by path.
"""

import asyncio
import importlib.util
import json
from pathlib import Path

import pytest

from speculators.data_generation import vllm_client
from speculators.data_generation.preprocessing import _normalize_conversation
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
# 1. extract_turns over the real dataset shapes
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
def test_extract_turns(row, prompt_field, expected):
    assert regen.extract_turns(row, prompt_field) == expected


def test_extract_turns_no_usable_input_returns_empty():
    # No conversation field and no prompt_field value -> nothing to regenerate.
    assert regen.extract_turns({"answer": "orphan"}, "question") == []


# ---------------------------------------------------------------------------
# 2. Emitted conversation is consumed correctly by the real downstream
#    normaliser (the train/infer contract).
# ---------------------------------------------------------------------------

# Multi-turn conversation carrying a system prompt, used to prove the contract.
_CONTRACT_ROW = {
    "conversations": [
        {"from": "system", "value": "You are a terse assistant."},
        {"from": "human", "value": "Hi"},
        {"from": "gpt", "value": "<original answer to drop>"},
        {"from": "human", "value": "And goodbye"},
        {"from": "gpt", "value": "<original answer to drop>"},
    ],
}


def _to_worker_output(turns) -> list[dict]:
    """Mirror worker(): map system->system and user->human, and append a
    regenerated assistant turn (with per-turn reasoning) after each user turn,
    exactly as the script writes the ``conversations`` field."""
    out_convs: list[dict] = []
    for turn in turns:
        if turn["role"] == "system":
            out_convs.append({"from": "system", "value": turn["content"]})
            continue
        out_convs.append({"from": "human", "value": turn["content"]})
        out_convs.append(
            {
                "from": "gpt",
                "value": f"answer:{turn['content']}",
                "reasoning_content": f"reason:{turn['content']}",
            }
        )
    return out_convs


def test_emitted_conversation_survives_downstream_normalization():
    turns = regen.extract_turns(_CONTRACT_ROW, "prompt")
    out_convs = _to_worker_output(turns)

    normalized = _normalize_conversation(out_convs)
    roles = [m["role"] for m in normalized]

    # System preserved; human/gpt mapped to user/assistant; order intact.
    assert roles == ["system", "user", "assistant", "user", "assistant"]
    assert normalized[0]["content"] == "You are a terse assistant."

    # Per-turn reasoning survives into training (top-level metadata would not).
    assistants = [m for m in normalized if m["role"] == "assistant"]
    assert [m["content"] for m in assistants] == ["answer:Hi", "answer:And goodbye"]
    assert [m["reasoning_content"] for m in assistants] == [
        "reason:Hi",
        "reason:And goodbye",
    ]


# ---------------------------------------------------------------------------
# 3. Stable resume identity.
#
# The prior resume keyed rows on ``uuid or idx`` (idx = streaming enumeration
# index), which is unstable across --limit/--language-filter/order changes and
# never matched the emitted output (which stored ``id`` + ``metadata.idx``, not
# top-level ``uuid``/``idx``). Resume now reads a single stable
# ``metadata.primary_id`` written by the script.
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
