"""Unit tests for schema normalization: role aliases, turn normalization, and
the automatic messages -> conversations rename."""

import pytest
from datasets import Dataset as HFDataset

from speculators.data_generation.configs import DATASET_CONFIGS
from speculators.data_generation.preprocessing import (
    ROLE_ALIASES,
    _normalize_conversation,
    _normalize_turn,
    _rename_messages_to_conversations,
)

# ---------------------------------------------------------------------------
# _normalize_turn / ROLE_ALIASES
# ---------------------------------------------------------------------------


@pytest.mark.sanity
@pytest.mark.parametrize(
    ("raw_role", "expected"),
    [
        ("human", "user"),
        ("user", "user"),
        ("gpt", "assistant"),
        ("assistant", "assistant"),
        ("system", "system"),
        ("tool", "tool"),
        ("observation", "tool"),
    ],
)
def test_normalize_turn_role_aliases(raw_role, expected):
    from_turn = _normalize_turn({"from": raw_role, "value": "x"})
    role_turn = _normalize_turn({"role": raw_role, "content": "x"})
    assert from_turn is not None
    assert role_turn is not None
    assert from_turn["role"] == expected
    assert role_turn["role"] == expected


@pytest.mark.sanity
def test_observation_maps_to_tool_not_user():
    # observation is a tool result; mapping it to user would break the tool-turn
    # contract (it can still carry tool_call_id).
    assert ROLE_ALIASES["observation"] == "tool"


@pytest.mark.sanity
def test_normalize_turn_unknown_role_returns_none():
    assert _normalize_turn({"role": "narrator", "content": "x"}) is None


@pytest.mark.sanity
def test_normalize_turn_mirrors_thinking_into_reasoning_content():
    # Chat templates (e.g. Qwen3) read reasoning_content, so a turn carrying only
    # 'thinking' must still populate reasoning_content (and vice versa).
    from_thinking = _normalize_turn(
        {"role": "assistant", "content": "a", "thinking": "T"}
    )
    from_reasoning = _normalize_turn(
        {"role": "assistant", "content": "a", "reasoning_content": "R"}
    )
    assert from_thinking is not None
    assert from_reasoning is not None
    assert from_thinking["thinking"] == from_thinking["reasoning_content"] == "T"
    assert from_reasoning["thinking"] == from_reasoning["reasoning_content"] == "R"

    # When both are present with different values, 'thinking' wins (or-short-circuit).
    both = _normalize_turn(
        {"role": "assistant", "content": "a", "thinking": "T", "reasoning_content": "R"}
    )
    assert both is not None
    assert both["thinking"] == both["reasoning_content"] == "T"


@pytest.mark.sanity
def test_normalize_turn_prefers_explicit_empty_value():
    # An explicit (even empty) ShareGPT "value" is respected and not overridden
    # by a stray "content"; a missing "value" falls back to "content".
    empty_value = _normalize_turn({"from": "gpt", "value": ""})
    stray = _normalize_turn({"from": "gpt", "value": "", "content": "x"})
    content_only = _normalize_turn({"role": "user", "content": "hi"})
    assert empty_value is not None
    assert stray is not None
    assert content_only is not None
    assert empty_value["content"] == ""
    assert stray["content"] == ""
    assert content_only["content"] == "hi"


@pytest.mark.sanity
def test_normalize_turn_preserves_tool_fields():
    result = _normalize_turn(
        {
            "role": "tool",
            "content": "result",
            "tool_calls": [{"id": "c1"}],
            "tool_call_id": "c1",
        }
    )
    assert result is not None
    assert result["tool_calls"] == [{"id": "c1"}]
    assert result["tool_call_id"] == "c1"


@pytest.mark.sanity
def test_normalize_conversation_empty_is_safe():
    # An empty conversation must normalize to an empty list, not raise.
    assert _normalize_conversation([]) == []


# ---------------------------------------------------------------------------
# Preset normalizers
# ---------------------------------------------------------------------------


@pytest.mark.sanity
def test_gsm8k_preset_builds_conversation_from_prompt_answer():
    normalize_fn = DATASET_CONFIGS["gsm8k"].normalize_fn
    assert normalize_fn is not None
    assert normalize_fn({"question": "Q", "answer": "A"})["conversations"] == [
        {"role": "user", "content": "Q"},
        {"role": "assistant", "content": "A"},
    ]


@pytest.mark.sanity
def test_ultrachat_preset_needs_no_normalizer():
    # ultrachat carries a 'messages' column; the automatic rename handles it.
    assert DATASET_CONFIGS["ultrachat"].normalize_fn is None


# ---------------------------------------------------------------------------
# automatic messages -> conversations rename
# ---------------------------------------------------------------------------


@pytest.mark.sanity
def test_rename_messages_to_conversations():
    ds = HFDataset.from_dict({"messages": [[{"role": "user", "content": "hi"}]]})
    out = _rename_messages_to_conversations(ds)
    assert "conversations" in out.column_names
    assert "messages" not in out.column_names


@pytest.mark.sanity
def test_rename_drops_messages_when_both_present():
    ds = HFDataset.from_dict(
        {
            "messages": [[{"role": "user", "content": "hi"}]],
            "conversations": [[{"role": "user", "content": "hi"}]],
        }
    )
    out = _rename_messages_to_conversations(ds)
    assert "messages" not in out.column_names
    assert "conversations" in out.column_names


@pytest.mark.sanity
def test_rename_noop_when_no_messages():
    ds = HFDataset.from_dict({"conversations": [[{"role": "user", "content": "hi"}]]})
    assert _rename_messages_to_conversations(ds).column_names == ["conversations"]
