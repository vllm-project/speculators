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

import importlib.util
from pathlib import Path

import pytest

from speculators.data_generation.preprocessing import _normalize_conversation

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
