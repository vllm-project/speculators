"""
Tests for render-boundary loss masks (see ``_render_boundary_rows``): routing
(packed vs per-turn fan-out), turn-terminator supervision, thinking models,
the generation-prompt scaffold fallback, truncation, instability failure, and
frozen real-data fixtures.
"""

import json
from pathlib import Path

import pytest
import torch
from transformers import AutoTokenizer

from speculators.data_generation.preprocessing import (
    BoundaryUnstableError,
    _normalize_conversation,
    _preprocess_batch,
    _render_boundary_rows,
    get_tokenizer,
    load_processor,
)

FIXTURES = Path(__file__).parent / "fixtures" / "preprocess"
MAX_LENGTH = 8192

# All ungated; Qwen2/2.5 pack (append-only), Qwen3/3.5 are thinking templates
PACKED_MODELS = ["Qwen/Qwen2-0.5B-Instruct", "Qwen/Qwen2.5-0.5B-Instruct"]
THINKING_MODEL = "Qwen/Qwen3-0.6B"
SCAFFOLD_MODEL = "Qwen/Qwen3.5-0.8B"  # generation prompt pre-fills <think> scaffold

PLAIN_CONV = [
    {"role": "user", "content": "What is 2+2?"},
    {"role": "assistant", "content": "The answer is 4."},
    {"role": "user", "content": "And 3+3?"},
    {"role": "assistant", "content": "That makes 6."},
]

REASONING_CONV = [
    {"role": "user", "content": "What is 2+2?"},
    {
        "role": "assistant",
        "content": "The answer is 4.",
        "reasoning_content": "Simple arithmetic: two plus two.",
    },
    {"role": "user", "content": "And 3+3?"},
    {
        "role": "assistant",
        "content": "That makes 6.",
        "reasoning_content": "Again arithmetic: three plus three.",
    },
]


def _supervised_text(tok, row) -> str:
    return tok.decode(
        [t for t, m in zip(row["input_ids"], row["loss_mask"], strict=True) if m]
    )


def _context_text(tok, row) -> str:
    return tok.decode(
        [t for t, m in zip(row["input_ids"], row["loss_mask"], strict=True) if not m]
    )


@pytest.mark.sanity
@pytest.mark.parametrize("model", [*PACKED_MODELS, THINKING_MODEL, SCAFFOLD_MODEL])
def test_boundary_coverage_matrix(model):
    """Every supported template derives boundary rows: masks aligned, responses
    supervised with their turn terminator, user text never supervised."""
    processor = load_processor(model, trust_remote_code=True)
    tok = get_tokenizer(processor)

    rows = _render_boundary_rows(
        _normalize_conversation(PLAIN_CONV), processor, MAX_LENGTH
    )
    assert rows, "no boundary rows produced"

    all_supervised = ""
    for row in rows:
        assert len(row["input_ids"]) == len(row["loss_mask"])
        assert sum(row["loss_mask"]) > 0
        all_supervised += _supervised_text(tok, row)

    assert "The answer is 4." in all_supervised
    assert "That makes 6." in all_supervised
    assert "What is 2+2?" not in all_supervised
    assert "<|im_end|>" in all_supervised, "turn terminator must be supervised"


@pytest.mark.sanity
def test_packed_route_single_row_all_turns():
    """Append-only templates pack a multi-turn conversation into one row with
    every assistant turn supervised: content + <|im_end|>, headers masked."""
    tok = AutoTokenizer.from_pretrained(PACKED_MODELS[0])
    rows = _render_boundary_rows(_normalize_conversation(PLAIN_CONV), tok, MAX_LENGTH)

    assert len(rows) == 1, "append-only template must pack to a single row"
    row = rows[0]

    im_end = tok.convert_tokens_to_ids("<|im_end|>")
    supervised_eot = [
        i
        for i, (t, m) in enumerate(zip(row["input_ids"], row["loss_mask"], strict=True))
        if t == im_end and m
    ]
    assert len(supervised_eot) == 2, "each assistant turn's terminator supervised"

    supervised = _supervised_text(tok, row)
    # Each turn's span is its content plus the terminator; anything beyond is
    # template whitespace, not frozen here (template revisions are unpinned).
    residue = supervised.replace("The answer is 4.<|im_end|>", "").replace(
        "That makes 6.<|im_end|>", ""
    )
    assert residue.strip() == ""
    assert "assistant" in _context_text(tok, row), "role headers stay context"


@pytest.mark.sanity
def test_fanout_route_reasoning_faithful_context():
    """Thinking templates rewrite history (strip past <think>), so multi-turn
    fans out per turn: each row supervises its own thinking against the same
    stripped-history context inference would see."""
    tok = AutoTokenizer.from_pretrained(THINKING_MODEL)
    rows = _render_boundary_rows(
        _normalize_conversation(REASONING_CONV), tok, MAX_LENGTH
    )

    assert len(rows) == 2, "history-rewriting template must fan out per turn"

    # Turn 1 supervises its own thinking
    sup1 = _supervised_text(tok, rows[0])
    assert "Simple arithmetic" in sup1
    assert sup1.rstrip().endswith("<|im_end|>")

    # Turn 2: current thinking supervised, turn-1 thinking stripped from context
    sup2 = _supervised_text(tok, rows[1])
    ctx2 = _context_text(tok, rows[1])
    assert "Again arithmetic" in sup2
    assert "Simple arithmetic" not in ctx2, "history thinking must be stripped"
    assert "Simple arithmetic" not in sup2
    assert "The answer is 4." in ctx2, "history content stays as context"


@pytest.mark.sanity
def test_thinking_template_plain_content_fans_out():
    """Qwen3 renders an empty think scaffold on the last turn only, so even
    plain-content conversations rewrite history and fan out -- each row then
    supervises the scaffold the model would actually generate."""
    tok = AutoTokenizer.from_pretrained(THINKING_MODEL)
    rows = _render_boundary_rows(_normalize_conversation(PLAIN_CONV), tok, MAX_LENGTH)

    assert len(rows) == 2
    for row in rows:
        assert "<think>" in _supervised_text(tok, row)


@pytest.mark.sanity
def test_scaffold_generation_prompt_lcp_fallback():
    """Qwen3.5's generation prompt pre-fills an empty <think> scaffold. With
    recorded reasoning the strict prefix check fails, and the boundary falls
    back to the longest common prefix -- supervising exactly what generation
    would produce after the auto-opened <think>."""
    processor = load_processor(SCAFFOLD_MODEL, trust_remote_code=True)
    tok = get_tokenizer(processor)

    conv = _normalize_conversation(REASONING_CONV[:2])  # single turn
    rows = _render_boundary_rows(conv, processor, MAX_LENGTH)
    assert len(rows) == 1
    supervised = _supervised_text(tok, rows[0])
    context = _context_text(tok, rows[0])

    assert "Simple arithmetic" in supervised, "reasoning must be supervised"
    assert supervised.rstrip().endswith("<|im_end|>")
    assert context.rstrip().endswith("<think>"), (
        "the auto-opened scaffold prefix stays context, like the serving "
        "engine's generation prompt"
    )

    # Plain content: the whole pre-filled scaffold is context, answer supervised
    rows_plain = _render_boundary_rows(
        _normalize_conversation(PLAIN_CONV[:2]), processor, MAX_LENGTH
    )
    supervised_plain = _supervised_text(tok, rows_plain[0])
    assert "<think>" not in supervised_plain
    assert "The answer is 4." in supervised_plain


@pytest.mark.sanity
def test_assistant_first_turn_kept_as_context_only():
    """A context-free leading assistant turn (common in ShareGPT dumps) has no
    boundary to render; it stays context for later turns, unsupervised."""
    tok = AutoTokenizer.from_pretrained(PACKED_MODELS[0])
    conv = [
        {"from": "gpt", "value": "Orphaned answer without a prompt."},
        {"from": "human", "value": "And 3+3?"},
        {"from": "gpt", "value": "That makes 6."},
    ]
    rows = _render_boundary_rows(_normalize_conversation(conv), tok, MAX_LENGTH)

    assert len(rows) == 1
    supervised = _supervised_text(tok, rows[0])
    assert "Orphaned answer" not in supervised
    assert "That makes 6." in supervised
    assert "Orphaned answer" in _context_text(tok, rows[0])


@pytest.mark.sanity
def test_trailing_non_assistant_messages_dropped():
    """Nothing after the last assistant turn is supervised or conditioned on,
    so trailing user messages are dropped from the row."""
    tok = AutoTokenizer.from_pretrained(PACKED_MODELS[0])
    conv = [
        *PLAIN_CONV,
        {"role": "user", "content": "TRAILING_NEVER_ANSWERED"},
    ]
    rows = _render_boundary_rows(_normalize_conversation(conv), tok, MAX_LENGTH)

    assert len(rows) == 1
    full_text = tok.decode(rows[0]["input_ids"])
    assert "TRAILING_NEVER_ANSWERED" not in full_text


@pytest.mark.sanity
def test_boundary_unstable_raises_and_batch_skips():
    """A template that rewrites history inside a turn boundary cannot yield a
    boundary mask: loud error, and _preprocess_batch skips the conversation."""

    class UnstableTok:
        chat_template = "sentinel"

        def apply_chat_template(
            self, conv, tokenize=False, tools=None, add_generation_prompt=False
        ):
            # Message count baked into the render start: appending a turn
            # rewrites everything before it.
            text = f"N{len(conv)}:" + "".join(t["content"] for t in conv)
            return text + ("<A>" if add_generation_prompt else "")

        def __call__(self, text, add_special_tokens=False):
            return {"input_ids": [ord(c) for c in text]}

    conv = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "yo"},
    ]
    with pytest.raises(BoundaryUnstableError):
        _render_boundary_rows(conv, UnstableTok(), MAX_LENGTH)  # type: ignore[arg-type]

    results = _preprocess_batch(
        {"conversations": [conv]},
        UnstableTok(),  # type: ignore[arg-type]
        max_length=64,
    )
    assert results["input_ids"] == []


@pytest.mark.sanity
def test_truncation_keeps_partial_supervision():
    """A window cutting mid-response keeps the in-window supervised prefix
    (the old regex path lost these rows to all-zero masks)."""
    tok = AutoTokenizer.from_pretrained(PACKED_MODELS[0])
    conv = [
        {"role": "user", "content": "Count."},
        {"role": "assistant", "content": " ".join(str(i) for i in range(200))},
    ]
    rows = _render_boundary_rows(_normalize_conversation(conv), tok, MAX_LENGTH)
    (row,) = rows
    boundary = row["loss_mask"].index(1)
    cut = boundary + 5  # five supervised tokens fit in the window

    results = _preprocess_batch({"conversations": [conv]}, tok, max_length=cut)
    assert len(results["input_ids"]) == 1
    assert int(results["loss_mask"][0].sum()) == 5

    # A window ending before the response has nothing to supervise: dropped.
    results = _preprocess_batch({"conversations": [conv]}, tok, max_length=boundary)
    assert results["input_ids"] == []


# ---------------------------------------------------------------------------
# Frozen real-data fixtures (see fixtures/preprocess/PROVENANCE.json)
# ---------------------------------------------------------------------------


def _load_fixture(name: str) -> tuple[list[dict], dict]:
    """Return (conversations, raw record) for a frozen fixture row."""
    with (FIXTURES / name).open(encoding="utf-8") as fh:
        record = json.loads(fh.readline())
    conversations = record.get("conversations") or record.get("messages")
    assert conversations is not None
    return conversations, record


@pytest.mark.sanity
@pytest.mark.parametrize(
    ("fixture", "model"),
    [
        ("sharegpt_multiturn.jsonl", PACKED_MODELS[0]),
        ("ultrachat_messages.jsonl", PACKED_MODELS[0]),
        ("hermes_toolcall.jsonl", PACKED_MODELS[1]),
        ("dolphin_reasoning.jsonl", THINKING_MODEL),
        ("dolphin_reasoning.jsonl", SCAFFOLD_MODEL),
    ],
)
def test_fixture_rows_supervised(fixture, model):
    """Real frozen rows produce supervised boundary rows with terminators in
    the mask and no user text leaking into supervision."""
    conversations, record = _load_fixture(fixture)
    processor = load_processor(model, trust_remote_code=True)
    tok = get_tokenizer(processor)

    examples: dict = {"conversations": [conversations]}
    if record.get("tools"):
        examples["tools"] = [record["tools"]]
    results = _preprocess_batch(examples, processor, max_length=8192)
    assert len(results["input_ids"]) >= 1

    norm = _normalize_conversation(conversations)
    user_texts = [
        t["content"]
        for t in norm
        if t["role"] == "user" and isinstance(t["content"], str)
    ]
    for ids, mask in zip(results["input_ids"], results["loss_mask"], strict=True):
        assert len(ids) == len(mask)
        assert int(mask.sum()) > 0
        supervised = tok.decode(ids[mask.to(torch.bool)])
        assert "<|im_end|>" in supervised
        for user_text in user_texts:
            assert user_text[:80] not in supervised


@pytest.mark.sanity
def test_fixture_dolphin_reasoning_supervises_thinking():
    """The frozen dolphin-r1 row's reasoning is supervised, matching what the
    target generates at inference; truncated windows keep a supervised prefix
    instead of the old all-zero mask."""
    conversations, _record = _load_fixture("dolphin_reasoning.jsonl")
    processor = load_processor(SCAFFOLD_MODEL, trust_remote_code=True)
    tok = get_tokenizer(processor)

    norm = _normalize_conversation(conversations)
    reasoning = next(
        t.get("reasoning_content") or t.get("thinking")
        for t in norm
        if t["role"] == "assistant"
    )
    assert reasoning, "fixture must carry reasoning content"

    rows = _render_boundary_rows(norm, processor, MAX_LENGTH)
    supervised = "".join(_supervised_text(tok, r) for r in rows)
    assert reasoning[:60] in supervised

    # Regression: a window cutting into the reasoning keeps partial supervision
    results = _preprocess_batch(
        {"conversations": [conversations]}, processor, max_length=256
    )
    if results["input_ids"]:
        assert int(results["loss_mask"][0].sum()) > 0
