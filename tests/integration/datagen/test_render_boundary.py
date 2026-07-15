"""Tests for the render-boundary loss mask and the vLLM render client.

The boundary logic (``_render_boundary_rows``) operates on token-id lists, so it
is exercised with a synthetic ``_encode_render`` -- no model, no HTTP -- which
pins every branch precisely. One end-to-end test drives ``build_eagle3_dataset``
through a real tokenizer standing in for the render endpoint (vLLM ``/render``
returns the same ids ``apply_chat_template`` does).
"""

import pytest
from datasets import Dataset as HFDataset
from transformers import AutoTokenizer

import speculators.data_generation.preprocessing as P  # noqa: N812
from speculators.data_generation import render_client as RC  # noqa: N812
from speculators.data_generation.preprocessing import build_eagle3_dataset
from speculators.data_generation.vllm_client import InvalidResponseError

TEXT_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"


def _conv(n: int) -> list[dict]:
    """A conversation of ``n`` turns alternating user/assistant from user."""
    roles = ["user", "assistant"] * ((n + 1) // 2)
    return [{"role": roles[i], "content": f"m{i}"} for i in range(n)]


def _patch_encode(monkeypatch, renders: dict[tuple[int, bool], list[int]]):
    """Stub ``_encode_render`` to return crafted ids keyed by (prefix_len, gen)."""

    def fake(conv_prefix, render_endpoint, *, add_generation_prompt, tools=None):
        return renders[(len(conv_prefix), add_generation_prompt)]

    monkeypatch.setattr(P, "_encode_render", fake)


# --------------------------------------------------------------------------- #
# _render_boundary_rows -- routing and boundary derivation                     #
# --------------------------------------------------------------------------- #
def test_packed_single_row_when_append_only(monkeypatch):
    # full(turn1) is a token-prefix of prompt(turn2): both spans live in one row.
    _patch_encode(
        monkeypatch,
        {
            (1, True): [1, 2, 3],
            (2, False): [1, 2, 3, 4, 5],
            (3, True): [1, 2, 3, 4, 5, 6, 7],
            (4, False): [1, 2, 3, 4, 5, 6, 7, 8, 9],
        },
    )
    rows = P._render_boundary_rows(_conv(4), "http://x", 100)
    assert len(rows) == 1
    assert rows[0]["input_ids"] == [1, 2, 3, 4, 5, 6, 7, 8, 9]
    assert rows[0]["loss_mask"] == [0, 0, 0, 1, 1, 0, 0, 1, 1]


def test_fanout_when_history_rewritten(monkeypatch):
    # prompt(turn2) diverges from full(turn1): chain breaks -> one row per turn.
    _patch_encode(
        monkeypatch,
        {
            (1, True): [1, 2, 3],
            (2, False): [1, 2, 3, 4, 5],
            (3, True): [1, 2, 90, 91, 92],
            (4, False): [1, 2, 90, 91, 92, 6, 7],
        },
    )
    rows = P._render_boundary_rows(_conv(4), "http://x", 100)
    assert len(rows) == 2
    assert rows[0]["input_ids"] == [1, 2, 3, 4, 5]
    assert rows[0]["loss_mask"] == [0, 0, 0, 1, 1]
    assert rows[1]["input_ids"] == [1, 2, 90, 91, 92, 6, 7]
    assert rows[1]["loss_mask"] == [0, 0, 0, 0, 0, 1, 1]


def test_scaffold_lcp_fallback(monkeypatch):
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
    rows = P._render_boundary_rows(_conv(2), "http://x", 100)
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
    with pytest.raises(P.BoundaryUnstableError):
        P._render_boundary_rows(_conv(2), "http://x", 100)


def test_assistant_first_turn_is_context_only(monkeypatch):
    # A leading assistant turn (index 0) has no prompt to bound against.
    conv = [
        {"role": "assistant", "content": "a0"},
        {"role": "user", "content": "u1"},
        {"role": "assistant", "content": "a2"},
    ]
    _patch_encode(monkeypatch, {(2, True): [1, 2, 3], (3, False): [1, 2, 3, 4]})
    rows = P._render_boundary_rows(conv, "http://x", 100)
    assert len(rows) == 1
    assert rows[0]["loss_mask"] == [0, 0, 0, 1]


def test_trailing_non_assistant_dropped(monkeypatch):
    # A conversation ending in a user turn: nothing after the last assistant
    # turn is rendered or kept.
    conv = [
        {"role": "user", "content": "u0"},
        {"role": "assistant", "content": "a1"},
        {"role": "user", "content": "u2"},
    ]
    _patch_encode(monkeypatch, {(1, True): [1, 2, 3], (2, False): [1, 2, 3, 4, 5]})
    rows = P._render_boundary_rows(conv, "http://x", 100)
    assert len(rows) == 1
    assert rows[0]["input_ids"] == [1, 2, 3, 4, 5]
    assert len(rows[0]["conv"]) == 2


def test_context_filling_window_yields_no_rows(monkeypatch):
    _patch_encode(monkeypatch, {(1, True): [1, 2, 3, 4, 5]})
    rows = P._render_boundary_rows(_conv(2), "http://x", max_length=5)
    assert rows == []


# --------------------------------------------------------------------------- #
# _append_row -- clip / filter / keep                                          #
# --------------------------------------------------------------------------- #
def test_append_row_statuses():
    results: dict[str, list] = {"input_ids": [], "loss_mask": [], "seq_len": []}
    assert P._append_row(results, [1, 2, 3], [0, 0, 0], 10, None) == "unsupervised"
    assert P._append_row(results, [1, 2, 3], [0, 1, 1], 10, 3) == "filtered"
    assert P._append_row(results, [1, 2, 3], [0, 1, 1], 10, 1) == "kept"
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


def test_render_conversation_returns_token_ids(monkeypatch):
    monkeypatch.setattr(
        RC.httpx, "post", lambda *a, **k: _Resp(200, {"token_ids": [1, 2, 3]})
    )
    out = RC.render_conversation(
        "http://x", [{"role": "user", "content": "hi"}], add_generation_prompt=True
    )
    assert out == [1, 2, 3]


def test_render_conversation_missing_token_ids_raises(monkeypatch):
    monkeypatch.setattr(RC.httpx, "post", lambda *a, **k: _Resp(200, {}))
    with pytest.raises(RC.RenderError):
        RC.render_conversation(
            "http://x", [], add_generation_prompt=False, max_retries=0
        )


def test_render_conversation_client_error_not_retried(monkeypatch):
    calls = []

    def post(*a, **k):
        calls.append(1)
        return _Resp(400, {}, "bad request")

    monkeypatch.setattr(RC.httpx, "post", post)
    with pytest.raises(InvalidResponseError):
        RC.render_conversation("http://x", [], add_generation_prompt=False)
    assert len(calls) == 1  # 4xx is deterministic: no retry


# --------------------------------------------------------------------------- #
# End-to-end through build_eagle3_dataset with a real tokenizer as the render  #
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def text_tokenizer():
    try:
        return AutoTokenizer.from_pretrained(TEXT_MODEL)
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"tokenizer {TEXT_MODEL} unavailable: {exc}")


def _install_tokenizer_render(monkeypatch, tokenizer):
    def fake(endpoint, messages, *, add_generation_prompt, tools=None, **_):
        out = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            tools=tools,
            add_generation_prompt=add_generation_prompt,
            return_dict=True,
        )["input_ids"]
        return out[0] if out and isinstance(out[0], list) else list(out)

    monkeypatch.setattr(P, "render_conversation", fake)


def test_build_eagle3_dataset_packed_end_to_end(monkeypatch, text_tokenizer):
    _install_tokenizer_render(monkeypatch, text_tokenizer)
    data = {
        "conversations": [
            [
                {"role": "user", "content": "What is 2+2?"},
                {"role": "assistant", "content": "It is 4."},
                {"role": "user", "content": "And 3+3?"},
                {"role": "assistant", "content": "It is 6."},
            ]
        ]
    }
    ds = build_eagle3_dataset(
        HFDataset.from_dict(data),
        text_tokenizer,
        max_length=2048,
        num_proc=1,
        render_endpoint="http://fake",
    )
    assert len(ds) == 1
    ids = ds[0]["input_ids"].tolist()
    mask = ds[0]["loss_mask"].tolist()
    supervised = text_tokenizer.decode(
        [t for t, m in zip(ids, mask, strict=True) if m]
    )
    # Both assistant turns supervised, neither user turn is, and the turn
    # terminator is included (the property the old regex tier lacked).
    assert "It is 4." in supervised
    assert "It is 6." in supervised
    assert "What is" not in supervised
    assert "<|im_end|>" in supervised


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
    # Rows already carrying (input_ids, loss_mask) pass through without an
    # endpoint -- the on-policy regeneration path.
    data = {"input_ids": [[1, 2, 3, 4]], "loss_mask": [[0, 0, 1, 1]]}
    ds = build_eagle3_dataset(HFDataset.from_dict(data), None, num_proc=1)
    assert len(ds) == 1
    assert ds[0]["input_ids"].tolist() == [1, 2, 3, 4]
    assert ds[0]["loss_mask"].tolist() == [0, 0, 1, 1]
