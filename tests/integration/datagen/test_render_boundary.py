"""Unit tests for branches a live render endpoint cannot reach, and the client.

The happy path -- fan-out and boundary derivation against a real chat template --
is covered against a live vLLM server in ``tests/e2e/smoke/test_render_boundary``.
What is left here is only what a real server cannot produce on demand: the
scaffold fallback (needs a template that pre-fills ``<think>``), the unstable
guard (needs a template that rewrites history), and the client's error paths.
"""

import json
import os
import time
from typing import cast

import httpx
import pytest
import torch
from datasets import Dataset as HFDataset
from datasets import concatenate_datasets, load_from_disk
from transformers import ProcessorMixin

from speculators.data_generation import preprocessing, render_client
from speculators.data_generation.preprocessing import (
    ProcessorLike,
    build_speculator_training_dataset,
)
from speculators.data_generation.vllm_client import InvalidResponseError
from speculators.train.data import _add_document_ids, _pad_samples
from speculators.train.data_types import DatasetRecord
from speculators.train.hidden_states_source import build_client_item

# Neither build path below reads the processor: the missing-endpoint guard
# raises before it is used, and speculator-format rows skip rendering entirely.
NO_PROCESSOR = cast("ProcessorLike", None)


def _conv(n: int) -> list[dict]:
    """A conversation of ``n`` turns alternating user/assistant from user."""
    roles = ["user", "assistant"] * ((n + 1) // 2)
    return [{"role": roles[i], "content": f"m{i}"} for i in range(n)]


def _patch_encode(monkeypatch, renders: dict[tuple[int, bool], list[int]]):
    """Stub ``_encode_render`` to return crafted ids keyed by (prefix_len, gen)."""

    def fake(
        conv_prefix,
        render_endpoint,
        *,
        add_generation_prompt,
        max_length=None,
        tools=None,
    ):
        return renders[(len(conv_prefix), add_generation_prompt)]

    monkeypatch.setattr(preprocessing, "_encode_render", fake)


def test_encode_render_sends_training_window(monkeypatch):
    request: dict[str, object] = {}

    def fake_render(
        endpoint,
        messages,
        *,
        add_generation_prompt,
        tools=None,
        truncate_prompt_tokens=None,
        truncation_side=None,
    ):
        request.update(
            truncate_prompt_tokens=truncate_prompt_tokens,
            truncation_side=truncation_side,
        )
        return [1, 2]

    monkeypatch.setattr(preprocessing, "render_conversation", fake_render)

    result = preprocessing._encode_render(
        [{"role": "user", "content": "hi"}],
        "http://x",
        add_generation_prompt=True,
        max_length=10,
    )

    assert result == [1, 2]
    assert request == {
        "truncate_prompt_tokens": 10,
        "truncation_side": "right",
    }


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
    assert results["loss_mask"][0].dtype == torch.bool
    assert results["loss_mask"][0].tolist() == [False, True, True]


def test_append_boundary_rows_counts_rows_at_limit_as_maybe_truncated():
    results: dict[str, list] = {"input_ids": [], "loss_mask": [], "seq_len": []}
    rows: list[preprocessing.BoundaryRow] = [
        {
            "input_ids": [1, 2, 3, 4],
            "loss_mask": [0, 1, 1, 1],
            "conv": _conv(2),
        }
    ]

    assert preprocessing._append_boundary_rows(results, rows, 4, None) == (1, 0, 1)


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


def test_render_conversation_sends_truncation_options(monkeypatch):
    request_body = {}

    def post(url, *, json, timeout):
        request_body.update(json)
        return _Resp(200, {"token_ids": [1, 2]})

    monkeypatch.setattr(render_client, "_post", post)
    result = render_client.render_conversation(
        "http://x",
        [],
        add_generation_prompt=False,
        truncate_prompt_tokens=10,
        truncation_side="right",
        max_retries=0,
    )

    assert result == [1, 2]
    assert request_body["truncate_prompt_tokens"] == 10
    assert request_body["truncation_side"] == "right"


def test_render_conversation_missing_token_ids_raises(monkeypatch):
    monkeypatch.setattr(render_client, "_post", lambda *a, **k: _Resp(200, {}))
    with pytest.raises(render_client.RenderError):
        render_client.render_conversation(
            "http://x", [], add_generation_prompt=False, max_retries=0
        )


def test_render_conversation_client_error_not_retried(monkeypatch):
    calls = []

    def post(*a, **k):
        calls.append(1)
        return _Resp(400, {}, "bad request")

    monkeypatch.setattr(render_client, "_post", post)
    with pytest.raises(InvalidResponseError):
        render_client.render_conversation("http://x", [], add_generation_prompt=False)
    assert len(calls) == 1  # 4xx is deterministic: no retry


def test_post_pools_client_per_process(monkeypatch):
    created = []

    class _FakeClient:
        def __init__(self):
            created.append(self)

        def post(self, url, *, json, timeout):
            return _Resp(200, {"token_ids": [1]})

    monkeypatch.setattr(render_client.httpx, "Client", _FakeClient)
    monkeypatch.setattr(render_client, "_client", None)
    monkeypatch.setattr(render_client, "_client_pid", None)

    render_client.render_conversation("http://x", [], add_generation_prompt=False)
    render_client.render_conversation("http://x", [], add_generation_prompt=False)
    assert len(created) == 1  # one client reused within the process

    # A forked map worker must not reuse the parent's pooled sockets.
    monkeypatch.setattr(render_client.os, "getpid", lambda: -1)
    render_client.render_conversation("http://x", [], add_generation_prompt=False)
    assert len(created) == 2


@pytest.mark.parametrize("status", [408, 429])
def test_render_conversation_transient_status_is_retried(monkeypatch, status):
    calls = []

    def post(*a, **k):
        calls.append(1)
        return _Resp(status, {}, "slow down")

    monkeypatch.setattr(render_client, "_post", post)
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


@pytest.mark.parametrize("mask", [[0, 0, 1, 1], [False, False, True, True]])
def test_loss_mask_stays_boolean_after_save_load_and_collation(tmp_path, mask):
    dataset = build_speculator_training_dataset(
        HFDataset.from_dict({"input_ids": [[1, 2, 3, 4]], "loss_mask": [mask]}),
        NO_PROCESSOR,
        num_proc=1,
    )
    assert dataset[0]["loss_mask"].dtype == torch.bool
    dataset.save_to_disk(tmp_path / "dataset")

    restored = load_from_disk(tmp_path / "dataset")
    row = restored[0]
    assert row["loss_mask"].dtype == torch.bool
    assert row["loss_mask"].tolist() == [False, False, True, True]
    assert row["input_ids"].dtype == torch.long
    assert row["input_ids"].tolist() == [1, 2, 3, 4]

    batch = _pad_samples(
        [
            {
                "input_ids": row["input_ids"],
                "loss_mask": row["loss_mask"],
                "lengths": torch.tensor([4]),
            }
        ],
        max_len=6,
        dtype=torch.bfloat16,
    )
    _add_document_ids(batch, 6)
    assert batch["loss_mask"].dtype == torch.bool
    assert batch["loss_mask"].tolist() == [[False, False, True, True, False, False]]


@pytest.mark.parametrize("status", [404, 405])
def test_preprocess_aborts_when_render_route_disappears(monkeypatch, status):
    requests = []

    def handle(request):
        body = json.loads(request.content)
        requests.append(body)
        if body["messages"][0]["content"] == "missing":
            return httpx.Response(status, text="render route unavailable")
        ids = [1, 2] if body["add_generation_prompt"] else [1, 2, 3]
        return httpx.Response(200, json={"token_ids": ids})

    conversations = [
        [{"role": "user", "content": text}, {"role": "assistant", "content": "ok"}]
        for text in ["good", "missing"]
    ]
    with httpx.Client(transport=httpx.MockTransport(handle)) as client:
        monkeypatch.setattr(render_client, "_client", client)
        monkeypatch.setattr(render_client, "_client_pid", os.getpid())
        with pytest.raises(InvalidResponseError, match=str(status)):
            preprocessing._preprocess_batch(
                {"conversations": conversations}, False, "http://render.test", 2048
            )
    assert len(requests) == 3  # two valid renders, then one permanent route failure


@pytest.mark.parametrize("status", [200, 400, 422])
def test_preprocess_keeps_conversation_specific_render_error_policy(
    monkeypatch, status
):
    def handle(request):
        body = json.loads(request.content)
        if status != 200 and body["messages"][0]["content"] == "bad":
            return httpx.Response(status, text="bad conversation")
        ids = [1, 2] if body["add_generation_prompt"] else [1, 2, 3]
        return httpx.Response(200, json={"token_ids": ids})

    conversations = [
        [{"role": "user", "content": text}, {"role": "assistant", "content": "ok"}]
        for text in ["good", "bad"]
    ]
    with httpx.Client(transport=httpx.MockTransport(handle)) as client:
        monkeypatch.setattr(render_client, "_client", client)
        monkeypatch.setattr(render_client, "_client_pid", os.getpid())
        result = preprocessing._preprocess_batch(
            {"conversations": conversations}, False, "http://render.test", 2048
        )
    assert len(result["input_ids"]) == (2 if status == 200 else 1)
    assert result["input_ids"][0].tolist() == [1, 2, 3]
    assert result["loss_mask"][0].tolist() == [False, False, True]


class _ProcessorMixinStub(ProcessorMixin):
    """Routes preprocessing down the multimodal path; never called."""

    def __init__(self):
        pass


def _build_with_processor(monkeypatch, conversations: list[list[dict]]) -> HFDataset:
    # One row per conversation; the conv itself is what `messages` would store.
    monkeypatch.setattr(
        preprocessing,
        "_render_conversation_rows",
        lambda conv, *a, **k: [
            {"input_ids": [1, 2, 3], "loss_mask": [0, 1, 1], "conv": conv}
        ],
    )
    return build_speculator_training_dataset(
        HFDataset.from_dict({"conversations": conversations}),
        cast("ProcessorLike", _ProcessorMixinStub()),
        num_proc=1,
        render_endpoint="http://x",
    )


def test_processor_text_only_rows_do_not_store_messages(monkeypatch):
    # The trainer only forwards `messages` for multimodal rows, so a text-only
    # prep through a processor-based model should not keep the column.
    dataset = _build_with_processor(monkeypatch, [_conv(2), _conv(2)])
    assert len(dataset) == 2
    assert "messages" not in dataset.column_names
    assert "is_multimodal" not in dataset.column_names


def test_processor_multimodal_rows_keep_messages(monkeypatch):
    image = {"type": "image", "url": "http://x/a.png", "text": None}
    text = {"type": "text", "url": None, "text": "what is this?"}
    conv = [
        {"role": "user", "content": [image, text]},
        {"role": "assistant", "content": [{**text, "text": "a cat"}]},
    ]
    dataset = _build_with_processor(monkeypatch, [conv])
    assert "is_multimodal" not in dataset.column_names
    user_parts = dataset[0]["messages"][0]["content"]
    assert user_parts[0]["image_url"] == {"url": "http://x/a.png"}
    record = DatasetRecord.from_arrow_row(dataset[0], storage_index=0)
    assert build_client_item(record)["messages"] == record.messages


def test_processor_text_part_lists_do_not_store_messages(monkeypatch):
    # Content given as a list of text parts carries no media either.
    conv = [
        {"role": "user", "content": [{"type": "text", "text": "hi"}]},
        {"role": "assistant", "content": [{"type": "text", "text": "hello"}]},
    ]
    dataset = _build_with_processor(monkeypatch, [conv])
    assert "messages" not in dataset.column_names


def test_mixed_text_and_multimodal_sources_build_client_items(monkeypatch):
    # Concatenating a text-only source with a multimodal one fills the
    # text-only rows' `messages` with None; those rows must still send ids.
    image = {"type": "image", "url": "http://x/a.png", "text": None}
    text = {"type": "text", "url": None, "text": "what is this?"}
    conv = [
        {"role": "user", "content": [image, text]},
        {"role": "assistant", "content": [{**text, "text": "a cat"}]},
    ]
    dataset = concatenate_datasets(
        [
            _build_with_processor(monkeypatch, [_conv(2)]),
            _build_with_processor(monkeypatch, [conv]),
        ]
    )
    dataset.set_format(type="torch")
    records = [
        DatasetRecord.from_arrow_row(dataset[i], storage_index=i) for i in range(2)
    ]
    text_item, mm_item = (build_client_item(record) for record in records)
    assert "messages" not in text_item
    assert mm_item["messages"] == records[1].messages


# --------------------------------------------------------------------------- #
# endpoint-level failures abort the build instead of dropping conversations    #
# --------------------------------------------------------------------------- #
def test_render_conversation_transport_error_is_an_endpoint_error(monkeypatch):
    # A dead server raises a transport error, which says nothing about the
    # conversation being rendered.
    def post(*a, **k):
        raise render_client.httpx.ConnectError("Connection refused")

    monkeypatch.setattr(render_client, "_post", post)
    monkeypatch.setattr(time, "sleep", lambda _: None)
    with pytest.raises(render_client.RenderEndpointError):
        render_client.render_conversation(
            "http://x", [], add_generation_prompt=False, max_retries=1
        )


def test_render_conversation_5xx_is_an_endpoint_error(monkeypatch):
    monkeypatch.setattr(
        render_client, "_post", lambda *a, **k: _Resp(503, {}, "unavailable")
    )
    monkeypatch.setattr(time, "sleep", lambda _: None)
    with pytest.raises(render_client.RenderEndpointError):
        render_client.render_conversation(
            "http://x", [], add_generation_prompt=False, max_retries=0
        )


def test_render_conversation_rows_reraises_an_endpoint_failure(monkeypatch):
    # The regression: the broad `except Exception` here used to turn an endpoint
    # failure into a dropped conversation, so the build finished with a dataset
    # silently missing everything after the first failure.
    def fake_render(*a, **k):
        raise render_client.RenderEndpointError("endpoint down")

    monkeypatch.setattr(preprocessing, "render_conversation", fake_render)
    with pytest.raises(render_client.RenderEndpointError):
        preprocessing._render_conversation_rows(_conv(2), None, 0, "http://x", 100)


def test_render_conversation_rows_still_skips_conversation_failures(monkeypatch):
    # A conversation the chat template cannot handle is still skipped: only
    # endpoint-level failures are fatal.
    def fake_render(*a, **k):
        raise preprocessing.BoundaryUnstableError("template rewrites history")

    monkeypatch.setattr(preprocessing, "render_conversation", fake_render)
    assert (
        preprocessing._render_conversation_rows(_conv(2), None, 0, "http://x", 100)
        == []
    )


def test_build_aborts_when_the_render_endpoint_fails(monkeypatch):
    # End to end: the run must surface the failure rather than return a partial
    # training set with no error.
    def fake_render(*a, **k):
        raise render_client.RenderEndpointError("connection refused")

    monkeypatch.setattr(preprocessing, "render_conversation", fake_render)
    data = {"conversations": [_conv(3), _conv(3)]}
    with pytest.raises(render_client.RenderEndpointError):
        build_speculator_training_dataset(
            HFDataset.from_dict(data),
            NO_PROCESSOR,
            num_proc=1,
            render_endpoint="http://x",
        )
