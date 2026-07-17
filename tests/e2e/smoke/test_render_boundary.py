"""E2E test: derive loss masks from a live vLLM ``/render`` endpoint.

The only test that exercises the render transport for real -- the URL, the
``token_ids`` response key, and the boundary derived from what vLLM actually
returns. Everything else stubs ``_encode_render``, which cannot catch a change
in the endpoint contract or a chat template that stops being prefix-stable.

Qwen3-0.6B is the model under test because its template rewrites history: it
injects a ``<think>`` scaffold into the current assistant turn and strips it
from past ones. That breaks the append-only chain, so every multi-turn
conversation fans out to one row per assistant turn -- the production default
for any thinking model.
"""

import pytest
from datasets import Dataset as HFDataset
from transformers import AutoTokenizer

from speculators.data_generation.preprocessing import build_eagle3_dataset
from tests.conftest import requires_cuda
from tests.e2e.utils import launch_vllm_server_context

MODEL = "Qwen/Qwen3-0.6B"
PORT = 8106

# One conversation covering three behaviours at once: a leading assistant turn
# (index 0 has no context to bound against, so it yields no row), two bounded
# assistant turns (one row each), and a trailing user turn (dropped).
CONVERSATION = [
    {"role": "assistant", "content": "Here is the first batch of data."},
    {"role": "user", "content": "What is 2+2?"},
    {"role": "assistant", "content": "It is 4."},
    {"role": "user", "content": "And 3+3?"},
    {"role": "assistant", "content": "It is 6."},
    {"role": "user", "content": "thanks"},
]


@pytest.mark.e2e
@pytest.mark.smoke
@requires_cuda
def test_render_boundary_masks_against_live_vllm(tmp_path):
    tokenizer = AutoTokenizer.from_pretrained(MODEL)

    with launch_vllm_server_context(
        MODEL,
        PORT,
        str(tmp_path / "hidden_states"),
        max_model_len=1024,
        gpu_memory_utilization=0.25,
    ):
        dataset = build_eagle3_dataset(
            HFDataset.from_dict({"conversations": [CONVERSATION]}),
            tokenizer,
            max_length=1024,
            num_proc=1,
            render_endpoint=f"http://localhost:{PORT}",
        )

    # Fan-out: the two bounded assistant turns each get a row. The leading
    # assistant turn and the trailing user turn contribute neither.
    assert len(dataset) == 2

    supervised_per_row, context_per_row = [], []
    for row in dataset:
        ids = row["input_ids"].tolist()
        mask = row["loss_mask"].tolist()

        supervised = tokenizer.decode([t for t, m in zip(ids, mask, strict=True) if m])
        context = tokenizer.decode([t for t, m in zip(ids, mask, strict=True) if not m])
        supervised_per_row.append(supervised)
        context_per_row.append(context)

        # The invariant the mask exists to hold: supervise the assistant, never
        # the prompt. A leaked user turn shows up as its role header.
        assert "<|im_start|>user" not in supervised
        assert "<|im_start|>assistant" in context

        # The turn terminator is supervised -- the model must learn to stop.
        assert "<|im_end|>" in supervised

    # Each row supervises its own turn and not the other's.
    assert "It is 4." in supervised_per_row[0]
    assert "It is 6." not in supervised_per_row[0]
    assert "It is 6." in supervised_per_row[1]

    # Row 1 carries turn 1 as context, re-rendered the way inference sees it:
    # Qwen3 strips the <think> scaffold from history, which is exactly why the
    # append-only chain breaks and this conversation fans out.
    assert "It is 4." in context_per_row[1]
    assert "<think>" not in context_per_row[1]
