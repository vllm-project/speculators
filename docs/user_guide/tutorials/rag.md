# Prepare RAG Data for Speculator Training

Speculative decoding can be used with retrieval-augmented generation (RAG). Retrieval builds the prompt; the draft and target models can then accelerate answer generation. It does not speed up retrieval itself, and a longer prompt still costs time and memory to process. A compatible pretrained draft can be used without RAG-specific training; benchmark it on your workload first.

When training a draft for a RAG workload, include the retrieved passages in the context **as the target model sees them at serving time**. Keep the question, passage order, source labels, system instructions, and relevant conversation history. The assistant response must come from the target model conditioned on that same context. Training on the question alone omits information the target used to produce the answer.

This walkthrough prepares one text-only RAG conversation and checks its training masks. It assumes the [training prerequisites](train.md#step-0-setup-your-environment) are installed and uses `meta-llama/Llama-3.1-8B-Instruct`, which requires access to the model on Hugging Face. One sample is a format check, not enough data to train a useful draft.

## 1. Record a target-model response

If you already have RAG conversations with responses from your target model, export them with a `conversations` column and continue to Step 2. Preserve the actual messages sent to the model rather than reconstructing a different prompt from the source documents.

Otherwise, start the target in the vLLM environment, using the vLLM version from the [training tutorial](train.md):

```bash
vllm serve meta-llama/Llama-3.1-8B-Instruct \
  --max-model-len 8192 \
  --port 8000
```

In the Speculators environment, run this example after the server is ready. The passages are illustrative retrieved results; replace them and the question with your application's prompts when building a dataset.

```python
import json
from pathlib import Path

from openai import OpenAI

messages = [
    {
        "role": "system",
        "content": "Answer using the supplied sources and cite their labels.",
    },
    {
        "role": "user",
        "content": (
            "Sources:\n"
            "[1] Standard deliveries arrive within five business days.\n"
            "[2] Express deliveries arrive within two business days.\n\n"
            "Question: How long does express delivery take?"
        ),
    },
]
client = OpenAI(base_url="http://localhost:8000/v1", api_key="EMPTY")
response = client.chat.completions.create(
    model="meta-llama/Llama-3.1-8B-Instruct",
    messages=messages,
    temperature=0,
    max_tokens=512,
)
choice = response.choices[0]
if choice.finish_reason != "stop" or not choice.message.content:
    raise RuntimeError("Expected a complete text answer; inspect the response.")
row = {
    "conversations": messages
    + [{"role": "assistant", "content": choice.message.content}]
}
Path("rag_on_policy.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")
```

The file contains the retrieved context, question, and generated answer together. Keep retrieved text in the same message roles your application uses; do not turn passages into assistant answers to make them training targets. This example uses a text-only answer. For reasoning or tool-calling models, preserve those response fields too; see [Response Regeneration](response_regeneration.md).

## 2. Prepare the conversation

Leave the target server running and use its rendering endpoint to apply the same chat template:

```bash
speculators prepare-data \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --data ./rag_on_policy.jsonl \
  --render-endpoint http://localhost:8000 \
  --output ./rag_prepared \
  --seq-length 8192 \
  --num-preprocessing-workers 1
```

Use the base URL for `--render-endpoint`, without `/v1`. Rendering tokenizes the existing conversation; it does not generate an answer. If your application overrides the chat template, configure the rendering server to use that same template.

Preprocessing creates one row per assistant turn. For the single-turn example, the system instruction, retrieved passages, and question are context with `loss_mask=0`; the assistant continuation has `loss_mask=1`. Masking passages out of the loss does **not** remove them from the model's context or eliminate their memory cost.

## 3. Budget and inspect the tokens

The sequence budget covers the whole rendered conversation, including chat-template tokens:

```text
system + history + retrieved passages + question + answer <= sequence budget
```

Measure tokens with the target's tokenizer and serving template, not character counts. Reserve room for the answer before choosing the number and size of retrieved passages. For example, an 8192-token window with a 1024-token answer budget leaves at most 7168 tokens for the rendered prompt. These are illustrative limits, not a recommended retrieval size.

The limits apply at different stages:

| Setting                     | What it bounds                                                      |
| --------------------------- | ------------------------------------------------------------------- |
| vLLM `--max-model-len`      | The server's context window, including prompt and generated tokens. |
| `prepare-data --seq-length` | Each prepared training row, including context and answer.           |
| `train --total-seq-len`     | The total sequence length of a packed training batch.               |

Keep each prepared row within the extraction server's context window and the training batch limit. Raising a limit requires corresponding model support and memory; it cannot recover tokens already removed during preparation.

`prepare-data` truncates from the **right**. An oversized context can leave no answer tokens, in which case that turn is skipped. A partially retained answer can still become a training row. Select or shorten passages **before generating the target answer**, then regenerate if you change the prompt. Do not rely on preprocessing truncation to choose relevant passages.

Inspect the saved dataset in the Speculators environment:

```python
from datasets import load_from_disk

dataset = load_from_disk("rag_prepared")
assert len(dataset) > 0, "No training rows were retained"
at_limit = 0
for index, row in enumerate(dataset):
    ids, mask = row["input_ids"], row["loss_mask"]
    assert len(ids) == len(mask)
    assert all(value in (0, 1) for value in mask)
    supervised = sum(mask)
    assert supervised > 0, f"Row {index} has no supervised tokens"
    at_limit += len(ids) >= 8192
    if index < 5:
        print(f"row={index}, tokens={len(ids)}, supervised={supervised}")
print(f"retained_rows={len(dataset)}, rows_at_limit={at_limit}")
```

Rows at the limit may have been truncated; a nonzero mask alone does not prove the answer is complete. Review preprocessing warnings and compare retained rows with the expected number of assistant turns. Inspect representative decoded samples, especially the longest ones. `--minimum-valid-tokens` can filter short surviving answers, but it does not fix an insufficient context budget. See the [prepare-data reference](../../cli/prepare_data.md) for filtering options.

## 4. Train and evaluate on the RAG workload

Follow [Train a Speculator](train.md) using `--data-path ./rag_prepared` and the same target model. That tutorial configures the hidden-state extraction server and the draft architecture; the ordinary serving command above does not configure hidden-state extraction.

Build training data from representative retrieved contexts and target responses. Keep a held-out evaluation set, and avoid splitting assistant turns from the same conversation between training and evaluation.

Compare the target alone with the target plus draft on identical held-out RAG prompts, sampling settings, output limits, hardware, and concurrency. Report prompt lengths and completed request counts alongside acceptance length, time to first token, inter-token latency, and total request latency. Keep retrieval fixed for the decoding comparison; measure retrieval separately when reporting end-to-end application latency. The [evaluation guide](evaluating_performance.md) describes the available metrics and tooling.

Long prompts with short answers may leave little decoding time to save. Acceptance length alone is not evidence of an application speedup; measure latency on your workload before deciding whether RAG-specific draft training is worthwhile.
