# prepare_data.py

Converts target-model data into the canonical format consumed by speculator training. It accepts two input representations:

1. On-policy natural-language conversations in a `messages` or `conversations` column.
2. Prepared rows that already contain `input_ids` and `loss_mask`, such as the output of [response regeneration](response_regeneration.md).

The command does not generate responses. Natural-language assistant turns must already come from the target model; the vLLM `/render` endpoint only applies the serving chat template and returns token IDs.

## Basic usage

Given `on_policy.jsonl`:

```json
{"messages":[{"role":"user","content":"Hello"},{"role":"assistant","content":"Hello!"}]}
```

run the target model with vLLM and render the conversations:

```bash
python scripts/prepare_data.py \
  --data ./on_policy.jsonl \
  --render-endpoint http://localhost:8000 \
  --output ./training_data
```

Prepared regeneration output needs no endpoint:

```bash
python scripts/prepare_data.py \
  --data ./magpie_Llama-3.3-70B-Instruct.jsonl \
  --output ./training_data
```

Raw source presets such as `sharegpt` and `gsm8k` are intentionally rejected. Pass them through [response regeneration](response_regeneration.md) first so their original assistant responses never enter training.

## Arguments

- **`--data`** (required, repeatable): A local JSON/JSONL file, a directory of JSON/JSONL shards, or `hf:<dataset>[:<subset>:<split>]`. Every input must be on-policy natural language or prepared `input_ids`/`loss_mask` rows.
- **`--render-endpoint`**: Base URL of the target model's vLLM server. Required for natural-language input and unused for prepared rows. The command appends `/v1/chat/completions/render`; pass `http://localhost:8000`, not a `/v1` endpoint.
- **`--seq-length`** (default: `8192`): Maximum prepared sequence length.
- **`--max-samples`**: Maximum number of rows after combining inputs.
- **`--minimum-valid-tokens`**: Drop rows with fewer supervised tokens.
- **`--token-freq-path`**: Token-frequency output path. Defaults to `{output}/token_freq.pt`.
- **`--output`** (default: `./output`): Prepared HuggingFace dataset directory.
- **`--overwrite`**: Replace an existing directory containing only artifacts created by this command.
- **`--allow-empty-output`**: Permit an empty output after filtering.
- **`--seed`** (default: `0`): Shuffle seed.
- **`--num-preprocessing-workers`** (default: `8`): Dataset map workers.

Tool-calling conversations may include a `tools` column containing an OpenAI-style list or a JSON-encoded list. Complete assistant tool calls and tool results remain part of the conversation; preparation never executes tools.
