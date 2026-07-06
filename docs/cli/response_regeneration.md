# response_regeneration

Regenerates assistant responses in existing datasets using a vLLM-served model. Given a dataset containing user prompts (e.g., Magpie, UltraChat), this pipeline extracts the prompts, sends them to a vLLM server, and produces a new dataset with the original prompts paired with freshly generated responses from the target model. This is useful for creating training data where you want a specific model's outputs in place of the original assistant responses.

The pipeline consists of two scripts:

| Script       | Purpose                                                        |
| ------------ | -------------------------------------------------------------- |
| `run_all.sh` | End-to-end pipeline: starts vLLM, regenerates responses, stops |
| `script.py`  | Standalone response regeneration against a running vLLM server |

## run_all.sh

Orchestrates the entire pipeline: starts a vLLM server (with optional data/tensor parallelism), regenerates responses for the dataset, and stops the server. Uses vLLM's built-in data parallelism (`--data-parallel-size`) for multi-GPU scaling with automatic load balancing.

### Basic Usage

```bash
./scripts/response_regeneration/run_all.sh \
  --model "meta-llama/Llama-3.3-70B-Instruct" \
  --dataset magpie
```

### Arguments

- **`--model`** (str, required) Model to serve and use for generation.

- **`--gpus`** (str, default: all visible) Comma-separated GPU IDs (sets `CUDA_VISIBLE_DEVICES`).

- **`--port`** (int, default: `8000`) Server port.

- **`--dp-size`** (int) Number of data parallel replicas (maps to vLLM's `--data-parallel-size`).

- **`--tp-size`** (int) Tensor parallel size per replica (maps to vLLM's `--tensor-parallel-size`).

- **`--keep-server`** (flag) Don't stop the vLLM server after processing completes.

All other arguments are passed through to `script.py`.

### Full Example

```bash
./scripts/response_regeneration/run_all.sh \
  --model "meta-llama/Llama-3.3-70B-Instruct" \
  --dp-size 4 --tp-size 2 \
  --dataset magpie \
  --limit 1000 \
  --concurrency 128 \
  --max-tokens 4096
```

## script.py

Extracts user prompts from a dataset, sends them to a vLLM chat completion endpoint, and writes out new prompt-response pairs with the target model's generated responses.

### Features

- **Auto-detects model** from vLLM server (no need to specify `--model`)
- **Resume capability** to skip already-processed rows
- **Async processing** with configurable concurrency

### Basic Usage

```bash
python scripts/response_regeneration/script.py --dataset magpie
```

### Arguments

#### Data Arguments

- **`--dataset`** (str, default: `ultrachat`, choices: `magpie`, `ultrachat`) Dataset to process.

- **`--split`** (str, default: dataset-specific) Dataset split. Defaults to `train` for magpie and `train_sft` for ultrachat.

- **`--limit`** (int, default: `None`) Stop after N rows.

- **`--language-filter`** (str, default: `None`) Only process rows where language matches this value (e.g., `EN`).

#### Server Arguments

- **`--endpoint`** (str, default: `http://127.0.0.1:8000/v1/chat/completions`) vLLM chat completions endpoint.

- **`--model`** (str, default: `None`) Model name exposed by vLLM. Auto-detected from the server if not specified.

#### Generation Arguments

- **`--concurrency`** (int, default: `64`) Max concurrent requests to the vLLM server.

- **`--max-tokens`** (int, default: `8192`) Max tokens for generation.

- **`--sampling-params`** (str, default: `None`) JSON object merged into each chat-completion request, e.g. `'{"temperature": 0.6, "top_p": 0.95, "seed": 0}'`. Unset keys use the server defaults.

#### Output Arguments

- **`--outfile`** (str, default: auto-generated) Output JSONL path. If not specified, auto-generated as `{dataset}_{model}.jsonl`.

- **`--resume`** (flag) Skip conversations already present in the output file (matched by `primary_id`: the row's `id`/`uuid` if it has one, otherwise a content hash).

### Full Example

```bash
python scripts/response_regeneration/script.py \
  --dataset magpie \
  --endpoint http://127.0.0.1:8000/v1/chat/completions \
  --limit 1000 \
  --concurrency 128 \
  --max-tokens 4096 \
  --outfile magpie_Llama-3.3-70B-Instruct.jsonl \
  --resume
```

## Supported Datasets

| Dataset   | HuggingFace ID                                    | Prompt Field  | Default Split |
| --------- | ------------------------------------------------- | ------------- | ------------- |
| Magpie    | `Magpie-Align/Magpie-Llama-3.1-Pro-300K-Filtered` | `instruction` | `train`       |
| UltraChat | `HuggingFaceH4/ultrachat_200k`                    | `prompt`      | `train_sft`   |

## Output Format

Rows are pre-tokenized and ready for training: one row per assistant turn, holding the prompt the target conditioned on followed by the tokens it generated. The endpoint must support `return_token_ids`, which the script uses to read the generation boundary directly instead of re-tokenizing the text and recovering the boundary with a regex.

```json
{
  "id": "conv-abc_turn0",
  "primary_id": "conv-abc",
  "input_ids": [151644, 872, ...],
  "loss_mask": [0, 0, ..., 1, 1],
  "conversations": [
    {"role": "user", "content": "What is the capital of France?"},
    {"role": "assistant", "content": "The capital of France is Paris."}
  ],
  "metadata": {
    "idx": 0,
    "finish_reason": "stop",
    "usage": {...},
    "endpoint": "http://127.0.0.1:8000/v1/chat/completions",
    "sampling_params": {...}
  }
}
```

- `loss_mask` is `0` over the prompt and `1` over the generated tokens. This *is* the generation boundary, so training applies no further masking.
- A conversation with N assistant turns yields N rows, each carrying the history before it. Turn `k`'s row is `{primary_id}_turn{k}`.
- `primary_id` is the conversation's stable id, used by `--resume`. The row `id` is turn-suffixed and never matches it.
- `conversations` is a human-readable twin of `input_ids` for review only. Training drops it.

Rows are written only after every turn of a conversation succeeds. A conversation that fails partway writes nothing to the output file and one row to a sibling error file instead (`--outfile out.jsonl` gives `out.errors.jsonl`), so `--resume` retries it whole:

```json
{
  "id": "conv-abc",
  "metadata": {
    "idx": 0,
    "error": "ConnectionError(...)",
    "turns_completed": 1,
    "endpoint": "http://127.0.0.1:8000/v1/chat/completions"
  }
}
```

If `--outfile` is not specified, the filename is auto-generated based on dataset and model (e.g., `magpie_Llama-3.3-70B-Instruct.jsonl`).
