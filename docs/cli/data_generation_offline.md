# data_generation_offline2.py

Generates training data for speculator models by extracting hidden states from a running vLLM server. Connects to a vLLM endpoint via the OpenAI-compatible API and saves output as individual `.safetensors` files for offline training. 

## Features

- **Automatic resumption** — Detects existing `.safetensors` files in the output directory and skips already-completed samples, so interrupted runs can be resumed without reprocessing.
- **Error handling with auto-retries** — Failed requests are automatically retried up to `--max-retries` times. Samples that still fail are skipped by default, allowing the rest of the dataset to complete.
- **Consecutive failure detection** — Aborts early after `--max-consecutive-errors` consecutive failures to avoid silently churning through the dataset when the server is unreachable.
- **Async concurrency** — Sends multiple requests to the vLLM server in parallel, controlled by `--concurrency`, for high throughput.
- **Output validation** — Optional `--validate-outputs` flag verifies that saved hidden states match expected token IDs and sequence lengths.

## Basic Usage

```bash
python scripts/data_generation_offline2.py \
  --preprocessed-data ./preprocessed_dataset \
  --output ./training_data \
  --max-samples 5000
```

## Arguments

### Model Arguments

- **`--endpoint`** (str, default: `http://localhost:8000/v1`)
  The address of the vLLM instance to use for hidden states generation. The vLLM instance must be configured for hidden states extraction (see [launch_vllm.py](launch_vllm.md)).

- **`--model`** (str, default: `None`)
  HuggingFace model ID or local path for the target model. Used for verification only - the model is auto-detected from the vLLM endpoint.

### Data Arguments

- **`--preprocessed-data`** (str, required)
  Path to preprocessed dataset (produced by [prepare_data.py](prepare_data.md)).

- **`--max-samples`** (int, default: `None`)
  Maximum number of samples to process. If `None`, processes all samples.

### Output Arguments

- **`--output`** (str, default: `None`)
  Directory to save generated `.safetensors` files. Defaults to `<preprocessed-data>/hidden_states`.

### Hidden States Generation Arguments

- **`--concurrency`** (int, default: `32`)
  Number of active vLLM requests at a time. The number of async workers is set to `2 * concurrency`.

- **`--validate-outputs`** (flag)
  Load generated safetensor files and verify that output token IDs match prompt tokens and hidden states sequence length matches the number of tokens.

- **`--request-timeout`** (float)
  Timeout in seconds for each individual vLLM request.

- **`--max-retries`** (int)
  Maximum number of retry attempts per request on failure.

- **`--fail-on-error`** (flag)
  Abort when a request fails after all retries. By default, failed samples are skipped.

- **`--max-consecutive-errors`** (int, default: value of `--concurrency`)
  Abort after this many consecutive sample failures (each sample already retried `--max-retries` times). Prevents silently churning through the entire dataset when the server is down. Ignored when `--fail-on-error` is set.

## Full Example

```bash
python scripts/data_generation_offline2.py \
  --endpoint http://localhost:8000/v1 \
  --preprocessed-data ./preprocessed_dataset \
  --output ./hidden_states \
  --concurrency 64 \
  --validate-outputs
```
