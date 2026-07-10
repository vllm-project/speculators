# data_generation_offline.py

Generates training data for speculator models by extracting hidden states from a running vLLM server. Connects to a vLLM endpoint via the OpenAI-compatible API and saves output as individual `.safetensors` files for offline training.

## Features

- **Validated resumption** — Resumes only from regular, non-symlink safetensors whose required tensors and shapes are valid and whose `token_ids` match the current dataset row.
- **Error handling with auto-retries** — Failed requests are automatically retried up to `--max-retries` times. Samples that still fail are skipped by default, allowing the rest of the dataset to complete.
- **Consecutive failure detection** — Aborts early after `--max-consecutive-errors` consecutive failures to avoid silently churning through the dataset when the server is unreachable.
- **Async concurrency** — Sends multiple requests to the vLLM server in parallel, controlled by `--concurrency`, for high throughput.
- **Output validation** — Every generated source passes a safetensors structure gate and exact-token (or validated multimodal-prefix) check before it can become a cache entry. The optional `--validate-outputs` flag additionally performs a full hidden-state value and NaN/Inf scan.

## Basic Usage

```bash
python scripts/data_generation_offline.py \
  --preprocessed-data ./preprocessed_dataset \
  --output ./training_data \
  --max-samples 5000
```

## Arguments

### Model Arguments

- **`--endpoint`** (str, default: `http://localhost:8000/v1`) The address of the vLLM instance to use for hidden states generation. The vLLM instance must be configured for hidden states extraction (see [launch_vllm.py](launch_vllm.md)).

- **`--model`** (str, default: `None`) HuggingFace model ID or local path for the target model. Used for verification only - the model is auto-detected from the vLLM endpoint.

### Data Arguments

- **`--preprocessed-data`** (str, required) Path to preprocessed dataset (produced by [prepare_data.py](prepare_data.md)).

- **`--max-samples`** (int, default: `None`) Maximum number of samples to process. If `None`, processes all samples.

### Output Arguments

- **`--output`** (str, default: `None`) Directory to save generated `.safetensors` files. Defaults to `<preprocessed-data>/hidden_states`.

- **`--source-hidden-states-root`** (str, default: resolved `--output`) Existing allowed root for source `.safetensors` paths returned by vLLM. Set this explicitly when the vLLM connector writes to a different shared-storage directory. Returned paths outside this root, relative paths, and paths containing symlink components are rejected.

### Hidden States Generation Arguments

- **`--concurrency`** (positive int, default: `32`) Number of active vLLM requests at a time. Zero and negative values are rejected. The number of async workers is set to `2 * concurrency`.

- **`--validate-outputs`** (flag) Load generated safetensor files and verify that output token IDs match prompt tokens and hidden states sequence length matches the number of tokens.

- **`--request-timeout`** (finite positive float) Timeout in seconds for each individual vLLM request. Zero, negative, NaN, and infinite values are rejected before any request is made.

- **`--max-retries`** (int) Maximum number of retry attempts per request on failure.

- **`--fail-on-error`** (flag) Abort when a request fails after all retries. By default, failed samples are skipped.

- **`--max-consecutive-errors`** (int, default: value of `--concurrency`) Abort after this many consecutive sample failures (each sample already retried `--max-retries` times). Prevents silently churning through the entire dataset when the server is down. Ignored when `--fail-on-error` is set.

### Multi-Node Arguments

- **`--world-size`** (int, default: `1`) Number of nodes participating in data generation. Each node is assigned a contiguous, non-overlapping chunk of the dataset. This is the number of nodes, not the number of GPUs.

- **`--rank`** (int, default: `0`) Zero-based index of the current node. Must be in the range `[0, world-size)`.

## Resume and commit safety

At startup, each canonical `hs_<index>.safetensors` file is checked before its
index enters the resume set. Empty or corrupt files, directories, symlinks,
missing tensors, non-floating or non-finite hidden states, invalid
`[seq_len, num_layers, hidden_size]` shapes, out-of-range indices, and token IDs
that do not match the current dataset are moved to `<output>/invalid/`. The
quarantined name
contains a reason and UTC timestamp, and a durable JSON evidence record preserves
the original path, quarantine path, timestamp, and full validation error. The
canonical name is then free for regeneration; invalid evidence is never silently
deleted.

Final cache commits use a persistent per-target cooperative file lock. A writer
may create a missing target or treat a byte-identical target as an idempotent
success. A different existing target raises an explicit conflict instead of being
silently overwritten. The only replacement exception is an explicitly validated
in-place multimodal prefix truncation.

Connector responses may not point at another managed
`hs_<index>.safetensors` sibling of the requested target, including when a broad
source root contains the output root. The current target itself is accepted only
for the explicit in-place truncation case. Use a separate
`--source-hidden-states-root` for connector staging when possible.

Source consumption is version-bound. Cross-device copies and post-validation
deletes verify that the source bytes did not change; a replacement is preserved
and the sample fails instead of deleting or committing the wrong version.

## Full Example

```bash
python scripts/data_generation_offline.py \
  --endpoint http://localhost:8000/v1 \
  --preprocessed-data ./preprocessed_dataset \
  --output ./hidden_states \
  --source-hidden-states-root /shared/vllm_hidden_states \
  --concurrency 64 \
  --validate-outputs
```
