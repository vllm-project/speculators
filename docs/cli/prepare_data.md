# prepare-data

Converts on-policy target-model data into the format consumed by speculator training. It accepts either:

1. Natural-language conversations whose assistant responses were produced by the target model.
2. Speculator-format rows that already contain `input_ids` and `loss_mask`.

For natural-language conversations, `prepare_data.py` asks the target model's vLLM `/render` endpoint to apply the serving chat template, tokenize each assistant turn, and derive its loss mask. Rendering only converts the data's representation: it does not generate responses or turn an arbitrary dataset into on-policy data.

Preparation loads no local model or tokenizer. Prepared token rows need no server; natural-language conversations need only the target model’s render endpoint. The `--model` and `--trust-remote-code` preparation options have been removed.

The output is ready for online training or offline hidden-state generation.

## Basic Usage

Given a natural-language JSONL file such as:

```json
{"conversations":[{"role":"user","content":"Hello"},{"role":"assistant","content":"Hello! How can I help?"}]}
```

where the assistant response came from the target model:

```bash
speculators prepare-data \
  --data ./on_policy_conversations.jsonl \
  --render-endpoint http://localhost:8000 \
  --output ./training_data \
  --max-samples 5000
```

`--render-endpoint` is not needed when every input row already contains `input_ids` and `loss_mask`.

## Prepared Records

Rendered conversations, regenerated responses, and prepared input all use the same training fields:

- `input_ids`: token IDs from the target model's generation or render endpoint.
- `loss_mask`: one value per token, `0` for context and `1` for supervision.
- `messages` (optional): messages in the serving API's format, retained when needed to carry media into hidden-state extraction.

Preparation validates equal ID/mask lengths and binary mask values before truncating either field. Invalid values are rejected even if they occur beyond `--seq-length`. It then truncates IDs and masks together, drops rows with no remaining supervision or fewer than `--minimum-valid-tokens`, and saves the retained rows as tensors in an Arrow dataset. Optional messages stay attached to their corresponding rows through filtering.

For example, IDs `[10, 11, 20, 21]` with mask `[0, 0, 1, 1]` become `[10, 11, 20]` and `[0, 0, 1]` at `--seq-length 3`. At length 2, the row is dropped because no supervised tokens remain. A mask containing `2` is invalid at any sequence length.

Regeneration's readable `conversations`, tool definitions, sample identifiers, and metadata are excluded from these training fields. Text-only hidden-state extraction uses the saved token IDs; rows containing media also send their retained messages.

## Arguments

### Data Arguments

- **`--data`** (str, required, repeatable) On-policy target-model data. Use a local JSON/JSONL file or directory, or an `hf:` dataset spec. Use multiple times to combine datasets.

  Example: `--data ./target_responses.jsonl --data hf:my-org/more-target-responses`

  Natural-language input uses a `conversations` column and requires `--render-endpoint`. Assistant responses must already have been produced by the target model. Tool-calling datasets may also include a separate `tools` column. Speculator-format input uses `input_ids` and `loss_mask`.

- **`--seq-length`** (int, default: `8192`) Maximum sequence length for each sample. Longer samples will be truncated.

- **`--max-samples`** (int, default: `None`) Maximum number of samples to process. If `None`, processes all samples.

- **`--token-freq-path`** (str, default: `{output}/token_freq.pt`) Path to save token frequency distribution. Defaults to `token_freq.pt` in the output directory.

- **`--render-endpoint`** (str, default: `None`) Base URL of the target model's running vLLM server (e.g. `http://localhost:8000`). The instance launched for hidden-state extraction ([launch_vllm.py](launch_vllm.md)) serves this too, so no second server is needed. Pass the base URL only: `/v1/chat/completions/render` is appended to it, so the `/v1`-suffixed form that [data_generation_offline.py](data_generation_offline.md) `--endpoint` takes will 404. Required for natural-language conversations; omit it when every input already contains `input_ids` and `loss_mask`.

- **`--minimum-valid-tokens`** (int, default: `None`) Drop samples whose loss mask contains fewer than this many trainable tokens.

### Output Arguments

- **`--output`** (str, default: `./output`) Directory to save the processed dataset.

- **`--overwrite`** (flag) Forcibly rerun preprocessing and overwrite existing content in output directory.

- **`--allow-empty-output`** (flag) Allow writing an empty preprocessed dataset. By default raises when normalization or filtering removes every sample.

### Processing Arguments

- **`--seed`** (int, default: `0`) Random seed for reproducibility. Must match the seed used in other scripts.

- **`--num-preprocessing-workers`** (int, default: a shared render budget using 75% of available CPUs, at most `128`) Number of CPU processes for dataset preprocessing. Each worker blocks on one render call at a time, so for natural-language input this is also the render concurrency. The default assumes roughly four CPUs per preprocessing worker, including the vLLM front end and native runtime threads.

  [launch_vllm.py](launch_vllm.md) derives a matching front end from the same affinity-aware CPU count. On the standard 384-CPU H100 node, the defaults resolve to `72` workers and `18` API servers with `2` renderer threads each, leaving headroom for native runtime threads and other application work. Smaller hosts scale down automatically.

## Full Example

```bash
speculators prepare-data \
  --data ./target_responses_part1.jsonl \
  --data ./target_responses_part2.jsonl \
  --render-endpoint http://localhost:8000 \
  --output ./prepared_data \
  --seq-length 4096 \
  --max-samples 10000 \
  --num-preprocessing-workers 16
```
