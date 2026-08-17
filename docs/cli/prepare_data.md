# prepare_data.py

Converts on-policy target-model data into the format consumed by speculator training. It accepts either:

1. Natural-language conversations whose assistant responses were produced by the target model.
2. Speculator-format rows that already contain `input_ids` and `loss_mask`.

For natural-language conversations, `prepare_data.py` asks the target model's vLLM `/render` endpoint to apply the serving chat template, tokenize each assistant turn, and derive its loss mask. Rendering only converts the data's representation: it does not generate responses or turn an arbitrary dataset into on-policy data.

The output is ready for online training or offline hidden-state generation.

## Basic Usage

Given a natural-language JSONL file such as:

```json
{"conversations":[{"role":"user","content":"Hello"},{"role":"assistant","content":"Hello! How can I help?"}]}
```

where the assistant response came from the target model:

```bash
python scripts/prepare_data.py \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --data ./on_policy_conversations.jsonl \
  --render-endpoint http://localhost:8000 \
  --output ./training_data \
  --max-samples 5000
```

`--render-endpoint` is not needed when every input row already contains `input_ids` and `loss_mask`.

## Arguments

### Model Arguments

- **`--model`** (str, required) HuggingFace model ID or local path for the target model.

  Example: `meta-llama/Llama-3.1-8B-Instruct`

- **`--trust-remote-code`** (flag) Allow executing code from HF Hub when loading the target model's processor.

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

- **`--output`** (str, required) Directory to save the processed dataset.

- **`--overwrite`** (flag) Forcibly rerun preprocessing and overwrite existing content in output directory.

### Processing Arguments

- **`--seed`** (int, default: `0`) Random seed for reproducibility. Must match the seed used in other scripts.

- **`--num-preprocessing-workers`** (int, default: `8`) Number of CPU processes for dataset preprocessing.

## Full Example

```bash
python scripts/prepare_data.py \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --data ./target_responses_part1.jsonl \
  --data ./target_responses_part2.jsonl \
  --render-endpoint http://localhost:8000 \
  --output ./prepared_data \
  --seq-length 4096 \
  --max-samples 10000 \
  --num-preprocessing-workers 16
```
