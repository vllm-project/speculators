# prepare_data.py

Prepares data for speculator training by:

1. Applying chat template and tokenizing each sample
2. Producing a loss/assistant mask for each sample
3. Recording token frequency statistics

The output is a processed dataset ready for online training or offline hidden states generation.

## Basic Usage

```bash
python scripts/prepare_data.py \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --data sharegpt \
  --output ./training_data \
  --max-samples 5000
```

## Arguments

### Model Arguments

- **`--model`** (str, required) HuggingFace model ID or local path for the target model.

  Example: `meta-llama/Llama-3.1-8B-Instruct`

### Data Arguments

- **`--data`** (str, required, repeatable) Path to training data. Can be a HuggingFace dataset name or local path. Use multiple times to specify multiple datasets.

  Example: `--data sharegpt --data ./custom_data.jsonl`

- **`--seq-length`** (int, default: `8192`) Maximum sequence length for each sample. Longer samples will be truncated.

- **`--max-samples`** (int, default: `None`) Maximum number of samples to process. If `None`, processes all samples.

- **`--token-freq-path`** (str, default: `{output}/token_freq.pt`) Path to save token frequency distribution. Defaults to `token_freq.pt` in the output directory.

- **`--assistant-pattern`** (str, default: `None`) Custom regex pattern for matching assistant responses. If not provided, auto-detected from chat template.

- **`--turn-dropout`** (flag) Enable turn dropout: randomly keeps first N consecutive turns per conversation for data augmentation.

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
  --data sharegpt \
  --data ./custom_conversations.jsonl \
  --output ./prepared_data \
  --seq-length 4096 \
  --max-samples 10000 \
  --turn-dropout \
  --num-preprocessing-workers 16
```
