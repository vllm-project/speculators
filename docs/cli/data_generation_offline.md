# data_generation_offline.py

Generates training data for speculator models by extracting hidden states from the target model using vLLM. The output is saved as individual `.pt` files for offline training.

## Basic Usage

```bash
python scripts/data_generation_offline.py \
  --target-model-path meta-llama/Llama-3.1-8B-Instruct \
  --train-data-path sharegpt \
  --output-dir ./training_data \
  --max-samples 5000
```

## Arguments

### Model Arguments

- **`--target-model-path`** (str, required)
  HuggingFace model ID or local path for the target model.

- **`--tensor-parallel-size`** (int, default: GPU count)
  Tensor parallel size for the target model. Defaults to the number of available GPUs.

- **`--gpu-memory-utilization`** (float, default: `0.8`)
  Target GPU memory utilization (0.0 to 1.0).

### Data Arguments

- **`--train-data-path`** (str, required, repeatable)
  Path to training data (same as used in preprocessing). Can be specified multiple times.

- **`--seq-length`** (int, default: `2048`)
  Maximum sequence length for preprocessing and model.

- **`--max-samples`** (int, default: `None`)
  Maximum number of samples to process. If `None`, processes all samples.

- **`--token-freq-path`** (str, default: `./token_freq.pt`)
  Path to save token frequency distribution.

- **`--hf-cache-dir`** (str, default: `None`)
  Directory for HuggingFace datasets cache. If not specified, uses `HF_DATASETS_CACHE` environment variable or default location.

- **`--assistant-pattern`** (str, default: `None`)
  Custom regex pattern for matching assistant responses. If not provided, auto-detected from chat template.

- **`--turn-dropout`** (flag)
  Enable turn dropout for data augmentation.

- **`--minimum-valid-tokens`** (int, default: `None`)
  Drop samples whose loss mask contains fewer than this many trainable tokens.

### Output Arguments

- **`--output-dir`** (str, required)
  Directory to save `.pt` files containing hidden states.

### Hidden States Generation Arguments

- **`--layer-ids`** (int list, default: auto-select)
  List of layer IDs from which to capture hidden states.
  Default: `[2, num_layers//2, num_layers-3, num_layers]`

  Example: `--layer-ids 2 16 30 32`

- **`--batch-size`** (int, default: `8`)
  Batch size for hidden states generation.

### Processing Arguments

- **`--seed`** (int, default: `0`)
  Random seed for reproducibility.

- **`--start-idx`** (int, default: `0`)
  Starting index for output files. Useful for resuming interrupted runs.

- **`--num-preprocessing-workers`** (int, default: `8`)
  Number of CPU processes for dataset preprocessing.

## Example

```bash
python scripts/data_generation_offline.py \
  --target-model-path meta-llama/Llama-3.1-70B-Instruct \
  --train-data-path sharegpt \
  --output-dir ./hidden_states \
  --tensor-parallel-size 4 \
  --batch-size 16 \
  --layer-ids 2 20 40 80 \
  --max-samples 50000 \
  --seq-length 4096
```

## See Also

- [CLI Reference Overview](README.md)
- [Previous Step: Prepare Data](prepare_data.md)
- [Next Step: Train Model](train.md)
- [Offline Hidden States Feature Guide](../user_guide/features/offline_hidden_states.md)
