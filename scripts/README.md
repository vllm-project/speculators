# Eagle3 Model Production

Speculators currently supports training of Eagle3 models. This functionality is available via the scripts in this directory.

1. [data_generation_offline.py](/scripts/data_generation_offline.py): Generate training data (verifier hidden states) using vLLM. Note: this script will also preprocess the data if it hasn't been already.
2. [build_vocab_mapping.py](/scripts/build_vocab_mapping.py): Uses the token frequency distribution file to build `d2t` (draft to target) and `t2d` (target to draft) vocabulary mappings.
3. [train.py](/scripts/train.py): Trains an Eagle3 model using the training data and vocabulary mappings.
4. (Optional) [gen_and_train.py](/scripts/gen_and_train.py): A convenience wrapper around the above scripts that runs the full pipeline in one command.

## Table of Contents

- **[Data Generation](#data-generation)**<br>
  - **[Quick Start](#quick-start)**<br>
  - **[Advanced Usage](#advanced-usage)**<br>
  - **[Troubleshooting](#troubleshooting)**<br>
- **[Vocab Mapping](#vocab-mapping)**<br>
  - **[Quick Start](#quick-start-1)**<br>
- **[Training](#training)**<br>
  <!-- duplicate subsection name, requires -1 suffix to avoid conflict -->
  - **[Quick Start](#quick-start-2)**<br>
  - **[Arguments](#arguments)**<br>
  - **[Example Command](#example-command)**<br>
- **[E2E Pipeline](#e2e-pipeline)**<br>
  - **[Overview](#overview)**<br>
  - **[Prerequisites](#prerequisites)**<br>
  - **[Usage](#usage)**<br>

## Data Generation

`scripts/data_generation_offline.py` provides the main entry point for generating training data for Eagle3 models. Data generation uses vLLM and requires the optional `datagen` install.

### Quick Start

Generate training data from ShareGPT using Llama 3.1 8B:

```bash
python scripts/data_generation_offline.py \
    --target-model-path meta-llama/Llama-3.1-8B-Instruct \
    --train-data-path sharegpt \
    --output-dir ./training_data \
    --max-samples 5000
```

The script automatically uses the tokenizer's built-in chat template via `apply_chat_template`. It will use vllm to generate target model hidden states for the training data, and save them to disk alongside the input_ids and loss_mask tensors, as .pt files.

For sample generated data, see: https://huggingface.co/datasets/nm-testing/sharegpt_llama3_8b_hidden_states

### Advanced Usage

With custom settings and multi-GPU:

```bash
python scripts/data_generation_offline.py \
    --target-model-path meta-llama/Llama-3.1-70B-Instruct \
    --train-data-path ./my_data.jsonl \
    --seq-length 4096 \
    --cache-dir ./cache \
    --output-dir ./training_data \
    --layer-ids 2 28 54 \
    --tensor-parallel-size 4 \
    --batch-size 16 \
    --max-samples 10000
```

### Data Config File

The script will produce a `data_config.json` file in the output directory, which contains the configuration used to generate the data, as well as other metadata about the data generation process.

Example file:

```json
{
  "version": "2.0",
  "generated_at": "2025-12-03T16:03:02.471808+00:00",
  "speculators_version": "0.3.0",
  "reproducibility": {
    "command": "data_generation_offline.py --target-model-path meta-llama/Llama-3.1-8B-Instruct --train-data-path sharegpt --output-dir ./training_data --max-samples 5000",
    "package_versions": {
      "torch": "2.8.0+cu128",
      "vllm": "0.11.0",
      "transformers": "4.57.3",
      "speculators": "0.3.0"
    },
    "gpu": "NVIDIA H100 80GB HBM3"
  },
  "model": {
    "target_model_path": "meta-llama/Llama-3.1-8B-Instruct",
    "tensor_parallel_size": 1,
    "max_model_len": 2048,
    "gpu_memory_utilization": 0.8,
    "hidden_size": 4096
  },
  "data": {
    "train_data_path": "sharegpt",
    "seq_length": 2048,
    "max_samples": 5000,
    "num_samples": 5000,
    "seed": 0,
    "chat_template_note": "Uses tokenizer's built-in chat template"
  },
  "hidden_states": {
    "layer_ids": [
      2,
      16,
      29,
      31
    ],
    "description": "Layers selected for EAGLE3 fusion and target logits"
  },
  "generation": {
    "cache_dir": "/home/***/.cache/huggingface/datasets"
  },
  "format": {
    "file_pattern": "data_{idx}.pt",
    "data_format_version": 1,
    "schema": {
      "input_ids": {
        "dtype": "torch.long",
        "shape": "[seq_len]",
        "description": "Tokenized input sequence"
      },
      "hidden_states": {
        "dtype": "list[torch.bfloat16]",
        "shape": "list of [seq_len, 4096]",
        "num_tensors": 4,
        "description": "Hidden states from 4 layers"
      },
      "loss_mask": {
        "dtype": "torch.long",
        "shape": "[seq_len]",
        "description": "1 for assistant tokens to train on, 0 elsewhere"
      }
    }
  }
}
```

### Token Frequency File

Along with the `data_config.json`, the data generation step will also generate a `token_freq.pt` file containing the token frequencies. If not specified, the default location for the token frequency file is `./token_freq.pt` i.e in the same directory where the script runs. This frequencies will be used to `d2t` i.e `draft-to-target` and `t2d` i.e `target-to-draft` vocabulary mappings.

#### Datasets

Built-in datasets (can be used directly by name in the `--train-data-path` argument):

- `sharegpt` - ShareGPT Vicuna unfiltered
- `ultrachat` - HuggingFace UltraChat 200k

Alternatively, you can use a different dataset by passing the HuggingFace dataset path or local JSON/JSONL file path in the `--train-data-path` argument.

#### Caching

Preprocessing is automatically cached by HuggingFace datasets using fingerprint-based cache invalidation. The cache automatically updates when:

- Tokenizer changes
- Preprocessing parameters change (seq_length, etc.)
- Dataset changes

**Cache Location:**

Default: `~/.cache/huggingface/datasets` (Optional) Use a custom cache directory by setting the `HF_HUB_CACHE` environment variable

```bash
# Example: Use custom cache directory
export HF_HUB_CACHE=/path/to/your/cache
python scripts/data_generation_offline.py ...
```

### Troubleshooting

1. **Out of memory during hidden state extraction**

   - Reduce `--batch-size`
   - Reduce `--seq-length`
   - Increase `--tensor-parallel-size`

2. **Layer index out of bounds**

   - Check model's actual number of layers
   - Auto-selection uses: `[2, num_layers // 2, num_layers - 3]`

3. **No assistant response spans found**

   - Ensure tokenizer has a chat template (supports `apply_chat_template`)
   - Check that conversations have assistant responses in correct format (role/content keys)

4. **Cache invalidation**

   - Delete cache directory if changing preprocessing parameters
   - Ensure `--seed` matches between runs for reproducibility

## Vocab Mapping

`scripts/build_vocab_mapping.py` Uses the token frequency distribution file to build `d2t` (draft to target) and `t2d` (target to draft) vocabulary mappings.

### Quick Start

Generate vocab mapping using Llama 3.1 8B:

by specifying `target-vocab-size` manually:

```bash
    python scripts/build_vocab_mapping.py \
        --token-freq-path ./token_freq.pt \
        --draft-vocab-size 32000 \
        --target-vocab-size 128256 \
        --output-path ./vocab_mapping
```

or by using `target-model-path` to automatically infer the target vocab size:

```bash
    python scripts/build_vocab_mapping.py \
        --token-freq-path ./token_freq.pt \
        --draft-vocab-size 32000 \
        --target-model-path meta-llama/Llama-3.1-8B-Instruct \
        --output-path ./vocab_mapping
```

If not specified, the default location for token frequency file is `./token_freq.pt`. Make sure `target-vocab-size` match the verifier model vocab size exactly. Once complete, this step will generate and save `t2d.npy` and `d2t.npy` files to disk.

## Training

`scripts/train.py` provides the main entry point for training Eagle3 models.

### Quick Start

To run in a single-node multi-GPU distributed training setup with FSDP, the scripts should be launched with `torchrun`:

```bash
torchrun --standalone --nproc_per_node=<num_gpus>  scripts/train.py
```

For single GPU training (useful for debugging), the script can be run directly:

```bash
python scripts/train.py
```

> [!NOTE]
> Use `CUDA_VISIBLE_DEVICES=<gpu_ids>` to control which GPUS are visible to the script.

### Arguments

The scripts has one required argument: `--verifier-name-or-path`, which is the name or path of the verifier model to use.

The scripts has the following optional arguments:

- `--data-path`: The path to the data directory. Defaults to `./data`. The script will collect all `.pt` files in this directory or its subdirectories and use them as training data.
- `--save-path`: The path to save the checkpoints. Defaults to `./checkpoints`. The script will create subdirectories for each epoch to save the model weights and optimizer states. e.g. `./checkpoints/0/`
- `--epochs`: The number of epochs to train for. Defaults to 20.
- `--lr`: The learning rate to use. Defaults to 1e-4.
- `--no-resume-from-checkpoint`: If set, the script will not resume from the last checkpoint if it exists, and will instead start from scratch and overwrite existing checkpoints.
- `--logger`: The logger to use. Defaults to empty string, which means no logging. Supported loggers are `trackio`, `wandb`, and `tensorboard`.
- `--total-seq-len`: The total sequence length to use. Defaults to 8192.
- `--data-format-version`: The version of the data format to use. Defaults to 1. The structure of the data to train on. `1` is the default and is the structure produced by Speculators generation scripts. `0` exists for backwards compatibility with the old data format.
- `--log-dir`: The path to save the logs. Defaults to `./logs`.
- `--run-name`: The name of the run. Defaults to None.
- `--num-layers`: The number of layers to use. Defaults to 1.
- `--d2t-path`: The path to the d2t tensor. Defaults to `d2t.npy`.
- `--t2d-path`: The path to the t2d tensor. Defaults to `t2d.npy`.
- `--ttt-steps`: The number of TTT steps to use. Defaults to 3.
- `--ttt-step-loss-decay`: The loss decay factor to use for the TTT steps. Defaults to 1.0.

### Example Command

```bash
torchrun --nnodes=1 --nproc_per_node=8 scripts/train.py \
    --verifier-name-or-path "meta-llama/Llama-3.1-8B-Instruct" \
    --data-path "./data/llama-3.1-8b_sharegpt/gen/" \
    --save-path "./checkpoints/llama-3.1-8b.eagle3" \
    --epochs 10 \
    --lr 1e-4 \
    --no-resume-from-checkpoint \
    --logger "tensorboard" \
    --total-seq-len 8192 \
    --data-format-version 1 \
    --log-dir "./logs/llama-3.1-8b.eagle3" \
    --run-name "llama-3.1-8b.eagle3" \
    --num-layers 1 \
    --d2t-path "./data/llama-3.1-8b_sharegpt/d2t.npy" \
    --t2d-path "./data/llama-3.1-8b_sharegpt/t2d.npy" \
    --ttt-steps 3 \
    --ttt-step-loss-decay 1.0
```

## E2E Pipeline

### Overview

`scripts/gen_and_train.py` can be used to run the full pipeline in one command. It also ensures each script is run with the correct arguments and dependencies.

Internally it calls the following scripts in order:

1. scripts/data_generation_offline.py
2. scripts/build_vocab_mapping.py
3. scripts/train.py

Using `uv` to produce ephemeral environments for each script.

### Prerequisites:

- python 3.10+
- uv (`pip install uv`)

### Usage:

> [!IMPORTANT]
> Update the script arguments section in the script file itself before running.

Then run:

```bash
python scripts/gen_and_train.py
```

> [!NOTE]
> You can call the script with environment variables (like `CUDA_VISIBLE_DEVICES` and `HF_HOME`) to control the behavior of the scripts. By default the script will use all available GPUs.

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python scripts/gen_and_train.py
```
