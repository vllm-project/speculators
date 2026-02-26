# Scripts

## Eagle3 Model Production

Speculators currently supports training of Eagle3 models. This functionality is available via the scripts in this directory.

1. [build_vocab_mapping.py](/scripts/build_vocab_mapping.py): Uses the token frequency distribution file to build `d2t` (draft to target) and `t2d` (target to draft) vocabulary mappings.
2. [train.py](/scripts/train.py): Trains an Eagle3 model using the training data and vocabulary mappings.

## Table of Contents

- **[Vocab Mapping](#vocab-mapping)**<br>
  - **[Quick Start](#quick-start-1)**<br>
- **[Training](#training)**<br>
  <!-- duplicate subsection name, requires -1 suffix to avoid conflict -->
  - **[Quick Start](#quick-start-2)**<br>
  - **[Arguments](#arguments)**<br>
  - **[Example Command](#example-command)**<br>

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
