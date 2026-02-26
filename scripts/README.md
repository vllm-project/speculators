# Scripts

## Eagle3 Model Production

Speculators currently supports training of Eagle3 models. This functionality is available via the scripts in this directory.

1. [train.py](/scripts/train.py): Trains an Eagle3 model using the downloaded hidden states training data.

## Table of Contents

- **[Training](#training)**<br>
  <!-- duplicate subsection name, requires -1 suffix to avoid conflict -->
  - **[Quick Start](#quick-start-2)**<br>
  - **[Arguments](#arguments)**<br>
  - **[Example Command](#example-command)**<br>

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
