# Eagle3 Training

`scripts/train.py` provides the main entry point for training Eagle3 models. 

## Running the training script

To run in a multi-node distributed training setup with FSDP, the scripts should be launched with `torchrun`:
```bash
torchrun --nnodes=1 --nproc_per_node=<num_gpus>  scripts/train.py
```

For single GPU training (useful for debugging), the script can be run directly:
```bash
python scripts/train.py
```

## Arguments
The scripts has one required argument: `--verifier_name_or_path`, which is the name or path of the verifier model to use.

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

## Example run command
```bash
torchrun --nnodes=1 --nproc_per_node=8 scripts/train.py \
    --verifier_name_or_path "meta-llama/Llama-3.1-8B" \
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