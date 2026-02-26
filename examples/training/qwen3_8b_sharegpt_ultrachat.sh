#!/bin/bash
set -e

# Install dependencies
pip install -q trackio

# Configuration
VERIFIER_NAME_OR_PATH="Qwen/Qwen3-8B"
HF_DATASET_ULTRACHAT="nm-testing/ultrachat_qwen3_8b_hidden_states"
HF_DATASET_SHAREGPT="nm-testing/sharegpt_qwen3_8b_hidden_states"
OUTPUT_PATH="./output/qwen3_8b_sharegpt_ultrachat"
DATA_PATH="./output/data/qwen3_8b"

# Download both datasets into the same directory
mkdir -p "$DATA_PATH"
echo "Downloading UltraChat dataset..."
python3 -c "from huggingface_hub import snapshot_download; snapshot_download('$HF_DATASET_ULTRACHAT', repo_type='dataset', local_dir='$DATA_PATH')"
echo "Downloading ShareGPT dataset..."
python3 -c "from huggingface_hub import snapshot_download; snapshot_download('$HF_DATASET_SHAREGPT', repo_type='dataset', local_dir='$DATA_PATH')"

# Detect GPUs - respect CUDA_VISIBLE_DEVICES
if [ -n "$CUDA_VISIBLE_DEVICES" ]; then
    NUM_GPUS=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)
else
    NUM_GPUS=$(nvidia-smi --list-gpus 2>/dev/null | wc -l || echo "1")
fi
# Get absolute path to train.py
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRAIN_SCRIPT="$SCRIPT_DIR/../../scripts/train.py"

# Run training
COMMON_ARGS="--verifier-name-or-path $VERIFIER_NAME_OR_PATH --data-path $DATA_PATH --save-path $OUTPUT_PATH --epochs 10 --lr 3e-5 --total-seq-len 8192 --logger trackio --run-name qwen3_8b_sharegpt_ultrachat"

if [ "$NUM_GPUS" -gt 1 ]; then
    torchrun --nnodes=1 --nproc_per_node=$NUM_GPUS "$TRAIN_SCRIPT" $COMMON_ARGS
else
    python3 "$TRAIN_SCRIPT" $COMMON_ARGS
fi
