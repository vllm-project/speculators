#!/bin/bash
set -e

# Install dependencies
pip install -q trackio

# Configuration
VERIFIER_NAME_OR_PATH="meta-llama/Llama-3.1-8B-Instruct"
HF_DATASET="nm-testing/sharegpt_llama3_8b_hidden_states"
OUTPUT_PATH="./output/llama3_8b_sharegpt"
DATA_PATH="./output/data/llama3_8b"

# Download dataset
mkdir -p "$DATA_PATH"
python3 -c "from huggingface_hub import snapshot_download; snapshot_download('$HF_DATASET', repo_type='dataset', local_dir='$DATA_PATH')"

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
COMMON_ARGS="--verifier-name-or-path $VERIFIER_NAME_OR_PATH --data-path $DATA_PATH --save-path $OUTPUT_PATH --epochs 5 --lr 3e-5 --total-seq-len 8192 --logger trackio --run-name llama3_8b_sharegpt"

if [ "$NUM_GPUS" -gt 1 ]; then
    torchrun --nnodes=1 --nproc_per_node=$NUM_GPUS "$TRAIN_SCRIPT" $COMMON_ARGS
else
    python3 "$TRAIN_SCRIPT" $COMMON_ARGS
fi
