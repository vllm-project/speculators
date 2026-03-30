#!/usr/bin/env bash
# Finetune the FastMTP speculator on pre-generated hidden state data.
#
# Usage (from repo root):
#   bash examples/fast_mtp/run_finetune.sh
#
# Prerequisites:
#   1. Run convert_checkpoint.py to produce SPECULATOR_PATH
#   2. Run generate_dataset.py to produce DATA_DIR
#
# Output: checkpoints written to OUTPUT_DIR, training log to local/logs/

set -euo pipefail

SPECULATOR_PATH="Qwen3-Next-80B-A3B-Instruct_mtp_speculator"
DATA_DIR="${DATA_DIR:-}"  # Set via env var or edit below; no hardcoded user paths
OUTPUT_DIR="output/qwen3next_gsm8k_finetuned"
NUM_GPUS=4

if [[ -z "${DATA_DIR}" ]]; then
    echo "ERROR: DATA_DIR is not set. Export it before running:" >&2
    echo "  export DATA_DIR=/path/to/dataset" >&2
    exit 1
fi

mkdir -p local/logs

CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun \
    --standalone \
    --nproc_per_node="${NUM_GPUS}" \
    examples/fast_mtp/finetune.py \
    --speculator-path "${SPECULATOR_PATH}" \
    --data-dir "${DATA_DIR}" \
    --output-dir "${OUTPUT_DIR}" \
    --max-len 4096 \
    --lr 5e-5 \
    --num-epochs 3 \
    --batch-size 16 \
    --train-ratio 0.9 \
    --scheduler-type cosine \
    --save-best \
    --logger wandb \
    --log-dir "${OUTPUT_DIR}/logs" \
    --run-name "fastmtp_gsm8k_{time}" 2>&1 | tee local/logs/gsm8k_finetune.log
