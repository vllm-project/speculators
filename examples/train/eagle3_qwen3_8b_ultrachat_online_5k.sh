#!/bin/bash
# Online Eagle3 Training Script
#
# Runs the full online training pipeline: data preparation, vLLM server launch,
# and training (with hidden states generated on-the-fly from the live server).
#
# Usage: Copy this script, modify the configuration variables below, then run:
#   bash examples/train/eagle3_qwen3_8b_ultrachat_online_5k.sh
#
# For a detailed walkthrough, see 
# https://docs.vllm.ai/projects/speculators/en/latest/user_guide/tutorials/train/

### Example E2E run for Qwen3-8B on 5k regenerated UltraChat-200k samples ###

# Note: With just 5k samples, the model performance will not be very good, however there
# are enough samples to verify that the pipeline is working correctly and that the model
# is learning something. This is a good sanity check when creating a drafter for a new
# target model.

# MT-Bench Results (question.jsonl, 80 prompts, 4096 max output tokens):
# acceptance rate: 29.38%
# acceptance length: 1.88
# per-position acceptance:
#   position 0: 52.68%
#   position 1: 24.46%
#   position 2: 11.00%

set -euo pipefail

# ============ Configuration ============
MODEL="Qwen/Qwen3-8B"
DATASET="hf:inference-optimization/speculators-ci-datasets:tutorial_regen"  # on-policy regenerated Qwen3-8B data (pretokenized); or a preset/path to custom data
OUTPUT_DIR="./output"
VLLM_PORT=8000
DRAFT_VOCAB_SIZE=32000
MAX_SAMPLES=5000
SEQ_LENGTH=8192
EPOCHS=5
LR=1e-4

# GPU assignments (online training needs separate GPUs for vLLM and training)
VLLM_GPUS="0,1"
TRAIN_GPUS="2,3"
NUM_TRAIN_GPUS=2
# =======================================

# Step 1: Prepare data
# The dataset is on-policy regenerated: assistant turns were produced by the
# target model and ship pretokenized (input_ids + loss_mask). prepare-data just
# packages them -- no --render-endpoint (or running vLLM server) needed here.
echo "=== Step 1: Preparing data ==="
speculators prepare-data \
    --model "$MODEL" \
    --data "$DATASET" \
    --output "$OUTPUT_DIR" \
    --max-samples "$MAX_SAMPLES" \
    --seq-length "$SEQ_LENGTH"

# Step 2: Launch vLLM server in the background
echo "=== Step 2: Launching vLLM server ==="
CUDA_VISIBLE_DEVICES="$VLLM_GPUS" python scripts/launch_vllm.py "$MODEL" \
    -- --data-parallel-size 2 --port "$VLLM_PORT" --gpu-memory-utilization 0.85 &
VLLM_PID=$!

# Ensure vLLM is cleaned up on exit
cleanup() {
    echo "Stopping vLLM server..."
    kill "$VLLM_PID" 2>/dev/null || true
    wait "$VLLM_PID" 2>/dev/null || true
}
trap cleanup EXIT

echo "Waiting for vLLM server to be ready..."
until curl -sf "http://localhost:${VLLM_PORT}/health" > /dev/null 2>&1; do
    sleep 2
done
echo "vLLM server ready."

# Step 3: Train against the live vLLM server
echo "=== Step 3: Training ==="
CUDA_VISIBLE_DEVICES="$TRAIN_GPUS" torchrun \
    --standalone --nproc_per_node "$NUM_TRAIN_GPUS" \
    -m speculators.train \
    --verifier-name-or-path "$MODEL" \
    --data-path "$OUTPUT_DIR" \
    --vllm-endpoint "http://localhost:${VLLM_PORT}/v1" \
    --save-path "$OUTPUT_DIR/checkpoints" \
    --draft-vocab-size "$DRAFT_VOCAB_SIZE" \
    --epochs "$EPOCHS" \
    --lr "$LR" \
    --total-seq-len "$SEQ_LENGTH" \
    --on-missing generate \
    --on-generate delete

echo "Done. Checkpoints saved to $OUTPUT_DIR/checkpoints/"
