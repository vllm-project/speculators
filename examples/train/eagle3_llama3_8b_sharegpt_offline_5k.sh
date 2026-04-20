#!/bin/bash
# Offline Eagle3 Training Script
#
# Runs the full offline training pipeline: data preparation, vLLM server launch,
# hidden states generation, and training (with pre-generated hidden states).
#
# Usage: Copy this script, modify the configuration variables below, then run:
#   bash examples/train/eagle3_llama3_8b_sharegpt_offline_5k.sh
#
# For a detailed walkthrough, see examples/OFFLINE_TRAINING.md

### Example E2E run for Llama 3.1 8B on 5k samples from ShareGPT ###

# Note: With just 5k samples, the model performance will not be very good, however there
# are enough samples to verify that the pipeline is working correctly and that the model
# is learning something. This is a good sanity check when creating a drafter for a new
# target model.

# Timing (on 2x NVIDIA H100 80GB GPUs)
# Data Preprocessing: 15 seconds
# vLLM Server Startup: 29 seconds
# Training (5 epochs): 1821 seconds (30 mins 21 secs)
# Total: 1896 seconds (31 mins 36 secs)

# Results on MT-Bench (80 prompts, up to 2048 output tokens):
# acceptance rate: 16.32%
# acceptance length: 1.49
# per-position acceptance:
#   position 0: 36.29%
#   position 1: 10.31%
#   position 2: 2.35%
# output throughput: ~196 tok/s

set -euo pipefail

# ============ Configuration ============
MODEL="meta-llama/Llama-3.1-8B-Instruct"
DATASET="ultrachat"                # sharegpt, ultrachat, or path to custom data
OUTPUT_DIR="./output"
HIDDEN_STATES_DIR="$OUTPUT_DIR/hidden_states"
VLLM_PORT=8000
MAX_SAMPLES=5000
DRAFT_VOCAB_SIZE=32000
SEQ_LENGTH=8192
EPOCHS=5
LR=1e-4
CONCURRENCY=32                    # Parallel requests to vLLM during data generation

# GPU assignments (offline reuses the same GPUs sequentially)
GPUS="0,1,2,3"
NUM_GPUS=4
# =======================================

# Step 1: Prepare data
echo "=== Step 1: Preparing data ==="
python scripts/prepare_data.py \
    --model "$MODEL" \
    --data "$DATASET" \
    --max-samples "$MAX_SAMPLES" \
    --output "$OUTPUT_DIR" \
    --seq-length "$SEQ_LENGTH"

# Step 2: Launch vLLM server in the background
echo "=== Step 2: Launching vLLM server ==="
CUDA_VISIBLE_DEVICES="$GPUS" python scripts/launch_vllm.py "$MODEL" \
    -- --port "$VLLM_PORT" &
VLLM_PID=$!

echo "Waiting for vLLM server to be ready..."
until curl -sf "http://localhost:${VLLM_PORT}/health" > /dev/null 2>&1; do
    sleep 2
done
echo "vLLM server ready."

# Step 3: Generate hidden states
echo "=== Step 3: Generating hidden states ==="
python scripts/data_generation_offline2.py \
    --preprocessed-data "$OUTPUT_DIR" \le
    --endpoint "http://localhost:${VLLM_PORT}/v1" \
    --output "$HIDDEN_STATES_DIR" \
    --max-samples "$MAX_SAMPLES" \
    --concurrency "$CONCURRENCY" \
    --validate-outputs

# Step 4: Stop vLLM server to free GPU memory for training
echo "=== Step 4: Stopping vLLM server ==="
kill "$VLLM_PID" 2>/dev/null || true
wait "$VLLM_PID" 2>/dev/null || true
echo "vLLM server stopped. GPUs freed for training."

# Step 5: Train using pre-generated hidden states
echo "=== Step 5: Training ==="
CUDA_VISIBLE_DEVICES="$GPUS" torchrun \
    --standalone --nproc_per_node "$NUM_GPUS" \
    scripts/train.py \
    --verifier-name-or-path "$MODEL" \
    --data-path "$OUTPUT_DIR" \
    --hidden-states-path "$HIDDEN_STATES_DIR" \
    --save-path "$OUTPUT_DIR/checkpoints" \
    --draft-vocab-size "$DRAFT_VOCAB_SIZE" \
    --epochs "$EPOCHS" \
    --lr "$LR" \
    --total-seq-len "$SEQ_LENGTH" \
    --on-missing raise

echo "Done. Checkpoints saved to $OUTPUT_DIR/checkpoints/"
