#!/bin/bash
# Offline MTP Finetuning Script
#
# Runs the full offline MTP finetuning pipeline: convert native MTP head,
# data preparation, vLLM server launch, hidden states generation, training,
# and stitching finetuned weights back into the verifier checkpoint.
#
# Unlike Eagle-3, DFlash, or P-EAGLE which train draft models from scratch,
# MTP finetuning starts from the model's native MTP head (converted to
# speculators format) and finetunes it on domain-specific data.
#
# Usage: Copy this script, modify the configuration variables below, then run:
#   bash examples/train/mtp_qwen3next_gsm8k_offline.sh
#
# For a detailed walkthrough, see
# https://docs.vllm.ai/projects/speculators/en/latest/user_guide/tutorials/train_mtp_offline/

### Example E2E run for Qwen3-Next-80B-A3B-Instruct on GSM8K ###

set -euo pipefail

# ============ Configuration ============
MODEL="Qwen/Qwen3-Next-80B-A3B-Instruct"
DATASET="gsm8k"                   # sharegpt, ultrachat, gsm8k, or path to custom data
OUTPUT_DIR="./output"
CONVERTED_DIR="$OUTPUT_DIR/converted_mtp"
HIDDEN_STATES_DIR="$OUTPUT_DIR/hidden_states"
STITCHED_DIR="$OUTPUT_DIR/stitched"
VLLM_PORT=8000
MAX_SAMPLES=5000
SEQ_LENGTH=8192
EPOCHS=3
LR=1e-4
STEP_WEIGHT_BETA=0.6              # Exponential decay factor for per-step loss weights
CONCURRENCY=32                    # Parallel requests to vLLM during data generation

# GPU assignments (offline reuses the same GPUs sequentially)
GPUS="0,1"
NUM_GPUS=2
# =======================================

# Step 1: Convert native MTP head to speculators format
echo "=== Step 1: Converting native MTP head ==="
python -c "
from speculators.convert import convert_model
convert_model(
    model='$MODEL',
    verifier='$MODEL',
    algorithm='mtp',
    output_path='$CONVERTED_DIR',
    num_speculative_steps=3,
)
"

# Step 2: Prepare data
echo "=== Step 2: Preparing data ==="
python scripts/prepare_data.py \
    --model "$MODEL" \
    --data "$DATASET" \
    --max-samples "$MAX_SAMPLES" \
    --output "$OUTPUT_DIR" \
    --seq-length "$SEQ_LENGTH"

# Step 3: Launch vLLM server in the background
echo "=== Step 3: Launching vLLM server ==="
CUDA_VISIBLE_DEVICES="$GPUS" python scripts/launch_vllm.py "$MODEL" \
    -- --data-parallel-size "$NUM_GPUS" --port "$VLLM_PORT" &
VLLM_PID=$!

echo "Waiting for vLLM server to be ready..."
until curl -sf "http://localhost:${VLLM_PORT}/health" > /dev/null 2>&1; do
    sleep 2
done
echo "vLLM server ready."

# Step 4: Generate hidden states
echo "=== Step 4: Generating hidden states ==="
python scripts/data_generation_offline.py \
    --preprocessed-data "$OUTPUT_DIR" \
    --endpoint "http://localhost:${VLLM_PORT}/v1" \
    --output "$HIDDEN_STATES_DIR" \
    --max-samples "$MAX_SAMPLES" \
    --concurrency "$CONCURRENCY" \
    --validate-outputs

# Step 5: Stop vLLM server to free GPU memory for training
echo "=== Step 5: Stopping vLLM server ==="
kill "$VLLM_PID" 2>/dev/null || true
wait "$VLLM_PID" 2>/dev/null || true
echo "vLLM server stopped. GPUs freed for training."

# Step 6: Train using pre-generated hidden states
echo "=== Step 6: Training ==="
CUDA_VISIBLE_DEVICES="$GPUS" torchrun \
    --standalone --nproc_per_node "$NUM_GPUS" \
    scripts/train.py \
    --verifier-name-or-path "$MODEL" \
    --data-path "$OUTPUT_DIR" \
    --hidden-states-path "$HIDDEN_STATES_DIR" \
    --save-path "$OUTPUT_DIR/checkpoints" \
    --speculator-type mtp \
    --from-pretrained "$CONVERTED_DIR" \
    --step-weight-beta "$STEP_WEIGHT_BETA" \
    --epochs "$EPOCHS" \
    --lr "$LR" \
    --total-seq-len "$SEQ_LENGTH" \
    --on-missing raise

# Step 7: Stitch finetuned weights back into the verifier
echo "=== Step 7: Stitching finetuned weights ==="
python scripts/stitch_mtp.py \
    "$OUTPUT_DIR/checkpoints/checkpoint_best" \
    "$MODEL" \
    --output-path "$STITCHED_DIR"

echo "Done. Stitched checkpoint saved to $STITCHED_DIR/"
echo "Deploy with: vllm serve $STITCHED_DIR"
