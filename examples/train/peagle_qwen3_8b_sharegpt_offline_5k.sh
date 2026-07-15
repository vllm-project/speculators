#!/bin/bash
# Offline P-EAGLE Training Script
#
# Runs the full offline P-EAGLE training pipeline: data preparation, vLLM server
# launch, hidden states generation, and training (with pre-generated hidden states).
#
# Usage: Copy this script, modify the configuration variables below, then run:
#   bash examples/train/peagle_qwen3_8b_sharegpt_offline_5k.sh
#
# For a detailed walkthrough, see
# https://docs.vllm.ai/projects/speculators/en/latest/user_guide/tutorials/train_peagle_offline/

### Example E2E run for P-EAGLE Qwen3-8B on 5k samples from ShareGPT ###

# Note: With just 5k samples, the model performance will not be very good, however there
# are enough samples to verify that the pipeline is working correctly and that the model
# is learning something. This is a good sanity check when creating a drafter for a new
# target model.

# Timing (on 4x NVIDIA H100 80GB GPUs, DP=2)
# Data Preprocessing: 26 seconds
# vLLM Server Startup: 82 seconds (1 min 22 secs)
# Hidden States Generation: ~4 mins
# Training (5 epochs): ~47 mins
# Total: ~53 mins

# Results on SpecBench (80 prompts, 256 output tokens):
# acceptance rate: 13.35%
# acceptance length: 1.53
# per-position acceptance:
#   position 0: 40.84%
#   position 1: 10.84%
#   position 2: 1.58%
#   position 3: 0.15%

set -euo pipefail

# ============ Configuration ============
MODEL="Qwen/Qwen3-8B"
DATASET="sharegpt"                # sharegpt, ultrachat, or path to custom data
OUTPUT_DIR="./output/peagle_qwen3_8b_sharegpt"
HIDDEN_STATES_DIR="$OUTPUT_DIR/hidden_states"
VLLM_PORT=8000
MAX_SAMPLES=5000
SEQ_LENGTH=4096
EPOCHS=5
LR=6e-4
CONCURRENCY=32                    # Parallel requests to vLLM during data generation

# P-EAGLE-specific parameters
SPECULATOR_TYPE="peagle"
NUM_LAYERS=4
NUM_DEPTHS=4
DOWN_SAMPLE_RATIO=0.7
DOWN_SAMPLE_RATIO_MIN=0.2

# GPU assignments (offline reuses the same GPUs sequentially)
GPUS="0,1"
NUM_GPUS=2

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
    --hidden-states-path "$HIDDEN_STATES_DIR" \
    -- --data-parallel-size 2 --port "$VLLM_PORT" &
VLLM_PID=$!

echo "Waiting for vLLM server to be ready..."
until curl -sf "http://localhost:${VLLM_PORT}/health" > /dev/null 2>&1; do
    sleep 2
done
echo "vLLM server ready."

# Step 3: Generate hidden states
echo "=== Step 3: Generating hidden states ==="
python scripts/data_generation_offline.py \
    --preprocessed-data "$OUTPUT_DIR" \
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
    --speculator-type "$SPECULATOR_TYPE" \
    --num-layers "$NUM_LAYERS" \
    --num-depths "$NUM_DEPTHS" \
    --down-sample-ratio "$DOWN_SAMPLE_RATIO" \
    --down-sample-ratio-min "$DOWN_SAMPLE_RATIO_MIN" \
    --no-norm-before-residual \
    --scheduler-type cosine \
    --epochs "$EPOCHS" \
    --lr "$LR" \
    --total-seq-len "$SEQ_LENGTH" \
    --on-missing raise

echo "Done. Checkpoints saved to $OUTPUT_DIR/checkpoints/"
