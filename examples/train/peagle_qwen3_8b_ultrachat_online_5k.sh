#!/bin/bash
# Online P-EAGLE Training Script
#
# Runs the full online P-EAGLE training pipeline: data preparation, vLLM server launch,
# and training (with hidden states generated on-the-fly from the live server).
#
# Usage: Copy this script, modify the configuration variables below, then run:
#   bash examples/train/peagle_qwen3_8b_ultrachat_online_5k.sh

### Example E2E run for P-EAGLE Qwen3-8B on 5k regenerated UltraChat-200k samples ###

# P-EAGLE (Parallel EAGLE) extends EAGLE-3 with parallel multi-token prediction using
# Conditional-On-Distribution (COD) sampling for memory-efficient training.

# Note: With just 5k samples, the model performance will not be very good, however there
# are enough samples to verify that the pipeline is working correctly and that the model
# is learning something. This is a good sanity check when creating a drafter for a new
# target model.

# MT-Bench Results (question.jsonl, 80 prompts, 4096 max output tokens):
# acceptance rate: 22.87%
# acceptance length: 1.91
# per-position acceptance:
#   position 0: 64.08%
#   position 1: 20.51%
#   position 2: 5.53%
#   position 3: 1.34%

set -euo pipefail

# ============ Configuration ============
MODEL="Qwen/Qwen3-8B"
DATASET="hf:inference-optimization/speculators-ci-datasets:tutorial_regen"  # on-policy regenerated Qwen3-8B data (pretokenized); or a preset/path to custom data
OUTPUT_DIR="./output/peagle_qwen3_8b_ultrachat_200k_regen"
VLLM_PORT=8108
MAX_SAMPLES=5000
SEQ_LENGTH=4096
EPOCHS=5
LR=6e-4

# P-EAGLE-specific parameters
SPECULATOR_TYPE="peagle"
NUM_LAYERS=4
NUM_DEPTHS=4
DOWN_SAMPLE_RATIO=0.7
DOWN_SAMPLE_RATIO_MIN=0.2
# GPU assignments (online training needs separate GPUs for vLLM and training)
VLLM_GPUS="2,3"
TRAIN_GPUS="4,5"
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
    --hidden-states-path "$OUTPUT_DIR/hidden_states" \
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
    --hidden-states-path "$OUTPUT_DIR/hidden_states" \
    --save-path "$OUTPUT_DIR/checkpoints" \
    --epochs "$EPOCHS" \
    --lr "$LR" \
    --total-seq-len "$SEQ_LENGTH" \
    --speculator-type "$SPECULATOR_TYPE" \
    --num-layers "$NUM_LAYERS" \
    --num-depths "$NUM_DEPTHS" \
    --down-sample-ratio "$DOWN_SAMPLE_RATIO" \
    --down-sample-ratio-min "$DOWN_SAMPLE_RATIO_MIN" \
    --no-norm-before-residual \
    --scheduler-type cosine \
    --on-missing generate \
    --on-generate delete

echo "Done. Checkpoints saved to $OUTPUT_DIR/checkpoints/"
