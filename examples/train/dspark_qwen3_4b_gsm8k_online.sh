#!/bin/bash
# Online DSpark Training Script (GSM8K)
#
# Runs the full online DSpark training pipeline on GSM8K math data:
# data preparation, vLLM server launch, and training (with hidden states
# generated on-the-fly from the live server).
#
# DSpark features anchor-based training with cross-attention, Markov head,
# and optional confidence head for speculative decoding.
#
# Usage:
#   bash examples/train/dspark_qwen3_8b_gsm8k_online.sh

### Example E2E run for DSpark Qwen3-8B on GSM8K ###

# Note: GSM8K is a math reasoning dataset. Training on it helps the draft
# model specialize in math-style reasoning patterns.

set -euo pipefail

# ============ Configuration ============
MODEL="/home/data/weights/Qwen3-4B"

# Use pre-regenerated GSM8K responses from the target model.
# To generate your own, see:
# https://docs.vllm.ai/projects/speculators/en/latest/user_guide/tutorials/response_regeneration/
DATASET="inference-optimization/Qwen3-8B-responses"
DATASET_FILE="/home/regenerated_0_to_568.json"
OUTPUT_DIR="./output/dspark_qwen3_8b_gsm8k"
VLLM_PORT=8000
MAX_SAMPLES=5000
SEQ_LENGTH=4096
EPOCHS=5
LR=3e-4

# DSpark-specific parameters
SPECULATOR_TYPE="dspark"
BLOCK_SIZE=8
NUM_ANCHORS=256
NUM_LAYERS=5
DRAFT_VOCAB_SIZE=32000
TARGET_LAYER_IDS="2 18 33"  # Must match vLLM's eagle_aux_hidden_state_layer_ids

# Markov head (set markov_rank=0 to disable)
MARKOV_RANK=128
MARKOV_HEAD_TYPE="vanilla"       # vanilla, gated, or rnn

# Confidence head
ENABLE_CONFIDENCE_HEAD="--enable-confidence-head"
CONFIDENCE_HEAD_WITH_MARKOV=""   # set to "--confidence-head-with-markov" to enable

# Loss weights
CE_LOSS_ALPHA=1.0
L1_LOSS_ALPHA=0.1
CONFIDENCE_HEAD_ALPHA=0.01
LOSS_DECAY_GAMMA=4.0

# GPU assignments (online training needs separate GPUs for vLLM and training)
VLLM_GPU="4"
TRAIN_GPU="5"
# =======================================

# Step 1: Download regenerated dataset and prepare data
# echo "=== Step 1: Downloading dataset and preparing data ==="
# DATASET_DIR="$OUTPUT_DIR/dataset"
# hf download "$DATASET" "$DATASET_FILE" --repo-type dataset --local-dir "$DATASET_DIR"

python scripts/prepare_data.py \
    --model "$MODEL" \
    --data "$DATASET_FILE" \
    --max-samples "$MAX_SAMPLES" \
    --output "$OUTPUT_DIR" \
    --seq-length "$SEQ_LENGTH"

# Step 2: Launch vLLM server in the background
echo "=== Step 2: Launching vLLM server ==="
CUDA_VISIBLE_DEVICES="$VLLM_GPU" \
ASCEND_RT_VISIBLE_DEVICES="$VLLM_GPU" \
python scripts/launch_vllm.py "$MODEL" \
    --target-layer-ids $TARGET_LAYER_IDS \
    --no-include-last-layer \
    -- --port "$VLLM_PORT" \
       --enforce-eager \
       --gpu-memory-utilization 0.80 \
       --max-model-len 4096 &
VLLM_PID=$!

# Ensure vLLM is cleaned up on exit
cleanup() {
    echo "Stopping vLLM server..."
    kill "$VLLM_PID" 2>/dev/null || true
    wait "$VLLM_PID" 2>/dev/null || true
}
trap cleanup EXIT

echo "Waiting for vLLM server to be ready..."
until curl -sf "http://127.0.0.1:${VLLM_PORT}/health" > /dev/null 2>&1; do
    sleep 2
done
echo "vLLM server ready."

# Additional wait for vLLM to fully warm up (models endpoint may not be ready)
echo "Waiting for vLLM /v1/models endpoint..."
until curl -sf "http://127.0.0.1:${VLLM_PORT}/v1/models" > /dev/null 2>&1; do
    sleep 2
done
echo "vLLM /v1/models endpoint ready."

# Step 3: Train against the live vLLM server
echo "=== Step 3: Training ==="
CUDA_VISIBLE_DEVICES="$TRAIN_GPU" \
ASCEND_RT_VISIBLE_DEVICES="$TRAIN_GPU" \
python \
    scripts/train.py \
    --verifier-name-or-path "$MODEL" \
    --data-path "$OUTPUT_DIR" \
    --vllm-endpoint "http://127.0.0.1:${VLLM_PORT}/v1" \
    --save-path "$OUTPUT_DIR/checkpoints" \
    --draft-vocab-size "$DRAFT_VOCAB_SIZE" \
    --epochs "$EPOCHS" \
    --lr "$LR" \
    --total-seq-len "$SEQ_LENGTH" \
    --speculator-type "$SPECULATOR_TYPE" \
    --block-size "$BLOCK_SIZE" \
    --num-anchors "$NUM_ANCHORS" \
    --num-layers "$NUM_LAYERS" \
    --draft-attn-impl eager 
    --target-layer-ids $TARGET_LAYER_IDS \
    --markov-rank "$MARKOV_RANK" \
    --markov-head-type "$MARKOV_HEAD_TYPE" \
    $ENABLE_CONFIDENCE_HEAD \
    $CONFIDENCE_HEAD_WITH_MARKOV \
    --ce-loss-alpha "$CE_LOSS_ALPHA" \
    --l1-loss-alpha "$L1_LOSS_ALPHA" \
    --confidence-head-alpha "$CONFIDENCE_HEAD_ALPHA" \
    --loss-decay-gamma "$LOSS_DECAY_GAMMA" \
    --on-missing generate \
    --on-generate delete \
    --num-workers 0

echo "Done. Checkpoints saved to $OUTPUT_DIR/checkpoints/"
