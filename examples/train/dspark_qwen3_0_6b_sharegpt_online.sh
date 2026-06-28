#!/bin/bash
# Online DSpark Training Script
#
# Runs the full online DSpark training pipeline: data preparation, vLLM server
# launch, and training with hidden states generated on-the-fly from the live
# server. DSpark extends DFlash with a sequential Markov head (intra-block token
# dependency) and a confidence head (per-position acceptance prediction), so the
# pipeline is identical to the DFlash one plus a few DSpark-specific flags.
#
# Unlike DeepSeek's DeepSpec reference (which precomputes a very large offline
# target-hidden-state cache), this trains DSpark fully online.
#
# Usage: Copy this script, modify the configuration variables below, then run:
#   bash examples/train/dspark_qwen3_0_6b_sharegpt_online.sh
#
# See the DFlash online tutorial for a detailed walkthrough of the shared steps:
# https://docs.vllm.ai/projects/speculators/en/latest/user_guide/tutorials/train_dflash_online/

### Example E2E run for DSpark Qwen3-0.6B on 5k samples from ShareGPT ###

# Note: With just 5k samples the absolute acceptance length will be low; the run
# is meant to verify the pipeline works and that DSpark is learning (in
# particular that the Markov head lifts later-position accuracy over plain DFlash
# and that the confidence head calibrates).

set -euo pipefail

# ============ Configuration ============
MODEL="Qwen/Qwen3-0.6B"
DATASET="sharegpt"                # sharegpt, ultrachat, or path to custom data
OUTPUT_DIR="./output/dspark_qwen3_0_6b_sharegpt"
VLLM_PORT=8000
MAX_SAMPLES=5000
SEQ_LENGTH=4096
EPOCHS=5
LR=3e-4

# DSpark-specific parameters
SPECULATOR_TYPE="dspark"
BLOCK_SIZE=8
MAX_ANCHORS=3072
NUM_LAYERS=3
DRAFT_VOCAB_SIZE=32000
TARGET_LAYER_IDS="2 14 25"  # Must match vLLM's eagle_aux_hidden_state_layer_ids

# Markov + confidence head settings
MARKOV_RANK=256
MARKOV_HEAD_TYPE="vanilla"   # vanilla | gated | rnn
CE_LOSS_ALPHA=0.1
L1_LOSS_ALPHA=0.9
CONFIDENCE_HEAD_ALPHA=1.0

# GPU assignments (online training needs separate GPUs for vLLM and training).
# Pick currently-idle devices (check `nvidia-smi`).
VLLM_GPUS="0"
TRAIN_GPUS="1"
NUM_TRAIN_GPUS=1
# =======================================

# Step 1: Prepare data (also writes token_freq.pt used to build the draft vocab)
echo "=== Step 1: Preparing data ==="
python scripts/prepare_data.py \
    --model "$MODEL" \
    --data "$DATASET" \
    --output "$OUTPUT_DIR" \
    --max-samples "$MAX_SAMPLES" \
    --seq-length "$SEQ_LENGTH"

# Step 2: Launch vLLM server in the background (exposes the aux hidden states)
echo "=== Step 2: Launching vLLM server ==="
CUDA_VISIBLE_DEVICES="$VLLM_GPUS" python scripts/launch_vllm.py "$MODEL" \
    --target-layer-ids $TARGET_LAYER_IDS \
    -- --port "$VLLM_PORT" &
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

# Step 3: Train DSpark against the live vLLM server
echo "=== Step 3: Training ==="
CUDA_VISIBLE_DEVICES="$TRAIN_GPUS" torchrun \
    --standalone --nproc_per_node "$NUM_TRAIN_GPUS" \
    scripts/train.py \
    --verifier-name-or-path "$MODEL" \
    --data-path "$OUTPUT_DIR" \
    --vllm-endpoint "http://localhost:${VLLM_PORT}/v1" \
    --save-path "$OUTPUT_DIR/checkpoints" \
    --draft-vocab-size "$DRAFT_VOCAB_SIZE" \
    --epochs "$EPOCHS" \
    --lr "$LR" \
    --total-seq-len "$SEQ_LENGTH" \
    --speculator-type "$SPECULATOR_TYPE" \
    --block-size "$BLOCK_SIZE" \
    --max-anchors "$MAX_ANCHORS" \
    --num-layers "$NUM_LAYERS" \
    --target-layer-ids $TARGET_LAYER_IDS \
    --markov-rank "$MARKOV_RANK" \
    --markov-head-type "$MARKOV_HEAD_TYPE" \
    --enable-confidence-head \
    --confidence-head-with-markov \
    --ce-loss-alpha "$CE_LOSS_ALPHA" \
    --l1-loss-alpha "$L1_LOSS_ALPHA" \
    --confidence-head-alpha "$CONFIDENCE_HEAD_ALPHA" \
    --on-missing generate \
    --on-generate delete

echo "Done. Checkpoints saved to $OUTPUT_DIR/checkpoints/"
