#!/bin/bash
# Online DFlash Training Script -- Best-Practices Recipe
#
# Same pipeline as dflash_qwen3_8b_sharegpt_online_5k.sh (data preparation, vLLM
# server launch, and online training), but using the recipe recommended in
# https://github.com/vllm-project/speculators/issues/979 ("DFlash Training Best
# Practices"): D-PACE per-position loss weighting (with cross-entropy), 5 draft
# layers, and block_size=16. As of this script's introduction these are also
# train.py's defaults for --speculator-type dflash, so passing them explicitly
# below is redundant with the CLI -- it's done anyway so this script is a
# self-contained reference for the full recipe regardless of what the defaults
# happen to be at the time you read it.
#
# Uses UltraChat instead of ShareGPT (the classic example's dataset) since it's
# one of the two datasets the recipe in #979 was actually validated on
# (alongside Magpie, which isn't wired into scripts/prepare_data.py's built-in
# dataset registry).
#
# Usage: Copy this script, modify the configuration variables below, then run:
#   bash examples/train/dflash_qwen3_8b_ultrachat_online_5k_bestpractices.sh
#
# For a detailed walkthrough, see
# https://docs.vllm.ai/projects/speculators/en/latest/user_guide/tutorials/train/
# and the recipe rationale/ablations at
# https://github.com/vllm-project/speculators/issues/979

### Example E2E run for DFlash Qwen3-8B on 5k samples from UltraChat ###

# Note: With just 5k samples, the model performance will not be very good, however there
# are enough samples to verify that the pipeline is working correctly and that the model
# is learning something. This is a good sanity check when creating a drafter for a new
# target model. Note also that block_size=16 (double the classic example's block_size=8)
# means more speculative positions per anchor, so this run trains and generates fewer,
# larger blocks -- expect different timing and per-position acceptance numbers than the
# classic example; run your own eval afterward (see examples/evaluate/) rather than
# relying on any specific numbers here, since the right comparison depends on your data
# and hardware.

set -euo pipefail

# ============ Configuration ============
MODEL="Qwen/Qwen3-8B"
DATASET="ultrachat"               # sharegpt, ultrachat, or path to custom data
OUTPUT_DIR="./output/dflash_qwen3_8b_ultrachat_bestpractices"
VLLM_PORT=8000
MAX_SAMPLES=5000
SEQ_LENGTH=8192
EPOCHS=5
LR=3e-4

# DFlash-specific parameters (best-practices recipe from RFC #979)
SPECULATOR_TYPE="dflash"
BLOCK_SIZE=16
MAX_ANCHORS=3072
NUM_LAYERS=5
PER_POSITION_LOSS_WEIGHT="dpace"  # requires --loss-fn ce
LOSS_FN="ce"
DRAFT_VOCAB_SIZE=32000
TARGET_LAYER_IDS="2 18 33"  # Must match vLLM's eagle_aux_hidden_state_layer_ids

# GPU assignments (online training needs separate GPUs for vLLM and training)
VLLM_GPUS="0,1"
TRAIN_GPUS="2,3"
NUM_TRAIN_GPUS=2
# =======================================

# Step 1: Prepare data
echo "=== Step 1: Preparing data ==="
python scripts/prepare_data.py \
    --model "$MODEL" \
    --data "$DATASET" \
    --output "$OUTPUT_DIR" \
    --max-samples "$MAX_SAMPLES" \
    --seq-length "$SEQ_LENGTH"

# Step 2: Launch vLLM server in the background
echo "=== Step 2: Launching vLLM server ==="
CUDA_VISIBLE_DEVICES="$VLLM_GPUS" python scripts/launch_vllm.py "$MODEL" \
    --target-layer-ids $TARGET_LAYER_IDS \
    -- --data-parallel-size 2 --port "$VLLM_PORT" &
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
    --per-position-loss-weight "$PER_POSITION_LOSS_WEIGHT" \
    --loss-fn "$LOSS_FN" \
    --target-layer-ids $TARGET_LAYER_IDS \
    --on-missing generate \
    --on-generate delete

echo "Done. Checkpoints saved to $OUTPUT_DIR/checkpoints/"
