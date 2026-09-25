#!/bin/bash
# Online multimodal (image+text) DFlash Training Script for Qwen3.5-4B
#
# Runs the full online pipeline: vLLM server launch, multimodal data
# preparation, and training (with hidden states generated on-the-fly from
# the live server).
#
# Usage: Copy this script, modify the configuration variables below, then run:
#   COCO_DIR=/path/to/coco bash examples/train/dflash_qwen3_5_4b_sharegpt4v_online_5k.sh
#
# For a detailed walkthrough, see
# https://docs.vllm.ai/projects/speculators/en/latest/user_guide/tutorials/train/

### Example E2E run for DFlash Qwen3.5-4B on 5k samples from ShareGPT4V ###

# Image rows ride the normal Chat Completions path: the `sharegpt4v_coco`
# dataset preset emits {"type": "image", "path": ...} parts, and vLLM is
# served with --allowed-local-media-path so it can read the local image
# files. The captured verifier hidden states already encode the image
# content; the draft itself never sees pixels.

# The draft config is built plain-rope for DFlash speculators: mrope_section
# (and the coupled partial_rotary_factor) from the verifier's text_config is
# stripped because the draft only consumes the captured verifier hidden
# states and vLLM's DFlash serving path rejects MRoPE draft configs.

# Draft geometry matches the official Qwen/Qwen3.5-4B-DFlash checkpoint:
# 5 draft layers, block_size 16, aux target layers [1, 8, 15, 22, 29]
# (Qwen3.5-4B has 32 text layers), vocab 248320.

set -euo pipefail

# ============ Configuration ============
MODEL="Qwen/Qwen3.5-4B"
DATASET="sharegpt4v_coco"           # requires COCO_DIR with COCO 2017 train images
COCO_DIR="${COCO_DIR:-coco}"        # local image root, served as allowed-local-media-path
OUTPUT_DIR="./output/dflash_qwen3_5_4b_sharegpt4v"
VLLM_PORT=8000
MAX_SAMPLES=5000
SEQ_LENGTH=1024
EPOCHS=5
LR=6e-4

# DFlash-specific parameters
SPECULATOR_TYPE="dflash"
BLOCK_SIZE=16
MAX_ANCHORS=128
NUM_LAYERS=5
DRAFT_VOCAB_SIZE=248320
TARGET_LAYER_IDS="1 8 15 22 29"  # Must match vLLM's eagle_aux_hidden_state_layer_ids

# GPU assignments (online training needs separate GPUs for vLLM and training)
VLLM_GPUS="0,1"
TRAIN_GPUS="2,3"
NUM_TRAIN_GPUS=2
# =======================================

[[ -d "$COCO_DIR" ]] || {
    echo "COCO_DIR does not exist: $COCO_DIR. Download COCO 2017 Train Images" >&2
    echo "from http://images.cocodataset.org/zips/train2017.zip first." >&2
    exit 1
}

# Step 1: Launch vLLM server in the background
echo "=== Step 1: Launching vLLM server ==="
CUDA_VISIBLE_DEVICES="$VLLM_GPUS" python scripts/launch_vllm.py "$MODEL" \
    --target-layer-ids $TARGET_LAYER_IDS \
    --allowed-local-media-path "$COCO_DIR" \
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

# Step 2: Prepare data (conversations are tokenized by the render endpoint)
echo "=== Step 2: Preparing data ==="
python scripts/prepare_data.py \
    --model "$MODEL" \
    --data "$DATASET" \
    --output "$OUTPUT_DIR" \
    --render-endpoint "http://localhost:${VLLM_PORT}" \
    --max-samples "$MAX_SAMPLES" \
    --seq-length "$SEQ_LENGTH"

# Step 3: Train against the live vLLM server
echo "=== Step 3: Training ==="
CUDA_VISIBLE_DEVICES="$TRAIN_GPUS" torchrun \
    --standalone --nproc_per_node "$NUM_TRAIN_GPUS" \
    scripts/train.py \
    --verifier-name-or-path "$MODEL" \
    --data-path "$OUTPUT_DIR" \
    --vllm-endpoint "http://localhost:${VLLM_PORT}/v1" \
    --save-path "$OUTPUT_DIR/checkpoints" \
    --speculator-type "$SPECULATOR_TYPE" \
    --num-layers "$NUM_LAYERS" \
    --block-size "$BLOCK_SIZE" \
    --no-sample-from-anchor \
    --draft-vocab-size "$DRAFT_VOCAB_SIZE" \
    --target-layer-ids $TARGET_LAYER_IDS \
    --total-seq-len "$SEQ_LENGTH" \
    --max-anchors "$MAX_ANCHORS" \
    --loss-fn '{"ce":0.1,"tv":0.9}' \
    --per-position-loss-weight fixed-exp-decay \
    --dflash-decay-gamma 4 \
    --epochs "$EPOCHS" \
    --lr "$LR" \
    --draft-attn-impl simple_flex_attention \
    --sliding-window-non-causal \
    --on-missing generate \
    --on-generate delete

echo "Done. Checkpoints saved to $OUTPUT_DIR/checkpoints/"
