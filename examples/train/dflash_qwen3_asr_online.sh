#!/bin/bash
# Online DFlash Training for Qwen3-ASR
#
# Usage: Edit the configuration below, then run from the repository root:
#   bash examples/train/dflash_qwen3_asr_online.sh
#
# For the audio data format and setup, see docs/user_guide/tutorials/qwen3_asr.md.
set -euo pipefail

# ============ Configuration ============
MODEL="Qwen/Qwen3-ASR-0.6B-hf"
DATASET="./qwen3_asr_conversations.jsonl"
MEDIA_ROOT="/data/audio"
OUTPUT_DIR="./output/dflash_qwen3_asr"
VLLM_PORT=8000
SEQ_LENGTH=4096
EPOCHS=2
LR=3e-4

# DFlash-specific parameters
SPECULATOR_TYPE="dflash"
BLOCK_SIZE=8
MAX_ANCHORS=3072
NUM_LAYERS=5
DRAFT_VOCAB_SIZE=32000
TARGET_LAYER_IDS="2 14 25"  # Must match vLLM's auxiliary hidden-state layers

# GPU assignments (online training needs separate GPUs for vLLM and training)
VLLM_GPUS="0"
TRAIN_GPUS="1,2"
NUM_TRAIN_GPUS=2
# =======================================

# Step 1: Launch vLLM for audio rendering and hidden-state extraction
mkdir -p "$OUTPUT_DIR"
CUDA_VISIBLE_DEVICES="$VLLM_GPUS" python scripts/launch_vllm.py train "$MODEL" \
    --hidden-states-path "$OUTPUT_DIR/hidden_states" \
    --target-layer-ids $TARGET_LAYER_IDS \
    --provenance-dir "$OUTPUT_DIR/vllm_provenance" \
    -- \
    --port "$VLLM_PORT" \
    --max-model-len "$((SEQ_LENGTH + 1))" \
    --allowed-local-media-path "$MEDIA_ROOT" &
VLLM_PID=$!

cleanup() {
    kill "$VLLM_PID" 2>/dev/null || true
    wait "$VLLM_PID" 2>/dev/null || true
}
trap cleanup EXIT

ready=0
for ((attempt = 0; attempt < 300; attempt++)); do
    if ! kill -0 "$VLLM_PID" 2>/dev/null; then
        echo "The vLLM launcher exited before becoming ready." >&2
        exit 1
    fi
    if curl --max-time 5 -fsS "http://localhost:${VLLM_PORT}/health" >/dev/null 2>&1; then
        ready=1
        break
    fi
    sleep 2
done
if [[ "$ready" != 1 ]]; then
    echo "Timed out waiting for vLLM on port $VLLM_PORT." >&2
    exit 1
fi

# Step 2: Prepare audio conversations using the live render endpoint
speculators prepare-data \
    --model "$MODEL" \
    --data "$DATASET" \
    --render-endpoint "http://localhost:${VLLM_PORT}" \
    --output "$OUTPUT_DIR/data" \
    --seq-length "$SEQ_LENGTH"

# Step 3: Train DFlash against the live vLLM server
CUDA_VISIBLE_DEVICES="$TRAIN_GPUS" torchrun \
    --standalone --nproc_per_node "$NUM_TRAIN_GPUS" \
    -m speculators.train \
    --verifier-name-or-path "$MODEL" \
    --data-path "$OUTPUT_DIR/data" \
    --hidden-states-path "$OUTPUT_DIR/hidden_states" \
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
    --on-missing generate

echo "Done. Checkpoints saved to $OUTPUT_DIR/checkpoints/"
