#!/bin/bash
# DSpark Training Script for Inkling-Small-NVFP4
#
# Trains a DSpark speculative decoding model for thinkingmachines/Inkling-Small-NVFP4
# (276B MoE, 12B active, 42 layers). Trains from scratch using a regenerated dataset.
#
# Prerequisites:
# - 8x H200 GPUs (4 for vLLM TP=4, 4 for training)
#
# Usage:
#   bash examples/train/dspark_inkling_small_nvfp4.sh

set -euo pipefail

# Ensure hostname resolves
grep -q "$(hostname)" /etc/hosts 2>/dev/null || echo "127.0.0.1 $(hostname)" >> /etc/hosts

export FLASHINFER_DISABLE_VERSION_CHECK=1

LOG_FILE="dspark_inkling_small_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG_FILE") 2>&1

# ============ Configuration ============
MODEL="thinkingmachines/Inkling-Small-NVFP4"
HF_DATASET="hf:orestis-z/Inkling-Small-NVFP4-Regenerated-Collection"
OUTPUT_DIR="./output/dspark_inkling_small"
VLLM_PORT=8000
SEQ_LENGTH=8192
EPOCHS=3
LR=1e-4

# Draft architecture
NUM_LAYERS=6
BLOCK_SIZE=16
DRAFT_VOCAB_SIZE=201024

# Target layer IDs for 42-layer model (evenly spaced through the network).
# The verifier's final hidden state (layer 42) is added automatically by
# launch_vllm.py (--include-last-layer, on by default).
TARGET_LAYER_IDS="2 10 18 26 34 39"

# DSpark-specific parameters
MARKOV_RANK=256
MARKOV_HEAD_TYPE="vanilla"
LOSS_FN='{"ce": 0.1, "tv": 0.9}'
CONFIDENCE_HEAD_ALPHA=1.0
MAX_ANCHORS=1024

# GPU assignments — Inkling-Small-NVFP4 (~171GB) requires TP=4.
VLLM_GPUS="0,1,2,3"
VLLM_TP=4
TRAIN_GPUS="4,5,6,7"
NUM_TRAIN_GPUS=4

# Logging
LOGGER="trackio"
RUN_NAME="dspark-inkling-small"

# =======================================

# Step 1: Prepare data (tokenize + Arrow format)
if [ ! -d "$OUTPUT_DIR/dataset" ] && [ -z "$(ls "$OUTPUT_DIR"/*.arrow 2>/dev/null)" ]; then
    echo "=== Step 1: Preparing data ==="
    /workspace/speculators/.venv/bin/python scripts/prepare_data.py \
        --model "$MODEL" \
        --data "$HF_DATASET" \
        --output "$OUTPUT_DIR" \
        --seq-length "$SEQ_LENGTH" \
        --trust-remote-code
else
    echo "=== Step 1: Prepared data already exists ==="
fi

# Step 2: Launch vLLM server in the background
echo "=== Step 2: Launching vLLM server (TP=$VLLM_TP) ==="
CUDA_VISIBLE_DEVICES="$VLLM_GPUS" /workspace/speculators/.venv/bin/python scripts/launch_vllm.py "$MODEL" \
    --target-layer-ids $TARGET_LAYER_IDS \
    -- --tensor-parallel-size "$VLLM_TP" \
       --port "$VLLM_PORT" \
       --max-model-len $((SEQ_LENGTH + 2)) \
       --trust-remote-code &

VLLM_PID=$!

cleanup() {
    echo "Stopping vLLM server..."
    kill "$VLLM_PID" 2>/dev/null || true
    wait "$VLLM_PID" 2>/dev/null || true
}
trap cleanup EXIT

echo "Waiting for vLLM server to be ready..."
until curl -sf "http://localhost:${VLLM_PORT}/health" > /dev/null 2>&1; do
    sleep 5
done
echo "vLLM server ready."

# Step 3: Train DSpark from scratch
echo "=== Step 3: Training DSpark ==="
CUDA_VISIBLE_DEVICES="$TRAIN_GPUS" /workspace/speculators/.venv/bin/python -m torch.distributed.run \
    --standalone --nproc_per_node "$NUM_TRAIN_GPUS" \
    scripts/train.py \
    --verifier-name-or-path "$MODEL" \
    --speculator-type dspark \
    --data-path "$OUTPUT_DIR" \
    --vllm-endpoint "http://localhost:${VLLM_PORT}/v1" \
    --save-path "$OUTPUT_DIR/checkpoints" \
    --draft-vocab-size "$DRAFT_VOCAB_SIZE" \
    --epochs "$EPOCHS" \
    --lr "$LR" \
    --total-seq-len "$SEQ_LENGTH" \
    --max-anchors "$MAX_ANCHORS" \
    --num-layers "$NUM_LAYERS" \
    --block-size "$BLOCK_SIZE" \
    --target-layer-ids $TARGET_LAYER_IDS \
    --markov-rank "$MARKOV_RANK" \
    --markov-head-type "$MARKOV_HEAD_TYPE" \
    --enable-confidence-head \
    --confidence-head-with-markov \
    --confidence-head-alpha "$CONFIDENCE_HEAD_ALPHA" \
    --loss-fn "$LOSS_FN" \
    --logger "$LOGGER" \
    --run-name "$RUN_NAME" \
    --on-missing generate \
    --on-generate delete \
    --checkpoint-freq 0.1 \
    --trust-remote-code

echo "Done. DSpark checkpoints saved to $OUTPUT_DIR/checkpoints/"
